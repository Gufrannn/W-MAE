import argparse
import numpy as np
import os
import time
import logging
from utils import logging_utils
logging_utils.config_logger()
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict
import einops
from tqdm import tqdm
import mindspore
import mindspore.context as context
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
from mindspore.communication.management import init, get_rank, get_group_size, get_local_rank
from mindspore.train.callback._callback import _handle_loss
mindspore.dataset.config.set_numa_enable(False)
import x2ms_adapter
from utils.img_utils import save_image
from utils.data_loader_multifiles import get_data_loader
from utils.YParams import YParams
import pandas as pd
import traceback

class ValueReduce(mindspore.nn.Cell):
    """
    from mindspore.train.callback._early_stop import ValueReduce
    Reduces the tensor data across all devices, all devices will get the same final result.
    For more details, please refer to :class:`mindspore.ops.AllReduce`.
    """
    def __init__(self):
        super(ValueReduce, self).__init__()
        self.allreduce = mindspore.ops.AllReduce(mindspore.ops.ReduceOp.SUM)

    def construct(self, x):
        return self.allreduce(x)

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    
    parser.add_argument('--batch_size', default=0, type=int,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    parser.add_argument('--n_in_channels', default=20, type=int)
    parser.add_argument('--n_out_channels', default=20, type=int)
    
    parser.add_argument('--model', default='mae_vit_base_patch16', 
                        choices=['mae_vit_base_patch16', 'mae_vit_large_patch16', 'mae_vit_huge_patch14'])

    parser.add_argument("--yaml_config", default='./config/AFNO.yaml', type=str)
    parser.add_argument("--config", default='mae_backbone', type=str)
    parser.add_argument("--resume_checkpoint_path", default='', type=str)
    parser.add_argument("--model_wind_path", default='', type=str)
    parser.add_argument("--run_num", default='0', type=str)
    parser.add_argument("--lr", default=-1.0, type=float)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    
    return parser.parse_args()

class SelfCallback(mindspore.train.callback.Callback):
    def __init__(self, model, params, eval_dataloader):
        super(SelfCallback, self).__init__()
        self.model = model
        self.eval_dataloader = eval_dataloader
        self.eval_data_size = eval_dataloader.get_dataset_size()
        self.eval_iterator = eval_dataloader.create_tuple_iterator( num_epochs=params.max_epochs * 2 + 2 )
        self.params = params
        self.best_valid_loss = np.inf
        self.best_valid_ep = -1
        self.training_logs = {"train_loss":[], "valid_loss":[]}
        self._reduce = ValueReduce()
        if params.mae_pretrain:
            self.label_img_idx = 0  # ['img', 'ids_keep', 'mask', 'ids_restore']
        else:
            self.label_img_idx = 1  # ['prev', 'future']

        
    def get_lr(self, optimizer):
        """
        The optimizer calls this interface to get the learning rate for the current step. User-defined optimizers based
        on :class:`mindspore.nn.Optimizer` can also call this interface before updating the parameters.

        Returns:
            float, the learning rate of current step.
        """
        lr = optimizer.learning_rate
        if optimizer.dynamic_lr:
            if optimizer.is_group_lr:
                lr = ()
                for learning_rate in optimizer.learning_rate:
                    current_dynamic_lr = learning_rate(optimizer.global_step).reshape(())
                    lr += (current_dynamic_lr,)
            else:
                lr = optimizer.learning_rate(optimizer.global_step).reshape(())
        return lr.asnumpy()
    
    def rearrang_v1(self, params, x):
        B = x.shape[0]   # (2, 4050, 5120)
        embed_dim = int(params.N_in_channels * (params.patch_size[0] * params.patch_size[1]))
        h = int(params.img_size[0] // (params.patch_size[0]))
        w = int(params.img_size[1] // (params.patch_size[1]))
        x = x.reshape(B, h, w, embed_dim)
        x = einops.rearrange(
            x,
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            h = h,
            w = w,
            p1=params.patch_size[0],
            p2=params.patch_size[1],
        )
        return x
    
    def epoch_begin(self, run_context):
        self.ep_start_time_ms = time.time()
    
    def evaluate(self, ):
        raise NotImplementedError
        
    def validate_one_epoch(self, run_context, save_for_the_best=False):
        cb_params = run_context.original_args()
        current_epoch = cb_params.cur_epoch_num
        model = cb_params.network
        is_distributed = cb_params.parallel_mode != "stand_alone"
        world_rank = get_rank() if is_distributed else 0
        model.set_train(mode=False)
        
        tqdm_datasize = min(32, self.eval_data_size)
        valid_bar = tqdm(self.eval_iterator, desc=f"Valid@{current_epoch}", total = tqdm_datasize)
        
        valid_loss = np.zeros(1)
        valid_steps = np.zeros(1)
        
        save_perfix = "best_{:0>2d}".format(world_rank) if save_for_the_best else "rank{:0>2d}_ep{:0>6d}".format(world_rank, current_epoch)
        
        
        for i, data in enumerate(valid_bar, 0):
            if i >= tqdm_datasize:
                break
            loss, pred, mask = model(*data)
            
            valid_loss += loss.asnumpy() / self.params.batch_size
            valid_steps += 1.
            
            if current_epoch % 10==0 or self.params['debug'] or save_for_the_best:
                try:
                    os.makedirs(self.params['experiment_dir'] + f"/img/{i}/")
                except:
                    None
                
                if not self.params.precip_mode:
                    pred = self.rearrang_v1(self.params, pred.asnumpy())
                
                true = data[self.label_img_idx].asnumpy()
                save_image(pred[0][0], self.params['experiment_dir'] + f"/img/{i}/pred_{save_perfix}.png")
                save_image(true[0][0], self.params['experiment_dir'] + f"/img/{i}/true_{save_perfix}.png")
                # save numpy data: too big
                if save_for_the_best:
                    with open(params['experiment_dir'] + f"/img/{i}/pred_{save_perfix}.npy", 'wb') as f:
                        np.save(f, pred)
                    with open(params['experiment_dir'] + f"/img/{i}/true_{save_perfix}.npy", 'wb') as f:
                        np.save(f, true)
            del loss, pred, mask

        if is_distributed:
            try:
                valid_loss = self._reduce( mindspore.Tensor(valid_loss.astype(np.float32)) ).asnumpy()
                valid_steps = self._reduce( mindspore.Tensor(valid_steps.astype(np.float32)) ).asnumpy()
            except Exception as e:
                traceback.print_exc()
                logging.info("Fail to call all_reduce op at valid...")

        valid_loss = valid_loss / valid_steps

        # download buffers
        valid_loss_numpy = valid_loss#.asnumpy()
                    
        # avoid
        self.model.set_train(mode=True)

        log = {"valid_loss": valid_loss_numpy[0]}
        return log

        
    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        current_epoch = cb_params.cur_epoch_num
        step_num = cb_params.cur_step_num
        train_loss = _handle_loss(cb_params.net_outputs) / self.params.batch_size
        model = cb_params.network
        now_lr = self.get_lr(cb_params.optimizer)
        _is_best = False

        self.ep_total_time_min = (time.time() - self.ep_start_time_ms) / 60
        valid_logs = self.validate_one_epoch(run_context, False)
        
        self.training_logs['train_loss'].append( train_loss )
        self.training_logs['valid_loss'].append( valid_logs["valid_loss"] )
        
        logging.info("epoch: {}/{} lr: {:.8f} train_loss: {:.6f} valid_loss: {:.6f} time: {:.3f}min".format(
            current_epoch, self.params.max_epochs, now_lr, 
            train_loss, valid_logs["valid_loss"], self.ep_total_time_min))
        
        # save model
        if self.params.save_checkpoint:
            mindspore.save_checkpoint(self.model, self.params.checkpoint_path) 
            if valid_logs["valid_loss"] < self.best_valid_loss:   # best
                logging.info("best model save...... from {} to {}".format(self.best_valid_loss, valid_logs["valid_loss"]))
                self.best_valid_loss, self.best_valid_ep = valid_logs["valid_loss"], current_epoch
                mindspore.save_checkpoint(self.model, self.params.best_checkpoint_path) 
                _is_best = True
        if _is_best and current_epoch>=50:
            logging.info("rerun for the best...")
            self.validate_one_epoch(run_context, True)
        
    def end(self, run_context):
        pd.DataFrame(self.training_logs).to_csv("{}/training_logs.csv".format(params['experiment_dir']))
        logging.info("best epoch at {}: loss: {}".format(self.best_valid_ep, self.best_valid_loss))
        
        
class Trainer():
    def is_initialized_parallel(self):
        return mindspore.context.get_auto_parallel_context("parallel_mode") != mindspore.context.ParallelMode.STAND_ALONE

    def count_parameters(self, net):
        """Count number of parameters in the network
        Args:
            net (mindspore.nn.Cell): Mindspore network instance
        Returns:
            total_params (int): Total number of trainable params
        """
        total_params = 0
        for param in net.trainable_params():
            total_params += np.prod(param.shape)
        return total_params
    
    def switch_off_grad(self, model):
        _count, _count_stop = 0, 0
        for param in model.get_parameters():
            if param.requires_grad==True:
                _count_stop+=1
            param.requires_grad = False
            _count += 1
        logging.info(f"switch_off_grad all:{_count}, stop:{_count_stop}")

    def __init__(self, params, world_rank):
        self.params = params
        self.world_rank = world_rank

        # init dataloader
        logging.info('rank %d, begin data loader init'%world_rank)
        self.train_data_loader, self.train_dataset, self.train_sampler = get_data_loader(params, params.train_data_path, self.is_initialized_parallel(), train=True)
        self.train_data_loader = self.train_data_loader.batch(params.batch_size, drop_remainder=True)

        self.valid_data_loader, self.valid_dataset = get_data_loader(params, params.valid_data_path, self.is_initialized_parallel(), train=False)
        self.valid_data_loader = self.valid_data_loader.batch(params.batch_size, drop_remainder=False)
        
        if params.mae_pretrain:
            logging.warning("MAE pretrain")
        if params.precip_mode:
            logging.warning("Start precip training")
        if params.mae_pretrain and params.precip_mode:
            raise Exception
            
        logging.info('rank %d, data loader initialized'%world_rank)
        logging.info(f"CHECK: dataset num: train: {len(self.train_data_loader)}; valid: {len(self.valid_data_loader)}")

        params.crop_size_x = self.valid_dataset.crop_size_x
        params.crop_size_y = self.valid_dataset.crop_size_y
        params.img_shape_x = self.valid_dataset.img_shape_x
        params.img_shape_y = self.valid_dataset.img_shape_y
        params.patchify_blocks_num = self.valid_dataset.patchify_blocks_num


        if params.precip_mode:
            if 'model_wind_path' not in params:
                raise Exception("no backbone model weights specified")
            from networks import afno_one_step
            logging.info("from networks import afno_one_step as models_mae")
            self.model_wind = afno_one_step.__dict__[args.model](
                norm_pix_loss=args.norm_pix_loss, 
                img_size= tuple(params.img_size), 
                patch_size = tuple(params.patch_size),
                patchify_blocks_num = params.patchify_blocks_num,
            )
            logging.info(f"precip predict step one loading model from: {params.model_wind_path}")
            mindspore.load_param_into_net(self.model_wind, mindspore.load_checkpoint(params.model_wind_path), strict_load=True)
            self.switch_off_grad(self.model_wind) # no backprop through the wind model
            self.model_wind.set_train(False)
            from networks import precip
            self.model = precip.__dict__[args.model](
                norm_pix_loss=args.norm_pix_loss, 
                img_size= tuple(params.img_size), 
                patch_size = tuple(params.patch_size),
                patchify_blocks_num = params.patchify_blocks_num,
            )
            logging.info(f"precip predict step two loading backbone model from: {params.model_wind_path}")
            from utils import custom_model_load
            custom_model_load.load_param_into_net(self.model, mindspore.load_checkpoint(params.model_wind_path), strict_load=False)
        else:
            if params.mae_pretrain:
                from networks import mae as models_mae
                logging.info("from networks import mae as models_mae")
            else:
                from networks import afno_one_step as models_mae
                logging.info("from networks import afno_one_step as models_mae")
            self.model = models_mae.__dict__[args.model](
                norm_pix_loss=args.norm_pix_loss, 
                img_size= tuple(params.img_size), 
                patch_size = tuple(params.patch_size),
                patchify_blocks_num = params.patchify_blocks_num
            )  
        
        if self.params.enable_nhwc:
            raise Exception

        learning_rate = mindspore.nn.CosineDecayLR(min_lr=0., max_lr=params.lr, 
                            decay_steps=self.train_data_loader.get_dataset_size()*params.max_epochs)
        logging.info("learning_rate = params.lr")

        self.optimizer = mindspore.nn.AdamWeightDecay(params=self.model.trainable_params(), learning_rate=learning_rate)

        self.iters = 0
        self.startEpoch = 0
        if params.resuming:
            logging.info("Loading checkpoint %s"%params.resume_checkpoint_path)
            mindspore.load_param_into_net(self.model, mindspore.load_checkpoint(params.resume_checkpoint_path), strict_load=True)
            logging.warning("unable to resume startEpoch and optimizer")

        self.epoch = self.startEpoch
        if params.log_to_screen:
            logging.info("Number of trainable model parameters: {}".format(self.count_parameters(self.model)))
        
        loss_scale_manager = mindspore.FixedLossScaleManager(loss_scale=128.0, drop_overflow_update=True)
        self.model_wrap_cell = mindspore.Model(self.model, loss_fn=None, optimizer=self.optimizer, 
                                metrics=None, eval_network=None, eval_indexes=None, 
                                amp_level="O2", boost_level="O0", loss_scale_manager=loss_scale_manager)
        
    def train(self):
        if self.params.log_to_screen:
            logging.info("Starting Training Loop...")
        
        self_cb = SelfCallback(self.model, self.params, self.valid_data_loader)
        self.model_wrap_cell.train(self.params.max_epochs, self.train_data_loader, callbacks=[self_cb])

if __name__ == '__main__':
    args = get_args_parser()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    if args.distributed:
        params['num_data_workers'] = 1
    elif "ma-user" in os.path.realpath(__file__):
        logging.warning("NOTEBOOK ENV")
        
    if args.batch_size!=0:
        params['batch_size'] = args.batch_size
    expDir = params['exp_dir']
    params['debug'] = args.debug
    params['mask_ratio'] = args.mask_ratio
    if params['debug']:
        params['max_epochs'] = 4 
        for i in range(5):
            logging.warning("DEBUG!!!!!!")

    if args.resume_checkpoint_path != '':
        logging.info(f"WARNING: resume training to {args.resume_checkpoint_path}")
        params.resume_checkpoint_path = args.resume_checkpoint_path
        params.resuming = True
    if args.model_wind_path != '':
        logging.info(f"WARNING: resume training & update resume_checkpoint_path from {params.model_wind_path} to {args.model_wind_path}")
        params.model_wind_path = args.model_wind_path
    elif params.precip_mode:
        raise Exception
    else:
        pass
    if args.lr>0:
        logging.info(f"update lr from {params.lr} to {args.lr}")
        params.lr = args.lr

    ##############################################
    # distributed
    ##############################################
    distributed = args.distributed
    device_num = 1
    world_rank = 0
    local_rank = 0
    if distributed:
        logging.info("distributed learning")
        init()
        parallel_mode = mindspore.ParallelMode.DATA_PARALLEL
        device_num = get_group_size()
        world_rank = get_rank()
        local_rank = int(os.getenv("LOCAL_RANK", '0'))
    else:
        parallel_mode = mindspore.ParallelMode.STAND_ALONE
    mindspore.reset_auto_parallel_context()
    mindspore.set_auto_parallel_context(device_num=device_num,
                                parallel_mode=parallel_mode,
                                parameter_broadcast=True,
                                gradients_mean=True)


    ##############################################
    # Set up directory
    ##############################################
    exp_dir = params['exp_dir']
    expDir = os.path.join(exp_dir, args.config, "{}_{}".format(args.run_num,time.strftime("%Y%m%d%H")) )
    notes = f"{local_rank}|{world_rank}"

    logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, f'out_{world_rank}.log'), notes=notes)
    logging_utils.log_versions()
    x2ms_adapter.tensor_api.log(params)

    if not os.path.isdir(expDir):
        try:
            os.makedirs(expDir)
        except:
            pass
    if not os.path.isdir(os.path.join(expDir, 'training_checkpoints/')):
        try:
            os.makedirs(os.path.join(expDir, 'training_checkpoints/'))
        except:
            pass

    params['experiment_dir'] = os.path.abspath(expDir)
    params['checkpoint_path'] = os.path.join(expDir, f'training_checkpoints/latest_ckpt_rank{world_rank}.ckpt')
    params['best_checkpoint_path'] = os.path.join(expDir, f'training_checkpoints/best_ckpt_rank{world_rank}.ckpt')

    if world_rank == 0:
        hparams = ruamelDict()
        yaml = YAML()
        for key, value in params.params.items():
            hparams[str(key)] = str(value)
        with open(os.path.join(expDir, 'hyperparams.yaml'), 'w') as hpfile:
            yaml.dump(hparams,  hpfile )

    params['in_channels'] = np.array(params['in_channels'])
    params['out_channels'] = np.array(params['out_channels'])
    if params.orography:
        params['N_in_channels'] = len(params['in_channels']) +1
    else:
        params['N_in_channels'] = len(params['in_channels'])

    params['N_out_channels'] = len(params['out_channels'])


    ##############################################
    # init & train model
    ##############################################
    trainer = Trainer(params, world_rank)
    trainer.train()
    logging.info('DONE ---- rank %d'%world_rank)


