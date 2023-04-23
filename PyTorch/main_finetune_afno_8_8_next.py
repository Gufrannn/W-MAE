import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
# todo manxin need to delete when using distributed-train or command line to run
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image
import timm
from tqdm import tqdm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc_finetune as misc
from util.misc_finetune import NativeScalerWithGradNormCount as NativeScaler

import models_mae_afno_8_8_finetune_next
from engine_finetune_afno_8_8_next import train_one_epoch, valid_one_epoch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import sys

# manxin todo for h5 file loading
from utils.data_loader_multifiles import get_data_loader
from utils.YParams import YParams
from collections import OrderedDict

# manxin todo save log info
class Logger(object):
    def __init__(self, logFile="Default.log"):
        self.terminal = sys.stdout
        self.log = open(logFile, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    # parser.add_argument('--batch_size', default=64, type=int,
    #                     help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    # parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=(720,1440), type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--output_dir',
                        default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
# manxin todo dataloader to shuffle or seed or fixed (for re-implement)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # manxin load the h5 data file

    parser.add_argument('--dt', default=1, type=int,
                        help='how many timesteps ahead the model will predict')
    parser.add_argument('--n_history', default=0, type=int,
                        help='how many previous timesteps to consider')
    parser.add_argument('--in_channels', default=[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10, 11, 12, 13, 14, 15, 16, 17, 18, 19], type=type([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10, 11, 12, 13, 14, 15, 16, 17, 18, 19]))
    parser.add_argument('--out_channels', default=[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10, 11, 12, 13, 14, 15, 16, 17, 18, 19], type=type([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10, 11, 12, 13, 14, 15, 16, 17, 18, 19]))

    parser.add_argument('--n_in_channels', default=20, type=int)
    parser.add_argument('--n_out_channels', default=20, type=int)
    parser.add_argument('--crop_size_x', default=None, type=bool)
    parser.add_argument('--crop_size_y', default=None, type=bool)

    parser.add_argument('--roll', default=False, type=bool)
    parser.add_argument('--two_step_training', default=False, type=bool)
    parser.add_argument('--orography', default=False, type=bool)
    # parser.add_argument('--precip', default=False, type=bool)
    parser.add_argument('--num_data_workers', default=10, type=int)
    parser.add_argument('--normalization', default='zscore', type=str)
    parser.add_argument('--add_grid', default=False, type=bool)

    # manxin todo
    parser.add_argument('--run_num', default='', type=str)
    parser.add_argument('--run_mode', default='pretraining', type=str)
    parser.add_argument('--save_dir', default='', type=str)
    parser.add_argument('--pretrained_ckpt_path', default='', type=str)

    parser.add_argument("--yaml_config", default='', type=str)
    parser.add_argument("--config", default='afno_backbone', type=str)

    return parser


def main(args):
    # manxin todo alternative command
    print(args.batch_size)
    print(args.output_dir)
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    log_writer = None #manxin
    train_data_loader_h5, train_dataset_h5, train_sampler_h5 = get_data_loader(args, args.train_data_path_h5,
                                                                               args.distributed,
                                                                              train=True)
    valid_data_loader_h5, valid_dataset_h5, valid_sampler_h5 = get_data_loader(args, args.valid_data_path_h5,
                                                                              args.distributed,
                                                                              train=True)

    # model = models_mae_afno_8_8_finetune.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, img_size= args.img_size, patch_size = args.patch_size)
    model = models_mae_afno_8_8_finetune_next.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, img_size= args.img_size, patch_size = args.patch_size)
    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    # manxin
    print(str(args.save_dir) + "/" + "log.log")
    # sys.stdout = Logger(str(args.save_dir) + "/" + "log.log")
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr_new is None:  # only base_lr is specified
        args.lr_new = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr_new * 256 / eff_batch_size), flush=True)
    print("actual lr: %.2e" % args.lr_new)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr_new, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    print(optimizer)

    # misc._model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    # misc.load_model_v1(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, ckpt_path=args.checkpoint_path)
    use_pretrained_model = False
    if args.resuming:
        ckpt_path = args.checkpoint_path
    else:
        ckpt_path = args.pretrained_ckpt_path
        args.resuming = True
        use_pretrained_model = True
    misc.load_model_v1(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler,
                       ckpt_path=ckpt_path)
    # if use_pretrained_model == True:
    #     args.start_epoch = 0

    print(f"Start training for {args.epochs} epochs from epoch: {args.start_epoch}")
    start_time = time.time()
    best_loss = 20.
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_data_loader_h5.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, train_data_loader_h5,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        valid_loss = valid_one_epoch(
            model, valid_data_loader_h5,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        print("valid_loss:".format(valid_loss))
        vaild_loss_value = valid_loss.pop()
        if float(vaild_loss_value) < best_loss:
            best_loss = vaild_loss_value
            if args.save_dir:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, best_loss_yn=True)
        # valid_loss_list = []
        # valid_loss_list.append(best_loss)
        # valid_loss = valid_one_epoch(
        #     model, valid_data_loader_h5,
        #     optimizer, device, epoch, loss_scaler,
        #     log_writer=log_writer,
        #     args=args
        # )
        # print("valid_loss:".format(valid_loss))
        # if valid_loss < valid_loss_list[len(valid_loss_list)-1]:
        #     if int(len(valid_loss_list))<=50:
        #         valid_loss_list.append(valid_loss)
        #
        #     elif (len(valid_loss_list))>50:
        #         valid_loss_list[len(valid_loss_list) - 1] = valid_loss
        #         valid_loss_list.sort()
        #         best_loss_yn = True
        #     if args.save_dir:
        #         misc.save_model(
        #             args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
        #             loss_scaler=loss_scaler, epoch=epoch, best_loss_yn=True)


        if args.save_dir:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'valid_loss': vaild_loss_value}

        if args.save_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.save_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if args.save_dir:
        if log_writer is not None:
            log_writer.flush()
        log_stats_time = {**{f'Training time': total_time_str},
                     'epoch': epoch, }
        with open(os.path.join(args.save_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats_time) + "\n")


def restore_checkpoint(args, model, optimizer, loss_scaler, ckpt_path):
   """ We intentionally require a checkpoint_dir to be passed
       in order to allow Ray Tune to use this function """
   checkpoint = torch.load(ckpt_path, map_location='cuda:{}'.format(misc.get_rank())) # manxin todo ??misc.get_rank()
   try:
       model.load_state_dict(checkpoint['model'])
       loss_scaler.load_state_dict(checkpoint['scaler'])
   except:
       new_state_dict = OrderedDict()
       for key, val in checkpoint['model'].items():
           name = key[7:]
           new_state_dict[name] = val
       model.load_state_dict(new_state_dict)
   # args.iters = checkpoint['iters'] # manxin todo
   args.startEpoch = checkpoint['epoch']
   if args.resuming:
       #restore checkpoint is used for finetuning as well as resuming. If finetuning (i.e., not resuming), restore checkpoint does not load optimizer state, instead uses config specified lr.
       optimizer.load_state_dict(checkpoint['optimizer'])
   return args, model, optimizer, loss_scaler

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args.run_mode = 'finetuning'
    args.run_num = 'f_afno_8_8_next_614' # 'p_1'
    args.pretrained_ckpt_path = '/home/lichangyu/data/codes/mae/mae-main/output_dir/p_afno_8_8_next_527/pretraining/ckpt/checkpoint-888.pth' # ckpt45
    args.output_dir = '/home/lichangyu/data/codes/mae/mae-main/output_dir'
    args.yaml_config = './config/AFNO_afno_8_8_finetune_next.yaml'

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    try:
        os.makedirs(args.output_dir + "/" + str(args.run_num) + "/" + str(args.run_mode))
    except:
        pass
    save_dir = args.output_dir + "/" + str(args.run_num) + "/" + str(args.run_mode)
    args.save_dir = save_dir

    params = YParams(
        os.path.abspath(args.yaml_config), args.config)
    params['output_dir'] = args.output_dir
    params['save_dir'] = args.save_dir
    params['run_num'] = args.run_num
    params['run_mode'] = args.run_mode

    if args.save_dir:
        try:
            os.makedirs(args.save_dir + "/" + "ckpt" )
        except:
            pass

    params['checkpoint_folder'] =  os.path.join(save_dir, 'ckpt')
    params['checkpoint_path'] =  os.path.join(save_dir, 'ckpt/checkpoint-cur.pth')  # manxin todo aaa

    print(params.checkpoint_path)
    params['resuming'] = True if os.path.isfile(params.checkpoint_path) else False
    params['pretrained_ckpt_path'] = args.pretrained_ckpt_path
    params['pretrained'] =  True if os.path.isfile(params.pretrained_ckpt_path) else False
    # sys.stdout = Logger(str(args.save_dir) + "/" + "log.log") #manxin
    sys.stdout = Logger(logFile=str(args.save_dir) + "/" + "log.log")
    main(params)

