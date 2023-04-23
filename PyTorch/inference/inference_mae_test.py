import os
import sys
import time
import numpy as np
import argparse

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
from numpy.core.numeric import False_
import h5py
import torch
import torchvision
from torchvision.utils import save_image
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel
import logging
from utils import logging_utils
from utils.weighted_acc_rmse import weighted_rmse_torch_channels, weighted_acc_torch_channels, \
    unweighted_acc_torch_channels, weighted_acc_masked_torch_channels

logging_utils.config_logger()
from utils.YParams import YParams
from utils.data_loader_multifiles import get_data_loader
# from networks.afnonet import AFNONet # manxin todo
import wandb
import matplotlib.pyplot as plt
import glob
from datetime import datetime
from tqdm import tqdm

#todo manxin for mae model paras
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import torch.backends.cudnn as cudnn
import models_mae_16_16_next_finetune
# import models_mae_16_16_next_finetune_twoStep

from einops import rearrange, repeat
# from goto import with_goto

# os.environ['CUDA_VISIBLE_DEVICES'] = '4'


fld = "z500"  # diff flds have diff decor times and hence differnt ics
if fld == "z500" or fld == "2m_temperature" or fld == "t850":
    DECORRELATION_TIME = 36  # 9 days (36) for z500, 2 (8 steps) days for u10, v10
else:
    DECORRELATION_TIME = 8  # 9 days (36) for z500, 2 (8 steps) days for u10, v10
idxes = {"u10": 0, "z500": 14, "2m_temperature": 2, "v10": 1, "t850": 5}

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


def gaussian_perturb(x, level=0.01, device=0):
    noise = level * torch.randn(x.shape).to(device, dtype=torch.float)
    return (x + noise)


def load_model(model, params, checkpoint_file):
    model.zero_grad()
    checkpoint_fname = checkpoint_file
    checkpoint = torch.load(checkpoint_fname)
    try:
        new_state_dict = OrderedDict()
        for key, val in checkpoint['model_state'].items():
            name = key[7:]
            if name != 'ged':
                new_state_dict[name] = val
        model.load_state_dict(new_state_dict)
    except:
        model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model

def load_model_state_mae(model, params, checkpoint_file):
    model.zero_grad()
    checkpoint_fname = checkpoint_file
    checkpoint = torch.load(checkpoint_fname, map_location='cpu')
    # model.load_state_dict(checkpoint['model'])
    try:
        new_state_dict = OrderedDict()
        for key, val in checkpoint['model'].items():
            # name = key[7:]
            name = key
            if name != 'ged':
                new_state_dict[name] = val
        model.load_state_dict(new_state_dict)
        # model.load_state_dict(new_state_dict, strict=False)
    except:
        model.load_state_dict(checkpoint['model'])
        # model.load_state_dict(checkpoint['model'], strict=False)
    print("Resume checkpoint %s" % checkpoint_fname)

    model.eval()
    return model

def load_model_mae(args, ckpt_path):
    # misc.init_distributed_mode(args)
    print("*************")
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    device = torch.device(args.device)
    print("*************")
    # fix the seed for reproducibility
    # seed = args.seed + misc.get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    cudnn.benchmark = True
    log_writer = None  # manxin
    # train_data_loader_h5, train_dataset_h5, train_sampler_h5 = get_data_loader(args, args.train_data_path_h5,
    #                                                                            args.distributed,
    #                                                                            train=True)
    norm_pix_loss = False # todo manxin changed
    model = models_mae_16_16_next_finetune.__dict__[args.model](norm_pix_loss=norm_pix_loss, img_size=args.img_size,patch_size=args.patch_size)
    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    # print(str(args.save_dir) + "/" + "inference.log")
    sys.stdout = Logger("/home/lichangyu/data/codes/mae/mae-main/output_dir/p_qkv_16_16_next_finetune_577/pretraining/inference_mae_0407_z500.log")


    # sys.stdout = Logger(str(args.save_dir) + "/" + "log.log")
    # eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    # if args.lr_new is None:  # only base_lr is specified
    #     args.lr_new = args.blr * eff_batch_size / 256
    # print("base lr: %.2e" % (args.lr_new * 256 / eff_batch_size), flush=True)
    # print("actual lr: %.2e" % args.lr_new)
    # print("accumulate grad iterations: %d" % args.accum_iter)
    # print("effective batch size: %d" % eff_batch_size)
    # following timm: set wd as 0 for bias and norm layers
    # param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    # optimizer = torch.optim.AdamW(param_groups, lr=args.lr_new, betas=(0.9, 0.95))
    # loss_scaler = NativeScaler()
    # print(optimizer)

    # misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    load_model_state_mae(model=model_without_ddp, params=args, checkpoint_file=ckpt_path)

    print("Start inference!!!!!!!!!!!!!!!")
    start_time = time.time()
    return model


def downsample(x, scale=0.125):
    return torch.nn.functional.interpolate(x, scale_factor=scale, mode='bilinear')


def setup(params):
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    # get data loader
    logging.info('Inference data path !!!!!!!!!!')
    valid_data_loader, valid_dataset = get_data_loader(params, params.inf_data_path, dist.is_initialized(), train=False)
    img_shape_x = valid_dataset.img_shape_x
    img_shape_y = valid_dataset.img_shape_y
    params.img_shape_x = img_shape_x
    params.img_shape_y = img_shape_y
    if params.log_to_screen:
        logging.info('Loading trained model checkpoint from {}'.format(params['best_checkpoint_path']))

    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    n_in_channels = len(in_channels)
    n_out_channels = len(out_channels)

    if params["orography"]:
        params['N_in_channels'] = n_in_channels + 1
    else:
        params['N_in_channels'] = n_in_channels
    params['N_out_channels'] = n_out_channels
    params.means = np.load(params.global_means_path)[0, out_channels]  # needed to standardize wind data
    params.stds = np.load(params.global_stds_path)[0, out_channels]

    # todo manxin ==> load the mae model
    checkpoint_file = params['best_checkpoint_path']
    print(checkpoint_file)

    if params.nettype == 'mae_16_16':
        model = load_model_mae(params, checkpoint_file)
        # model = AFNONet(params).to(device)
    else:
        raise Exception("not implemented")

    # todo manxin ==> load the afno model
    # if params.nettype == 'afno':
    #     model = AFNONet(params).to(device)
    # else:
    #     raise Exception("not implemented")

    # checkpoint_file = params['best_checkpoint_path']
    # model = load_model(model, params, checkpoint_file)
    # model = model.to(device)
    # load the validation data
    files_paths = glob.glob(params.inf_data_path + "/*.h5")
    files_paths.sort()
    # which year
    yr = 0
    if params.log_to_screen:
        logging.info('Loading inference data')
        logging.info('Inference data from {}'.format(files_paths[yr]))

    valid_data_full = h5py.File(files_paths[yr], 'r')['fields']

    return valid_data_full, model


def autoregressive_inference(params, ic, valid_data_full, model, ff, fff):
    ic = int(ic)
    # initialize global variables
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    exp_dir = params['experiment_dir']
    dt = int(params.dt)
    prediction_length = int(params.prediction_length / dt)
    n_history = params.n_history
    img_shape_x = params.img_shape_x
    img_shape_y = params.img_shape_y
    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    n_in_channels = len(in_channels)
    n_out_channels = len(out_channels)
    means = params.means
    stds = params.stds

    # initialize memory for image sequences and RMSE/ACC
    valid_loss = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    acc = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)

    # compute metrics in a coarse resolution too if params.interp is nonzero
    valid_loss_coarse = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    acc_coarse = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    acc_coarse_unweighted = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)

    acc_unweighted = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    seq_real = torch.zeros((prediction_length, n_in_channels, img_shape_x, img_shape_y)).to(device, dtype=torch.float)
    seq_pred = torch.zeros((prediction_length, n_in_channels, img_shape_x, img_shape_y)).to(device, dtype=torch.float)

    acc_land = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    acc_sea = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    if params.masked_acc:
        maskarray = torch.as_tensor(np.load(params.maskpath)[0:720]).to(device, dtype=torch.float)

    valid_data = valid_data_full[ic:(ic + prediction_length * dt + n_history * dt):dt, in_channels,
                 0:720]  # extract valid data from first year
    # standardize
    valid_data = (valid_data - means) / stds
    valid_data = torch.as_tensor(valid_data).to(device, dtype=torch.float)

    # load time means
    if not params.use_daily_climatology:
        m = torch.as_tensor((np.load(params.time_means_path)[0][out_channels] - means) / stds)[:,
            0:img_shape_x]  # climatology
        m = torch.unsqueeze(m, 0)
    else:
        # use daily clim like weyn et al. (different from rasp)
        dc_path = params.dc_path
        with h5py.File(dc_path, 'r') as f:
            dc = f['time_means_daily'][ic:ic + prediction_length * dt:dt]  # 1460,21,721,1440
        m = torch.as_tensor((dc[:, out_channels, 0:img_shape_x, :] - means) / stds)

    m = m.to(device, dtype=torch.float)
    if params.interp > 0:
        m_coarse = downsample(m, scale=params.interp)

    std = torch.as_tensor(stds[:, 0, 0]).to(device, dtype=torch.float)

    orography = params.orography
    orography_path = params.orography_path
    if orography:
        orog = torch.as_tensor(
            np.expand_dims(np.expand_dims(h5py.File(orography_path, 'r')['orog'][0:720], axis=0), axis=0)).to(device,
                                                                                                              dtype=torch.float)
        logging.info("orography loaded; shape:{}".format(orog.shape))

    # autoregressive inference
    if params.log_to_screen:
        logging.info('Begin autoregressive inference')

    with torch.no_grad():
        for i in range(valid_data.shape[0]):
            if i == 0:  # start of sequence
                first = valid_data[0:n_history + 1]
                future = valid_data[n_history + 1]
                for h in range(n_history + 1):
                    seq_real[h] = first[h * n_in_channels: (h + 1) * n_in_channels][
                                  0:n_out_channels]  # extract history from 1st
                    seq_pred[h] = seq_real[h]
                if params.perturb:
                    first = gaussian_perturb(first, level=params.n_level, device=device)  # perturb the ic
                if orography:
                    # todo manxin
                    list_tmp = [torch.cat((first, orog), axis=1),torch.cat((first, orog), axis=1)]
                    # list_tmp[0] = torch.cat((first, orog), axis=1)
                    # list_tmp[1] = torch.cat((first, orog), axis=1)
                    _, pred, _ = model(list_tmp, mask_ratio = 0.)
                    pred = rearrang_v1(params, pred)
                    future_pred = pred
                    # future_pred = model(torch.cat((first, orog), axis=1))
                else:
                    list_tmp = [first, first]
                    # list_tmp[0] = first
                    # list_tmp[1] = first
                    _, pred, _ = model(list_tmp, mask_ratio = 0.)
                    pred = rearrang_v1(params, pred)
                    future_pred = pred
                    # future_pred = model(first)
            else:
                if i < prediction_length - 1:
                    future = valid_data[n_history + i + 1]
                if orography:
                    list_tmp = [torch.cat((future_pred, orog), axis=1), torch.cat((future_pred, orog), axis=1)]
                    # list_tmp[0] = torch.cat((future_pred, orog), axis=1)
                    # list_tmp[1] = torch.cat((future_pred, orog), axis=1)
                    _, pred, _ = model(list_tmp, mask_ratio = 0.)
                    pred = rearrang_v1(params, pred)
                    future_pred = pred
                    # future_pred = model(torch.cat((future_pred, orog), axis=1))  # autoregressive step
                else:
                    list_tmp = [future_pred, future_pred]
                    # list_tmp[0] = future_pred
                    # list_tmp[1] = future_pred
                    _, pred, _ = model(list_tmp, mask_ratio = 0.)
                    pred = rearrang_v1(params, pred)
                    future_pred = pred
                    # future_pred = model(future_pred)  # autoregressive step

            if i < prediction_length - 1:  # not on the last step
                seq_pred[n_history + i + 1] = future_pred
                seq_real[n_history + i + 1] = future
                history_stack = seq_pred[i + 1:i + 2 + n_history]

            future_pred = history_stack

            # Compute metrics
            if params.use_daily_climatology:
                clim = m[i:i + 1]
                if params.interp > 0:
                    clim_coarse = m_coarse[i:i + 1]
            else:
                clim = m
                if params.interp > 0:
                    clim_coarse = m_coarse

            pred = torch.unsqueeze(seq_pred[i], 0)
            tar = torch.unsqueeze(seq_real[i], 0)
            valid_loss[i] = weighted_rmse_torch_channels(pred, tar) * std
            acc[i] = weighted_acc_torch_channels(pred - clim, tar - clim)
            acc_unweighted[i] = unweighted_acc_torch_channels(pred - clim, tar - clim)

            if params.masked_acc:
                acc_land[i] = weighted_acc_masked_torch_channels(pred - clim, tar - clim, maskarray)
                acc_sea[i] = weighted_acc_masked_torch_channels(pred - clim, tar - clim, 1 - maskarray)

            if params.interp > 0:
                pred = downsample(pred, scale=params.interp)
                tar = downsample(tar, scale=params.interp)
                valid_loss_coarse[i] = weighted_rmse_torch_channels(pred, tar) * std
                acc_coarse[i] = weighted_acc_torch_channels(pred - clim_coarse, tar - clim_coarse)
                acc_coarse_unweighted[i] = unweighted_acc_torch_channels(pred - clim_coarse, tar - clim_coarse)

            if params.log_to_screen:
                idx = idxes[fld]
                logging.info('Predicted timestep {} of {}. {} RMS Error: {}, ACC: {}'.format(i, prediction_length, fld,
                                                                                             valid_loss[i, idx],
                                                                                             acc[i, idx]))
                ff.write('Predicted timestep {} of {}. {} RMS Error: {}, ACC: {} \n'.format(i, prediction_length, fld,
                                                                                             valid_loss[i, idx],
                                                                                             acc[i, idx]))
                fff.write('Predicted timestep {}, RMS Error: {}, ACC: {} \n'.format(i,
                                                                                            valid_loss[i, idx],
                                                                                            acc[i, idx]))
                if params.interp > 0:
                    logging.info(
                        '[COARSE] Predicted timestep {} of {}. {} RMS Error: {}, ACC: {} \n'.format(i, prediction_length,
                                                                                                 fld, valid_loss_coarse[
                                                                                                     i, idx],
                                                                                                 acc_coarse[i, idx]))
                    ff.write('[COARSE] Predicted timestep {} of {}. {} RMS Error: {}, ACC: {} \n'.format(i, prediction_length,
                                                                                                 fld, valid_loss_coarse[
                                                                                                     i, idx],
                                                                                                 acc_coarse[i, idx]))

                ff.flush()
                fff.flush()

    seq_real = seq_real.cpu().numpy()
    seq_pred = seq_pred.cpu().numpy()
    valid_loss = valid_loss.cpu().numpy()
    acc = acc.cpu().numpy()
    acc_unweighted = acc_unweighted.cpu().numpy()
    acc_coarse = acc_coarse.cpu().numpy()
    acc_coarse_unweighted = acc_coarse_unweighted.cpu().numpy()
    valid_loss_coarse = valid_loss_coarse.cpu().numpy()
    acc_land = acc_land.cpu().numpy()
    acc_sea = acc_sea.cpu().numpy()

    return (
    np.expand_dims(seq_real[n_history:], 0), np.expand_dims(seq_pred[n_history:], 0), np.expand_dims(valid_loss, 0),
    np.expand_dims(acc, 0),
    np.expand_dims(acc_unweighted, 0), np.expand_dims(valid_loss_coarse, 0), np.expand_dims(acc_coarse, 0),
    np.expand_dims(acc_coarse_unweighted, 0),
    np.expand_dims(acc_land, 0),
    np.expand_dims(acc_sea, 0))

# todo manxin
def rearrang_v1(args, x):
    # print("x.shape::::::::::::::::::::")
    # print(x.shape)
    # print(args.img_size)
    # print(args.input_size)
    # print("x.shape::::::::::::::::::::")
    B = x.shape[0]
    in_channels = np.array(args.in_channels)
    embed_dim = int(len(in_channels) * (args.patch_size[0] * args.patch_size[1]))
    h = int(args.img_size[0] // (args.patch_size[0]))
    w = int(args.img_size[1] // (args.patch_size[1]))
    x = x.reshape(B, h, w, embed_dim)
    x = rearrange(
        x,
        "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
        h = h,
        w = w,
        p1=args.patch_size[0],
        p2=args.patch_size[1],
    )
    return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--run_num", default='p_qkv_16_16_next_finetune_twoStep', type=str)
    # parser.add_argument("--yaml_config", default='./config/AFNO_16_16_next_finetune_twoStep.yaml', type=str)
    parser.add_argument("--run_num", default='p_qkv_16_16_next_finetune_577', type=str)
    parser.add_argument("--yaml_config", default='./config/AFNO_16_16_next_finetune.yaml', type=str)
    parser.add_argument("--config", default='afno_backbone', type=str)  # full_field
    parser.add_argument("--use_daily_climatology", action='store_true')
    parser.add_argument("--vis", action='store_true')
    # parser.add_argument("--override_dir", default='/home/lichangyu/data/codes/mae/mae-main/output_dir/p_qkv_16_16_next_finetune_twoStep/finetuning/ckpt/', type=str,
    #                     help='Path to store inference outputs; must also set --weights arg')
    parser.add_argument("--override_dir",
                        default='/home/lichangyu/data/codes/mae/mae-main/output_dir/p_qkv_16_16_next_finetune_577/pretraining/ckpt/',
                        type=str,
                        help='Path to store inference outputs; must also set --weights arg')
    parser.add_argument("--interp", default=0, type=float)
    parser.add_argument("--weights", default='', type=str,
                        help='Path to model weights, for use with override_dir option')

    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params['world_size'] = 1
    params['interp'] = args.interp
    params['use_daily_climatology'] = args.use_daily_climatology
    params['global_batch_size'] = params.batch_size

    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    vis = args.vis

    # Set up directory
    for iii in range(721, 722): #718 725 717 679
        # if iii%5 == 0:
        #     continue
        ff = open(r"/home/lichangyu/data/codes/mae/mae-main/output_dir/p_qkv_16_16_next_finetune_577/pretraining/inference/inference_" + "z500" + ".txt", "a", encoding="UTF-8")
        fff = open(r"/home/lichangyu/data/codes/mae/mae-main/output_dir/p_qkv_16_16_next_finetune_577/pretraining/inference/inference_mean_" + "z500" + ".txt", "a", encoding="UTF-8")

        # ff = open(r"/home/lichangyu/data/codes/mae/mae-main/output_dir/p_qkv_16_16_next_finetune_twoStep/finetuning/inference_for_valid.txt", "a", encoding="UTF-8")
        ff.write(str(iii)+'\n')
        fff.write(str(iii) + '\n')

        flag_mx = 'true'
        args.weights = '/home/lichangyu/data/codes/mae/mae-main/output_dir/p_qkv_16_16_next_finetune_577/pretraining/ckpt/checkpoint-' + str(iii) + '.pth'
        # args.weights = '/home/lichangyu/data/codes/mae/mae-main/output_dir/p_qkv_16_16_next_finetune_twoStep/finetuning/ckpt/checkpoint-' + str(iii) + '.pth'
        if args.override_dir is not None:
            assert args.weights is not None, 'Must set --weights argument if using --override_dir'
            expDir = args.override_dir
        else:
            assert args.weights is None, 'Cannot use --weights argument without also using --override_dir'
            expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))

        if not os.path.isdir(expDir):
            os.makedirs(expDir)

        params['experiment_dir'] = os.path.abspath(expDir)
        params['best_checkpoint_path'] = args.weights if args.override_dir is not None else os.path.join(expDir,
                                                                                                         'training_checkpoints/best_ckpt.tar')
        params['resuming'] = True
        params['local_rank'] = 0

        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'inference_out_2.log'))
        logging_utils.log_versions()
        params.log()

        n_ics = params['n_initial_conditions']

        if fld == "z500" or fld == "t850":
            n_samples_per_year = 1336
        else:
            n_samples_per_year = 1460

        logging.info("SSSSSSSStep!!!!! Initial condition {} of {}".format(fld, n_samples_per_year))  # manxin

        if params["ics_type"] == 'default':
            num_samples = n_samples_per_year - params.prediction_length
            stop = num_samples
            ics = np.arange(0, stop, DECORRELATION_TIME)
            if vis:  # visualization for just the first ic (or any ic)
                ics = [0]
            n_ics = len(ics)
        elif params["ics_type"] == "datetime":
            date_strings = params["date_strings"]
            ics = []
            if params.perturb:  # for perturbations use a single date and create n_ics perturbations
                n_ics = params["n_perturbations"]
                date = date_strings[0]
                date_obj = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
                day_of_year = date_obj.timetuple().tm_yday - 1
                hour_of_day = date_obj.timetuple().tm_hour
                hours_since_jan_01_epoch = 24 * day_of_year + hour_of_day
                for ii in range(n_ics):
                    ics.append(int(hours_since_jan_01_epoch / 6))
            else:
                for date in date_strings:
                    date_obj = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
                    day_of_year = date_obj.timetuple().tm_yday - 1
                    hour_of_day = date_obj.timetuple().tm_hour
                    hours_since_jan_01_epoch = 24 * day_of_year + hour_of_day
                    ics.append(int(hours_since_jan_01_epoch / 6))
            n_ics = len(ics)

        logging.info("Inference for {} initial conditions".format(n_ics))
        try:
            autoregressive_inference_filetag = params["inference_file_tag"]
        except:
            autoregressive_inference_filetag = ""

        if params.interp > 0:
            autoregressive_inference_filetag = "_coarse"

        autoregressive_inference_filetag += "_" + fld + ""
        if vis:
            autoregressive_inference_filetag += "_vis"
        # get data and models
        params["nettype"] = "mae_16_16"  # todo manxin note
        valid_data_full, model = setup(params)

        # initialize lists for image sequences and RMSE/ACC
        valid_loss = []
        valid_loss_coarse = []
        acc_unweighted = []
        acc = []
        acc_coarse = []
        acc_coarse_unweighted = []
        seq_pred = []
        seq_real = []
        acc_land = []
        acc_sea = []

        # run autoregressive inference for multiple initial conditions
        for i, ic in enumerate(ics):
            logging.info("Initial condition {} of {}".format(i + 1, n_ics))
            ff.write("Initial condition {} of {} \n".format(i + 1, n_ics))
            fff.write("Initial condition {} of {} \n".format(i + 1, n_ics))
            # if i == 0:
            sr, sp, vl, a, au, vc, ac, acu, accland, accsea = autoregressive_inference(params, ic, valid_data_full,
                                                                                           model, ff, fff)
            # else:
            #     flag_mx = 'fasle'
            #     break
            if i == 0 or len(valid_loss) == 0:
                seq_real = sr
                seq_pred = sp
                valid_loss = vl
                valid_loss_coarse = vc
                acc = a
                acc_coarse = ac
                acc_coarse_unweighted = acu
                acc_unweighted = au
                acc_land = accland
                acc_sea = accsea
            else:
                #        seq_real = np.concatenate((seq_real, sr), 0)
                #        seq_pred = np.concatenate((seq_pred, sp), 0)
                valid_loss = np.concatenate((valid_loss, vl), 0)
                valid_loss_coarse = np.concatenate((valid_loss_coarse, vc), 0)
                acc = np.concatenate((acc, a), 0)
                acc_coarse = np.concatenate((acc_coarse, ac), 0)
                acc_coarse_unweighted = np.concatenate((acc_coarse_unweighted, acu), 0)
                acc_unweighted = np.concatenate((acc_unweighted, au), 0)
                acc_land = np.concatenate((acc_land, accland), 0)
                acc_sea = np.concatenate((acc_sea, accsea), 0)

        if flag_mx == 'fasle':
            continue
        prediction_length = seq_real[0].shape[0]
        n_out_channels = seq_real[0].shape[1]
        img_shape_x = seq_real[0].shape[2]
        img_shape_y = seq_real[0].shape[3]

        # save predictions and loss
        if params.log_to_screen:
            logging.info("Saving files at {}".format(os.path.join(params['experiment_dir'],
                                                                  'autoregressive_predictions' + autoregressive_inference_filetag + '.h5')))
        with h5py.File(os.path.join(params['experiment_dir'],
                                    'autoregressive_predictions' + autoregressive_inference_filetag + '.h5'), 'a') as f:
            if vis:
                try:
                    f.create_dataset("ground_truth", data=seq_real,
                                     shape=(n_ics, prediction_length, n_out_channels, img_shape_x, img_shape_y),
                                     dtype=np.float32)
                except:
                    del f["ground_truth"]
                    f.create_dataset("ground_truth", data=seq_real,
                                     shape=(n_ics, prediction_length, n_out_channels, img_shape_x, img_shape_y),
                                     dtype=np.float32)
                    f["ground_truth"][...] = seq_real

                try:
                    f.create_dataset("predicted", data=seq_pred,
                                     shape=(n_ics, prediction_length, n_out_channels, img_shape_x, img_shape_y),
                                     dtype=np.float32)
                except:
                    del f["predicted"]
                    f.create_dataset("predicted", data=seq_pred,
                                     shape=(n_ics, prediction_length, n_out_channels, img_shape_x, img_shape_y),
                                     dtype=np.float32)
                    f["predicted"][...] = seq_pred

            if params.masked_acc:
                try:
                    f.create_dataset("acc_land",
                                     data=acc_land)  # , shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
                except:
                    del f["acc_land"]
                    f.create_dataset("acc_land",
                                     data=acc_land)  # , shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
                    f["acc_land"][...] = acc_land

                try:
                    f.create_dataset("acc_sea",
                                     data=acc_sea)  # , shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
                except:
                    del f["acc_sea"]
                    f.create_dataset("acc_sea",
                                     data=acc_sea)  # , shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
                    f["acc_sea"][...] = acc_sea

            try:
                f.create_dataset("rmse", data=valid_loss, shape=(n_ics, prediction_length, n_out_channels),
                                 dtype=np.float32)
            except:
                del f["rmse"]
                f.create_dataset("rmse", data=valid_loss, shape=(n_ics, prediction_length, n_out_channels),
                                 dtype=np.float32)
                f["rmse"][...] = valid_loss

            try:
                f.create_dataset("acc", data=acc, shape=(n_ics, prediction_length, n_out_channels), dtype=np.float32)
            except:
                del f["acc"]
                f.create_dataset("acc", data=acc, shape=(n_ics, prediction_length, n_out_channels), dtype=np.float32)
                f["acc"][...] = acc

            try:
                f.create_dataset("rmse_coarse", data=valid_loss_coarse,
                                 shape=(n_ics, prediction_length, n_out_channels),
                                 dtype=np.float32)
            except:
                del f["rmse_coarse"]
                f.create_dataset("rmse_coarse", data=valid_loss_coarse,
                                 shape=(n_ics, prediction_length, n_out_channels),
                                 dtype=np.float32)
                f["rmse_coarse"][...] = valid_loss_coarse

            try:
                f.create_dataset("acc_coarse", data=acc_coarse, shape=(n_ics, prediction_length, n_out_channels),
                                 dtype=np.float32)
            except:
                del f["acc_coarse"]
                f.create_dataset("acc_coarse", data=acc_coarse, shape=(n_ics, prediction_length, n_out_channels),
                                 dtype=np.float32)
                f["acc_coarse"][...] = acc_coarse

            try:
                f.create_dataset("acc_unweighted", data=acc_unweighted,
                                 shape=(n_ics, prediction_length, n_out_channels),
                                 dtype=np.float32)
            except:
                del f["acc_unweighted"]
                f.create_dataset("acc_unweighted", data=acc_unweighted,
                                 shape=(n_ics, prediction_length, n_out_channels),
                                 dtype=np.float32)
                f["acc_unweighted"][...] = acc_unweighted

            try:
                f.create_dataset("acc_coarse_unweighted", data=acc_coarse_unweighted,
                                 shape=(n_ics, prediction_length, n_out_channels), dtype=np.float32)
            except:
                del f["acc_coarse_unweighted"]
                f.create_dataset("acc_coarse_unweighted", data=acc_coarse_unweighted,
                                 shape=(n_ics, prediction_length, n_out_channels), dtype=np.float32)
                f["acc_coarse_unweighted"][...] = acc_coarse_unweighted

    f.close()
