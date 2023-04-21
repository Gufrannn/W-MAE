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
from utils.weighted_acc_rmse import weighted_rmse_torch_channels, weighted_acc_torch_channels, unlog_tp_torch, \
    top_quantiles_error_torch

logging_utils.config_logger()
from utils.YParams import YParams
from utils.data_loader_multifiles_precip import get_data_loader
from afnonet import AFNONet, PrecipNet
import wandb
import matplotlib.pyplot as plt
import glob
from datetime import datetime

# todo manxin for mae model paras
from tqdm import tqdm
import util.misc_pre as misc
from util.misc_pre import NativeScalerWithGradNormCount as NativeScaler
# import util.misc as misc
# from util.misc import NativeScalerWithGradNormCount as NativeScaler
import torch.backends.cudnn as cudnn
# import models_mae_16_16_next_finetune
# import models_mae_16_16_next_finetune_twoStep
import models_mae_16_16_next_precip
import models_mae_16_16_next_precip_outchannel
from engine_pretrain_16_16_next_precip import train_one_epoch
from einops import rearrange, repeat

os.environ['CUDA_VISIBLE_DEVICES'] = '6'


DECORRELATION_TIME = 8  # 2 days for preicp


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


def load_model_mae(args, ckpt_path, flag_wind=False):
    # misc.init_distributed_mode(args)
    print("*************")
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    device = torch.device(args.device)
    print("*************")

    cudnn.benchmark = True
    log_writer = None  # manxin
    # train_data_loader_h5, train_dataset_h5, train_sampler_h5 = get_data_loader(args, args.train_data_path_h5,
    #                                                                            args.distributed,
    #                                                                            train=True)
    norm_pix_loss = False  # todo manxin changed
    precip_model = PrecipNet().to(device)

    if flag_wind == True:
        model = models_mae_16_16_next_precip.__dict__[args.model](norm_pix_loss=norm_pix_loss, img_size=args.img_size,
                                                                patch_size=args.patch_size)
    else:
        model = models_mae_16_16_next_precip_outchannel.__dict__[args.model](precip_model=precip_model, norm_pix_loss=args.norm_pix_loss, img_size= args.img_size, patch_size = args.patch_size)

    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    # print(str(args.save_dir) + "/" + "inference.log")
    sys.stdout = Logger("./inference_mae_16_16_precip.log")

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
    valid_data_loader, valid_dataset = get_data_loader(params, params.inf_data_path, dist.is_initialized(), train=False)
    img_shape_x = valid_dataset.img_shape_x
    img_shape_y = valid_dataset.img_shape_y
    params.img_shape_x = img_shape_x
    params.img_shape_y = img_shape_y
    if params.log_to_screen:
        logging.info('Loading trained model checkpoint from {}'.format(params['best_checkpoint_path']))

    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.in_channels)  # for the backbone model, will be reset later
    n_in_channels = len(in_channels)
    n_out_channels = len(out_channels)
    if params["orography"]:
        params['N_in_channels'] = n_in_channels + 1
    else:
        params['N_in_channels'] = n_in_channels
    params['N_out_channels'] = n_out_channels
    params.means = np.load(params.global_means_path)[0, out_channels]  # needed to standardize wind data
    params.stds = np.load(params.global_stds_path)[0, out_channels]

    # # load wind model
    # if params.nettype_wind == 'afno':
    #   model_wind = AFNONet(params).to(device)
    # if 'model_wind_path' not in params:
    #   raise Exception("no backbone model weights specified")
    # checkpoint_file  = params['model_wind_path']
    # model_wind = load_model(model_wind, params, checkpoint_file)
    # model_wind = model_wind.to(device)

    # todo manxin ==> load the mae model
    checkpoint_file = params['best_checkpoint_path_precip']
    checkpoint_file_precip = params['best_checkpoint_path']
    print(checkpoint_file)

    if params.nettype == 'mae_16_16':
        model_wind = load_model_mae(params, checkpoint_file, flag_wind=True)
        model_wind = model_wind.to(device)

        # model = AFNONet(params).to(device)
    else:
        raise Exception("not implemented")
    # todo manxin ==> load the mae model

    # reset channels for precip
    params['N_out_channels'] = len(params['out_channels'])
    # load the model
    if params.nettype == 'mae_16_16':
        model = load_model_mae(params, checkpoint_file_precip, flag_wind=False)
        model = model.to(device)
    else:
        raise Exception("not implemented")

    # model = load_model_mae(params, checkpoint_file_precip)
    # # checkpoint_file  = params['best_checkpoint_path']
    # # model = load_model(model, params, checkpoint_file)
    # model = model.to(device)

    # load the validation data
    files_paths = glob.glob(params.inf_data_path + "/*.h5")
    files_paths.sort()
    # which year
    yr = 0
    if params.log_to_screen:
        logging.info('Loading validation data')
        logging.info('Validation data from {}'.format(files_paths[yr]))

    valid_data_full = h5py.File(files_paths[yr], 'r')['fields']

    # precip paths
    path = params.precip + '/out_of_sample'
    precip_paths = glob.glob(path + "/*.h5")
    precip_paths.sort()
    if params.log_to_screen:
        logging.info('Loading validation precip data')
        logging.info('Validation data from {}'.format(precip_paths[0]))
    valid_data_tp_full = h5py.File(precip_paths[0], 'r')['tp']
    return valid_data_full, valid_data_tp_full, model_wind, model


def autoregressive_inference(params, ic, valid_data_full, valid_data_tp_full, model_wind, model, ff):
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

    # initialize memory for image sequences and RMSE/ACC, tqe for precip
    valid_loss = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    acc = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    acc_unweighted = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    tqe = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)

    # wind seqs
    seq_real = torch.zeros((prediction_length, n_in_channels, img_shape_x, img_shape_y)).to(device, dtype=torch.float)
    seq_pred = torch.zeros((prediction_length, n_in_channels, img_shape_x, img_shape_y)).to(device, dtype=torch.float)
    # precip sequences
    seq_real_tp = torch.zeros((prediction_length, n_out_channels, img_shape_x, img_shape_y)).to(device,
                                                                                                dtype=torch.float)
    seq_pred_tp = torch.zeros((prediction_length, n_out_channels, img_shape_x, img_shape_y)).to(device,
                                                                                                dtype=torch.float)

    valid_data = valid_data_full[ic:(ic + prediction_length * dt + n_history * dt):dt, in_channels,
                 0:720]  # extract valid data from first year
    # standardize
    valid_data = (valid_data - means) / stds
    valid_data = torch.as_tensor(valid_data).to(device, dtype=torch.float)

    len_ic = prediction_length * dt
    valid_data_tp = valid_data_tp_full[ic:(ic + prediction_length * dt):dt, 0:720].reshape(len_ic, n_out_channels, 720,
                                                                                           img_shape_y)  # extract valid data from first year
    # log normalize
    eps = params.precip_eps
    valid_data_tp = np.log1p(valid_data_tp / eps)
    valid_data_tp = torch.as_tensor(valid_data_tp).to(device, dtype=torch.float)

    m = torch.as_tensor(np.load(params.time_means_path_tp)[0][out_channels])[:, 0:img_shape_x]  # climatology
    m = torch.unsqueeze(m, 0)
    m = m.to(device, dtype=torch.float)

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
                first_tp = valid_data_tp[0:1]
                future = valid_data[n_history + 1]
                future_tp = valid_data_tp[1]
                for h in range(n_history + 1):
                    seq_real[h] = first[h * n_in_channels:(h + 1) * n_in_channels][
                                  0:n_in_channels]  # extract history from 1st
                    seq_pred[h] = seq_real[h]
                seq_real_tp[0] = unlog_tp_torch(first_tp)
                seq_pred_tp[0] = unlog_tp_torch(first_tp)
                if params.perturb:
                    first = gaussian_perturb(first, level=params.n_level, device=device)  # perturb the ic
                if orography:
                    future_pred = model_wind(torch.cat((first, orog), axis=1))
                else:
                    list_tmp = [first, future]
                    # print("0000000000")
                    # print(first.shape)
                    # print("0000000011111110")
                    # print(future.shape)
                    future_pred, _ = model_wind(list_tmp, mask_ratio=0.)
                    future_pred = rearrang_v1(params, future_pred)
                list_tmp_precip = [future_pred, first_tp]
                # print("000000222")
                # print(future_pred.shape)
                # print("000000333")
                # print(first_tp.shape)
                _, future_pred_tp = model(list_tmp_precip, mask_ratio=0., precip_flag=True)
                # print("000000444")
                # print(future_pred_tp.shape)
            else:
                if i < prediction_length - 1:
                    future = valid_data[n_history + i + 1]
                    future_tp = valid_data_tp[i + 1]
                if orography:
                    future_pred = model_wind(torch.cat((future_pred, orog), axis=1))  # autoregressive step
                else:
                    list_tmp = [future_pred, future]
                    # print("000000555")
                    # print(future_pred.shape)
                    # print("000000666")
                    # print(future.shape)
                    future_pred, _ = model_wind(list_tmp, mask_ratio=0.)  # autoregressive step
                    future_pred = rearrang_v1(params, future_pred)
                    if i != 40:
                        future_tp = torch.unsqueeze(future_tp, 1)
                    list_tmp_precip = [future_pred, future_tp]
                    # print("000000777")
                    # print(future_pred.shape)
                    # print("000000888")
                    # print(future_tp.shape)
                    # print("000000999")
                    # print(i)
                    _, future_pred_tp = model(list_tmp_precip, mask_ratio=0., precip_flag=True)  # tp diagnosis
                    # print("000000888")
                    # print(prediction_length)
                    # print(future_pred_tp.shape)
            if i < prediction_length - 1:  # not on the last step
                seq_pred[n_history + i + 1] = future_pred
                seq_real[n_history + i + 1] = future
                seq_pred_tp[i + 1] = unlog_tp_torch(
                    future_pred_tp)  # this predicts 6-12 precip: 0 -> 6 (afno) -> 6-12 precip
                seq_real_tp[i + 1] = unlog_tp_torch(future_tp)  # which is the i+1th validation data
                # collect history
                history_stack = seq_pred[i + 1:i + 2 + n_history]

            # ic for next wind step
            future_pred = history_stack

            pred = torch.unsqueeze(seq_pred_tp[i], 0)
            tar = torch.unsqueeze(seq_real_tp[i], 0)
            valid_loss[i] = weighted_rmse_torch_channels(pred, tar)
            acc[i] = weighted_acc_torch_channels(pred - m, tar - m)
            tqe[i] = top_quantiles_error_torch(pred, tar)

            if params.log_to_screen:
                logging.info(
                    'Timestep {} of {}. TP RMS Error: {}, ACC: {}'.format((i), prediction_length, valid_loss[i, 0],
                                                                          acc[i, 0]))
                ff.write('Timestep {} of {}. TP RMS Error: {}, ACC: {}'.format((i), prediction_length, valid_loss[i, 0],
                                                                               acc[i, 0]))
                ff.flush()

    seq_real_tp = seq_real_tp.cpu().numpy()
    seq_pred_tp = seq_pred_tp.cpu().numpy()
    valid_loss = valid_loss.cpu().numpy()
    acc = acc.cpu().numpy()
    acc_unweighted = acc_unweighted.cpu().numpy()
    tqe = tqe.cpu().numpy()
    return np.expand_dims(seq_real_tp, 0), np.expand_dims(seq_pred_tp, 0), np.expand_dims(valid_loss, 0), \
           np.expand_dims(acc, 0), np.expand_dims(acc_unweighted, 0), np.expand_dims(tqe, 0)


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
        h=h,
        w=w,
        p1=args.patch_size[0],
        p2=args.patch_size[1],
    )
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--run_num", default='00', type=str)
    # parser.add_argument("--yaml_config", default='./config/AFNO.yaml', type=str)
    # parser.add_argument("--config", default='full_field', type=str)
    parser.add_argument("--vis", action='store_true')
    # parser.add_argument("--override_dir", default=None, type=str,
    #                     help='Path to store inference outputs; must also set --weights arg')
    # parser.add_argument("--weights", default=None, type=str,
    #                     help='Path to model weights, for use with override_dir option')

    parser.add_argument("--run_num", default='p_qkv_16_16_precip', type=str)
    parser.add_argument("--yaml_config", default='./config/AFNO_16_16_precip.yaml', type=str)
    parser.add_argument("--config", default='precip', type=str)  # full_field
    parser.add_argument("--override_dir",
                        default='/home/lichangyu/data/codes/mae/mae-main/output_dir/p_qkv_16_16_precip/finetuning/ckpt/',
                        type=str,
                        help='Path to store inference outputs; must also set --weights arg')
    parser.add_argument("--weights", default='', type=str,
                        help='Path to model weights, for use with override_dir option')

    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params['world_size'] = 1
    params['global_batch_size'] = params.batch_size

    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    vis = args.vis

    # Set up directory
    for iii in range(400, 728):
        ff = open(
            r"/home/lichangyu/data/codes/mae/mae-main/output_dir/p_qkv_16_16_precip/finetuning/inference_for_valid.txt",
            "a", encoding="UTF-8")
        ff.write(str(iii) + '\n')
        flag_mx = 'true'
        # args.pretrained_ckpt_path = '/home/lichangyu/data/codes/mae/mae-main/output_dir/p_qkv_16_16_next_finetune_577/pretraining/ckpt/checkpoint-721.pth'
        args.pretrained_ckpt_path = '/home/lichangyu/data/codes/mae/mae-main/output_dir/p_qkv_16_16_next_finetune_twoStep/finetuning/ckpt/checkpoint-190.pth'
        args.weights = '/home/lichangyu/data/codes/mae/mae-main/output_dir/p_qkv_16_16_precip/finetuning/ckpt/checkpoint-' + str(
            iii) + '.pth'
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
        params['best_checkpoint_path_precip'] = args.pretrained_ckpt_path if args.override_dir is not None else os.path.join(expDir,
                                                                                                         'training_checkpoints/best_ckpt.tar')

        params['resuming'] = True
        params['local_rank'] = 0

        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'inference_out.log'))
        logging_utils.log_versions()
        params.log()

        n_ics = params['n_initial_conditions']
        ics = [1066, 1050, 1034]

        n_samples_per_year = 1460
        logging.info("SSSSSSSStep!!!!! Initial condition {} of {}".format(n_ics, n_samples_per_year))  # manxin

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

        autoregressive_inference_filetag += "_tp"
        # get data and models
        params["nettype"] = "mae_16_16"  # todo manxin note
        valid_data_full, valid_data_tp_full, model_wind, model = setup(params)

        # initialize lists for image sequences and RMSE/ACC
        valid_loss = np.zeros
        acc_unweighted = []
        acc = []
        tqe = []
        seq_pred = []
        seq_real = []

        # run autoregressive inference for multiple initial conditions
        for i, ic in enumerate(ics):
            t1 = time.time()
            logging.info("Initial condition {} of {}".format(i + 1, n_ics))
            ff.write("Initial condition {} of {} \n".format(i + 1, n_ics))
            if i == 0:
                sr, sp, vl, a, au, tq = autoregressive_inference(params, ic, valid_data_full, valid_data_tp_full,
                                                                 model_wind, model, ff)
            else:
                flag_mx = 'fasle'
                break

            if i == 0:
                seq_real = sr
                seq_pred = sp
                valid_loss = vl
                acc = a
                acc_unweighted = au
                tqe = tq
            else:
                #        seq_real = np.concatenate((seq_real, sr), 0)
                #        seq_pred = np.concatenate((seq_pred, sp), 0)
                valid_loss = np.concatenate((valid_loss, vl), 0)
                acc = np.concatenate((acc, a), 0)
                acc_unweighted = np.concatenate((acc_unweighted, au), 0)
                tqe = np.concatenate((tqe, tq), 0)
            t2 = time.time() - t1
            print("time for 1 autoreg inference = ", t2)

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
                f.create_dataset("acc_unweighted", data=acc_unweighted,
                                 shape=(n_ics, prediction_length, n_out_channels), dtype=np.float32)
            except:
                del f["acc_unweighted"]
                f.create_dataset("acc_unweighted", data=acc_unweighted,
                                 shape=(n_ics, prediction_length, n_out_channels), dtype=np.float32)
                f["acc_unweighted"][...] = acc_unweighted

            try:
                f.create_dataset("tqe", data=tqe, shape=(n_ics, prediction_length, n_out_channels), dtype=np.float32)
            except:
                del f["tqe"]
                f.create_dataset("tqe", data=tqe, shape=(n_ics, prediction_length, n_out_channels), dtype=np.float32)
                f["tqe"][...] = tqe

        f.close()
