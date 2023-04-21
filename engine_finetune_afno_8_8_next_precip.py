import math
import os
import sys
from typing import Iterable

import torch
import util.misc_pre as misc
import util.lr_sched as lr_sched
from einops import rearrange, repeat
from torchvision.utils import save_image
from tqdm import tqdm
from afnonet_8 import PrecipNet
from utils.darcy_loss import LpLoss

def train_one_epoch(model: torch.nn.Module, tp_model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    # model.train(True)
    tp_model.train(True)

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=15, fmt='{value:.8f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    optimizer.zero_grad()
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    sample_iter_step = 0
    # precip_model = PrecipNet().to(device)
    # data_loader_bar = tqdm(data_loader)
    loss_obj = LpLoss()
    for data_iter_step, (samples) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        # samples = torch.load('/home/manxin/codes/fcn/pt_file/' + str(sample_iter_step) + '.pt')
        sample_iter_step += 1
        #manxin todo
        if epoch == 0 and data_iter_step ==50:
            inp = samples[1]
            for iii in range(1):
                try:
                    os.mkdir(args.save_dir + "/" + str(iii))
                except:
                    pass
                save_image(inp[0][iii], args.save_dir + "/" + str(iii) +"/tar_precip.png")
        samples[0] = samples[0].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                pred, _ = model(samples, mask_ratio=args.mask_ratio)
                pred = rearrang_v1(args, pred)
                if data_iter_step == 50:
                    save_image(pred[0][0],
                    args.save_dir + "/"  + "oneStepImgs" + "/"  + str(epoch) + "_" + str(data_iter_step) + "_onestep.png")

            samples[0] = pred.detach().to(device, non_blocking=True)
            # samples[0] = pred.to(device, non_blocking=True)

            samples[1] = samples[1].to(device, non_blocking=True)
            loss, precip_pred = tp_model(samples, mask_ratio=0., precip_flag=True)
            if data_iter_step % 200 == 0 or data_iter_step == 50:
                # pred = rearrang_v1(args, pred)
                for jjj in range(1):
                    save_image(precip_pred[0][jjj], args.save_dir + "/" + str(jjj) + "/" + str(epoch)+ "_" +str(data_iter_step)+ ".png")
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        # # manxin todo
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # manxin todo
        # torch.nan_to_num(loss)
        loss_scaler(loss.nan_to_num(), optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def patchify(args, imgs):
    """
    imgs: (N, 20, H, W)
    x: (N, L, patch_size**2 *20)
    """
    p = args.patch_size[0]
    # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = imgs.shape[2] // p
    w = imgs.shape[3] // p
    x = imgs.reshape(shape=(imgs.shape[0],imgs.shape[1], h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * imgs.shape[1]))
    return x

def forward_loss(args, imgs, pred):
    """
    imgs: [N, 3, H, W]
    pred: [N, L, p*p*3]
    mask: [N, L], 0 is keep, 1 is remove,
    """
    pred = patchify(args, pred)
    target = patchify(args, imgs)
    # if self.norm_pix_loss:
    #     mean = target.mean(dim=-1, keepdim=True)
    #     var = target.var(dim=-1, keepdim=True)
    #     target = (target - mean) / (var + 1.e-6) ** .5
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    mask = torch.ones_like(loss)
    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return loss

def rearrang_v1(args, x):
    B = x.shape[0]
    embed_dim = int(args.n_in_channels * (args.patch_size[0] * args.patch_size[1]))
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

def rearrang_v2(args, x):
    B = x.shape[0]
    embed_dim = int(args.n_in_channels * (args.patch_size[0] * args.patch_size[1]))
    h = int(args.img_size[0] // (args.patch_size[0]))
    w = int(args.img_size[1] // (args.patch_size[1]))
    x = x.reshape(B, h, w, embed_dim)
    return x

def rearrang_v0(args, x):
    # print("x.shape::::::::::::::::::::")
    # print(x.shape)
    # print(args.img_size)
    # print(args.input_size)
    # print("x.shape::::::::::::::::::::")
    B = x.shape[0]
    # embed_dim = int(args.n_out_channels * (args.patch_size[0] * args.patch_size[1]))
    # embed_dim = int((args.patch_size[0] * args.patch_size[1]))
    h = int(args.img_size[0] // (args.patch_size[0]))
    w = int(args.img_size[1] // (args.patch_size[1]))
    x = x.reshape(B, args.n_in_channels, args.img_size[0], args.img_size[1])
    # x = rearrange(
    #     x,
    #     "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
    #     h = h,
    #     w = w,
    #     p1=args.patch_size[0],
    #     p2=args.patch_size[1],
    # )
    return x