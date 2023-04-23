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

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    optimizer.zero_grad()
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    sample_iter_step = 0
    # data_loader_bar = tqdm(data_loader)
    for data_iter_step, (samples) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        # samples = torch.load('/home/manxin/codes/fcn/pt_file/' + str(sample_iter_step) + '.pt')
        sample_iter_step += 1
        #manxin todo
        if epoch == 0 and data_iter_step ==1:
            inp = samples[0]
            for iii in range(20):
                try:
                    os.mkdir(args.save_dir + "/" + str(iii))
                except:
                    pass
                save_image(inp[0][iii], args.save_dir + "/" + str(iii) +"/ori.png")
        samples[0] = samples[0].to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            loss, pred, _ = model(samples[0], mask_ratio=args.mask_ratio)
            # print(pred.shape)
            # print(pred.shape)
            # if data_iter_step == 1 and (((epoch % 5) == 0) or (epoch == (args.epochs - 2))):
            if data_iter_step == 1:
                pred = rearrang_v1(args, pred)
                for jjj in range(20):
                    save_image(pred[0][jjj], args.save_dir + "/" + str(jjj) + "/" + str(epoch)+ ".png")
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
        loss_scaler(loss, optimizer, parameters=model.parameters(),
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

def rearrang_v1(args, x):
    # print("x.shape::::::::::::::::::::")
    # print(x.shape)
    # print(args.img_size)
    # print(args.input_size)
    # print("x.shape::::::::::::::::::::")
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