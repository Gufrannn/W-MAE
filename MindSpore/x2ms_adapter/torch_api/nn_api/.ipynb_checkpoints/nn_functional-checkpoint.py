#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
import math
import itertools
import numpy as np

import mindspore
import mindspore.nn
import mindspore.ops as ops
from .loss import CrossEntropyLoss, SmoothL1Loss, legacy_parameter, MarginRankingLoss
from ...utils.util_api import inplace_adaptor


def relu(input, inplace=False):
    relu_func = ops.ReLU()
    return inplace_adaptor(input, relu_func(input), inplace)


def relu_(input):
    return relu(input, inplace=True)


def softmax(input, dim=None, _stacklevel=3, dtype=None):
    if dim is None:
        dim = -1
    soft_max = ops.Softmax(axis=dim)
    if dtype:
        return soft_max(input).astype(dtype)
    return soft_max(input)


def avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True,
               divisor_override=None):
    """
    The input parameter 'ceil_mode', 'count_include_pad', 'divisor_override' are not supported.
    """
    ms_stride = stride
    if ms_stride is None:
        ms_stride = kernel_size
    pad_mode = 'valid'
    if padding > 0:
        pad_mode = 'same'

    avg_pool2d_func = ops.AvgPool(kernel_size=kernel_size, strides=ms_stride, pad_mode=pad_mode)

    return avg_pool2d_func(input)


def avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True,
               divisor_override=None):
    if divisor_override is None:
        divisor_override = 0
    if stride is None:
        stride = kernel_size
    avg_pool3d_func = ops.AvgPool3D(kernel_size=kernel_size, strides=stride, pad_mode='pad', pad=padding,
                                    ceil_mode=ceil_mode, count_include_pad=count_include_pad,
                                    divisor_override=divisor_override)
    return avg_pool3d_func(input)


def avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
    """
    The input parameter 'ceil_mode', 'count_include_pad' are not supported.
    """
    ms_stride = stride
    if ms_stride is None:
        ms_stride = kernel_size
    pad_mode = 'valid'
    if padding > 0:
        pad_mode = 'same'
    avg_pool1d_func = mindspore.nn.AvgPool1d(kernel_size=kernel_size, stride=ms_stride, pad_mode=pad_mode)
    return avg_pool1d_func(input)


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    if weight is None:
        weight = mindspore.ops.Ones()(tuple(normalized_shape), mindspore.float32)
    if bias is None:
        bias = mindspore.ops.Zeros()(tuple(normalized_shape), mindspore.float32)
    origin_dtype = input.dtype
    trans_flag = (origin_dtype != mindspore.float32)
    if trans_flag:
        input = input.astype(mindspore.float32)
    axis = input.ndim - len(normalized_shape)
    result = mindspore.ops.LayerNorm(axis, axis, epsilon=eps)(input, weight, bias)[0]
    if trans_flag:
        result = result.astype(origin_dtype)
    return result


def pixel_shuffle(input, upscale_factor):
    block_size = upscale_factor
    if block_size < 2:
        raise NotImplementedError(f"For 'DepthToSpace', the 'block_size' should be >= : 2")
    return mindspore.ops.DepthToSpace(block_size)(input)


def sigmoid(input):
    return ops.Sigmoid()(input)


def dropout(input, p=0.5, training=True, inplace=False):
    if not training:
        return input
    dropout_func = ops.Dropout(1 - p)
    output, _ = dropout_func(input)
    return output


def hardshrink(input, lambd=0.5):
    return mindspore.ops.HShrink(lambd)(input)


def dropout2d(input, p=0.5, training=True, inplace=False):
    if not training:
        return input
    dropout_obj = mindspore.ops.Dropout2D(1 - p)
    output, _ = dropout_obj(input)
    return inplace_adaptor(input, output, inplace)


def dropout3d(input, p=0.5, training=True, inplace=False):
    if not training:
        return input
    dropout_obj = mindspore.ops.Dropout3D(1 - p)
    output, _ = dropout_obj(input)
    return inplace_adaptor(input, output, inplace)


def adaptive_avg_pool2d(input, output_size):
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    if output_size == input.shape[-2:]:
        return ops.Identity()(input)

    if output_size == (1, 1):
        return ops.ReduceMean(keep_dims=True)(input, (-1, -2))

    return ops.AdaptiveAvgPool2D(output_size)(input)


def adaptive_avg_pool1d(input, output_size):
    if output_size == 1:
        return ops.ReduceMean(keep_dims=True)(input, -1)
    if input.ndim == 2:
        input = ops.ExpandDims()(input, 0)
        output = ops.AdaptiveAvgPool2D((None, output_size))(input)
        return output.squeeze(0)
    else:
        return ops.AdaptiveAvgPool2D((None, output_size))(input)


def gelu(input):
    return ops.GeLU()(input)


def x2ms_pad(input, pad, mode='constant', value=0):
    if not isinstance(pad, (list, tuple)) or len(pad) % 2 != 0 or len(pad) // 2 > input.ndim:
        raise ValueError(f'Invalid arg \'pad\' {pad}')
    input_dim = input.ndim
    new_pad = list((0, 0) for _ in range(input_dim))
    for i in range(len(pad) // 2):
        new_pad[-1 - i] = (pad[2 * i], pad[2 * i + 1])
    new_pad = tuple(new_pad)
    return ops.Pad(new_pad)(input)


def one_hot(tensor, num_classes=-1):
    if num_classes == -1:
        num_classes = int(tensor.asnumpy().max().item()) + 1
    return ops.OneHot()(tensor, num_classes,
                        mindspore.Tensor(1.0, mindspore.float32),
                        mindspore.Tensor(0.0, mindspore.float32)).astype(mindspore.int64)


def conv2d(data, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    out_channel = weight.shape[0]
    kernel_size = (weight.shape[2], weight.shape[3])
    if isinstance(padding, (list, tuple)) and len(padding) == 2:
        padding = (padding[0], padding[1]) * 2
    op_conv2d = ops.Conv2D(out_channel, kernel_size, mode=1, pad_mode="pad", pad=padding,
                           stride=stride, dilation=dilation, group=groups)
    return op_conv2d(data, weight)


def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    out_channel = weight.shape[0]
    kernel_size = (weight.shape[2], weight.shape[3], weight.shape[4])
    if isinstance(padding, (list, tuple)) and len(padding) == 3:
        padding = tuple(padding) * 2
    op_conv3d = ops.Conv3D(out_channel, kernel_size, mode=1, pad_mode="pad", pad=padding,
                           stride=stride, dilation=dilation, group=groups)
    return op_conv3d(input, weight)


def tensor_max_pool2d(tensor, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
    if stride is None:
        stride = kernel_size
    if kernel_size == 2 * padding + 1 or (ceil_mode and padding == 0):
        max_pool2d_func = mindspore.ops.MaxPool(kernel_size=kernel_size, strides=stride, pad_mode="same")
        return max_pool2d_func(tensor)
    elif padding == 0:
        max_pool2d_func = mindspore.ops.MaxPool(kernel_size=kernel_size, strides=stride, pad_mode="valid")
        return max_pool2d_func(tensor)
    else:
        raise NotImplementedError("Unsupported padding value")


def max_pool2d(obj, *args, **kwargs):
    if not isinstance(obj, mindspore.Tensor):
        raise TypeError('obj must be a MindSpore tensor.')
    return tensor_max_pool2d(obj, *args, **kwargs)


def max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    if stride is None:
        stride = 1
    if dilation != 1:
        raise ValueError('mindspore does not support dilation. Use default dilation=1')
    if return_indices is True:
        raise ValueError('mindspore does not support returning indices. Use default return_indices=False')
    if ceil_mode is True:
        raise ValueError('mindspore does not support ceiling mode. Use default ceil_mode=False')
    if padding == 0:
        max_pool1d_func = mindspore.nn.MaxPool1d(kernel_size=kernel_size, stride=stride, pad_mode='valid')
    else:
        if stride != 0:
            pt_sh = math.floor((input.shape[2] + 2 * padding - kernel_size) / stride + 1)
            ms_sh = math.ceil(input.shape[2] / stride)
            if pt_sh == ms_sh:
                max_pool1d_func = mindspore.nn.MaxPool1d(kernel_size=kernel_size, stride=stride, pad_mode='same')
            else:
                raise ValueError('mindspore.nn.MaxPool1d only support two padding modes: "valid" and "same"')
        else:
            raise ValueError('stride can not be zero.')
    return max_pool1d_func(input)


def max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    if stride is None:
        stride = 1
    if dilation != 1:
        raise ValueError('mindspore does not support dilation. Use default dilation=1')
    if return_indices is True:
        raise ValueError('mindspore does not support returning indices. Use default return_indices=False')
    max_pool3d_func = mindspore.ops.MaxPool3D(kernel_size=kernel_size, strides=stride, pad_mode="pad", pad_list=padding,
                                              ceil_mode=ceil_mode)
    if input.dim() == 4:
        input = ops.ExpandDims()(input, 0)
        return max_pool3d_func(input).squeeze(0)
    return max_pool3d_func(input)


def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
    def _check_size_scale_factor():
        if size is None and scale_factor is None:
            raise ValueError('either size or scale_factor should be defined')
        if size is not None and scale_factor is not None:
            raise ValueError('only one of size or scale_factor should be defined')

    _check_size_scale_factor()
    if size is not None:
        resize = ops.ResizeNearestNeighbor(size)
        return resize(input)
    dim = input.dim() - 2
    if isinstance(scale_factor, (tuple, list)):
        if len(scale_factor) != dim:
            raise ValueError('scale_factor shape must match input shape. '
                             'Input is {}D, scale_factor is {}'.format(dim, len(scale_factor)))
        scale_factors = scale_factor
    else:
        scale_factors = list(scale_factor for _ in range(dim))
    size = list(int(math.floor(input.shape[i + 2] * scale_factors[i])) for i in range(dim))
    resize = ops.ResizeNearestNeighbor(size)
    return resize(input)


def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100,
                  reduce=None, reduction='mean'):
    loss_func = CrossEntropyLoss(weight=weight, size_average=size_average, ignore_index=ignore_index,
                                 reduce=reduce, reduction=reduction)
    cross_entropy_loss = loss_func(input, target)
    return cross_entropy_loss


def binary_cross_entropy_with_logits(input, target, weight=None, size_average=None, reduce=None, reduction='mean',
                                     pos_weight=None):
    sigmoid_op = mindspore.ops.Sigmoid()
    bce_op = mindspore.ops.BinaryCrossEntropy(reduction=reduction)
    return bce_op(sigmoid_op(input), target, weight)


def cosine_embedding_loss(input1, input2, target, margin=0.0, size_average=None, reduce=None, reduction='mean'):
    reduction = legacy_parameter(size_average, reduce, reduction)
    margin = margin / 1.0
    cosine_embedding_loss_func = mindspore.nn.CosineEmbeddingLoss(margin=margin, reduction=reduction)
    return cosine_embedding_loss_func(input1, input2, target)


def nll_loss(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
    if len(input.shape) == 2:
        input = ops.ExpandDims()(input, 2)
        target = ops.ExpandDims()(target, 1)
    if weight is None:
        weight = mindspore.Tensor(np.ones(input.shape[1]), dtype=mindspore.dtype.float32)
    if ignore_index >= 0:
        weight[ignore_index] = 0
    if target.dtype not in (mindspore.int32,):
        target = target.astype(mindspore.int32)
    reduction = legacy_parameter(size_average, reduce, reduction)
    indices = [range(sh) for sh in input.shape[2:]]
    indices = itertools.product(*indices)
    nll_loss_func = mindspore.ops.NLLLoss(reduction=reduction)
    total_loss = mindspore.Tensor(0.0, mindspore.float32)
    total_weight = mindspore.Tensor(0.0, mindspore.float32)
    for index in indices:
        logits_slices = [slice(0, input.shape[0]), slice(0, input.shape[1])]
        labels_slices = [slice(0, input.shape[0])]
        for _index in index:
            logits_slices.append(slice(_index, _index + 1))
            labels_slices.append(slice(_index, _index + 1))
        logits = input[logits_slices].squeeze(list(range(2, len(input.shape))))
        labels = target[labels_slices].squeeze(list(range(1, len(target.shape))))
        loss, _total_weight = nll_loss_func(logits, labels, weight)
        if reduction == 'mean':
            total_loss += loss * _total_weight
        else:
            total_loss += loss
        total_weight += _total_weight
    if reduction == 'mean':
        if total_weight > 0:
            return total_loss / total_weight
        else:
            raise ZeroDivisionError('division by zero')
    else:
        return total_loss


def tanh(input):
    return ops.Tanh()(input)


def binary_cross_entropy(input, target, weight=None, size_average=None, reduce=None, reduction='mean'):
    return mindspore.ops.BinaryCrossEntropy(reduction=reduction)(input, target, weight)


def mse_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    mseloss = mindspore.nn.MSELoss(reduction=reduction)
    return mseloss(input, target)


def kl_div(input, target, size_average=None, reduce=None, reduction='mean', log_target=False):
    """
        The input parameter log_target is not implemented.
    """
    reduction = legacy_parameter(size_average, reduce, reduction)
    if reduction not in ['none', 'mean', 'sum']:
        raise NotImplementedError(f"unsupported {reduction} reduction mode")
    kl_div_loss = mindspore.ops.KLDivLoss(reduction=reduction)
    return kl_div_loss(input, target)


def l1_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    l1loss = mindspore.nn.L1Loss(reduction=reduction)
    return l1loss(input, target)


def silu(input, inplace=False):
    return input * mindspore.ops.Sigmoid()(input)


def mish(input, inplace=False):
    return mindspore.ops.Mish()(input)


def relu6(input, inplace=False):
    return mindspore.ops.ReLU6()(input)


def leaky_relu(input, negative_slope=0.01, inplace=False):
    return mindspore.nn.LeakyReLU(negative_slope)(input)


def elu(input, alpha=1.0, inplace=False):
    return mindspore.ops.Elu(alpha)(input)


def celu(input, alpha=1.0, inplace=False):
    return mindspore.nn.CELU(alpha)(input)


def selu(input, inplace=False):
    return mindspore.ops.SeLU()(input)


def hardswish(input, inplace=False):
    if mindspore.context.get_context('device_target') == 'Ascend':
        return input * mindspore.ops.ReLU6()(input + 3) / 6
    else:
        return mindspore.ops.HSwish()(input)


def hardsigmoid(input, inplace=False):
    return mindspore.ops.HSigmoid()(input)


def log_softmax(input, dim=None, _stacklevel=3, dtype=None):
    if dim is None:
        dim = 0 if input.dim() in (0, 1, 3) else 1
    return mindspore.ops.LogSoftmax(axis=dim)(input)


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if padding:
        conv1d_cell = mindspore.nn.Conv1d(input.shape[1], weight.shape[0], weight.shape[2], padding=padding,
                                          group=groups, pad_mode='pad', dilation=dilation, stride=stride)
    else:
        conv1d_cell = mindspore.nn.Conv1d(input.shape[1], weight.shape[0], weight.shape[2], group=groups,
                                          dilation=dilation, stride=stride)
    conv1d_cell.weight.assign_value(weight.reshape(*weight.shape[:2], 1, weight.shape[2]))
    if bias is not None:
        conv1d_cell.bias.assign_value(bias)
    return conv1d_cell(input)


def conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    if isinstance(stride, (tuple, list)) and len(stride) == 1:
        stride = stride[0]
    if isinstance(padding, (tuple, list)) and len(padding) == 1:
        padding = padding[0]
    if isinstance(output_padding, (tuple, list)) and len(output_padding) == 1:
        output_padding = output_padding[0]
    if isinstance(dilation, (tuple, list)) and len(dilation) == 1:
        dilation = dilation[0]
    if isinstance(padding, (tuple, list)) and len(padding) == 3:
        padding = (padding[0], padding[0], padding[1], padding[1], padding[2], padding[2])
    if bias is not None:
        raise NotImplementedError("The bias parameter can only be set to None in MindSpore 1.8.")
    kernel_size = weight.shape[-3:]
    in_channel = weight.shape[0]
    out_channel = weight.shape[1]
    conv3d_transpose = mindspore.ops.Conv3DTranspose(in_channel=in_channel, out_channel=out_channel,
                                                     kernel_size=kernel_size, mode=1, stride=stride, pad_mode='pad',
                                                     pad=padding, dilation=dilation, group=groups,
                                                     output_padding=output_padding, data_format='NCDHW')
    return conv3d_transpose(input, weight)


def tanhshrink(input):
    return mindspore.nn.Tanhshrink()(input)


def rrelu(input, lower=1 / 8, upper=1 / 3, inplace=False):
    rrelu_func = mindspore.nn.RReLU(lower, upper)
    return inplace_adaptor(input, rrelu_func(input), inplace)


def linear(input, weight, bias=None):
    if bias is not None:
        return mindspore.ops.matmul(input, weight.T) + bias
    else:
        return mindspore.ops.matmul(input, weight.T)


def softplus(input, beta=1, threshold=20):
    if beta != 1:
        raise NotImplementedError(f"MindSpore softplus does not support beta!=1. but got {beta}")
    return mindspore.ops.Softplus()(input)


def normalize(input, p=2, dim=1, eps=1e-12, out=None):
    if p != 2.0:
        raise NotImplementedError(f"MindSpore normalize does not support p!=2. but got {p}")
    return mindspore.ops.L2Normalize(axis=dim, epsilon=eps)(input)


def smooth_l1_loss(input, target, size_average=None, reduce=None, reduction='mean', beta=1.0):
    loss_func = SmoothL1Loss(size_average=size_average, reduce=reduce, reduction=reduction, beta=beta)
    return loss_func(input, target)


def soft_margin_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    reduction = legacy_parameter(size_average, reduce, reduction)
    soft_margin_loss_func = mindspore.nn.SoftMarginLoss(reduction=reduction)
    return soft_margin_loss_func(input, target)


def softsign(input):
    softsign_func = mindspore.ops.Softsign()
    return softsign_func(input)


def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05):
    batch_norm_func = ops.BatchNorm(is_training=training, epsilon=eps, momentum=momentum)
    if weight is None:
        weight = mindspore.ops.Ones()(input.shape[1], mindspore.float32)
    if bias is None:
        bias = mindspore.ops.Zeros()(input.shape[1], mindspore.float32)
    return batch_norm_func(input, weight, bias, running_mean, running_var)[0]


def margin_ranking_loss(input1, input2, target, margin=0, size_average=None, reduce=None, reduction='mean'):
    loss_func = MarginRankingLoss(margin, size_average, reduce, reduction)
    loss = loss_func(input1, input2, target)
    return loss


def local_response_norm(input, size, alpha=0.0001, beta=0.75, k=1.0):
    return mindspore.ops.LRN(depth_radius=int(size / 2), bias=k, alpha=alpha, beta=beta)(input)


def prelu(input, weight):
    return mindspore.ops.PReLU()(input, weight)


def fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
    if input.ndim == 2:
        if isinstance(kernel_size, int):
            input = input.reshape((1, -1, kernel_size ** 2, input.shape[1]))
        elif isinstance(kernel_size, tuple):
            input = input.reshape((1, -1, kernel_size[0] * kernel_size[1], input.shape[1]))
        return mindspore.ops.col2im(input, output_size, kernel_size, dilation, padding, stride).squeeze(axis=0)
    elif input.ndim == 3:
        if isinstance(kernel_size, int):
            input = input.reshape((input.shape[0], -1, kernel_size ** 2, input.shape[2]))
        elif isinstance(kernel_size, tuple):
            input = input.reshape((input.shape[0], -1, kernel_size[0] * kernel_size[1], input.shape[2]))
        return mindspore.ops.col2im(input, output_size, kernel_size, dilation, padding, stride)
    else:
        raise ValueError(f'Input must be a 2D or 3D tensor.')


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    return mindspore.ops.gumbel_softmax(logits, tau=tau, hard=hard, dim=dim)


def softshrink(input, lambd=0.5):
    return mindspore.ops.soft_shrink(input, lambd)


def hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False):
    hardtanh_func = mindspore.nn.Hardtanh(min_val=min_val, max_val=max_val)
    return inplace_adaptor(input, hardtanh_func(input), inplace)


def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None):
    if mode == 'bicubic':
        raise NotImplementedError("'bilinear' is not supported in mindspore yet")
    if align_corners is None:
        align_corners = False
    return mindspore.ops.grid_sample(input, grid, mode, padding_mode, align_corners)
