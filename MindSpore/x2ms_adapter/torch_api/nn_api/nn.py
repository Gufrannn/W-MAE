#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
import collections
import copy
import warnings
from collections import OrderedDict
from distutils.version import LooseVersion
from typing import Any, Iterable, Iterator, Optional
import math

import mindspore.nn
import mindspore.ops as ops
from mindspore.nn import Tanh, Sigmoid  # noqa
from mindspore.common.initializer import initializer

from . import nn_functional
from .nn_functional import adaptive_avg_pool2d, adaptive_avg_pool1d, layer_norm, hardswish
from .nn_init import uniform_, normal_
from ...utils.util_api import logger


class AdaptiveAvgPool2d(mindspore.nn.Cell):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def construct(self, input):
        return adaptive_avg_pool2d(input, self.output_size)


class BatchNorm3d(mindspore.nn.BatchNorm3d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,
                 device=None, dtype=None):
        # momentum in MindSpore should be (1-momentum) in PyTorch, refer to official api mapping for more details
        if track_running_stats:
            use_batch_statistics = None
        else:
            raise NotImplementedError('"track_running_stats=False" is not supported in MindSpore.')
        if num_features == int(num_features):
            num_features = int(num_features)
        super().__init__(num_features, eps=eps, momentum=1 - momentum, affine=affine,
                         use_batch_statistics=use_batch_statistics)

    @property
    def running_mean(self):
        return self.moving_mean

    @property
    def running_var(self):
        return self.moving_variance

    @running_mean.setter
    def running_mean(self, mean):
        self.moving_mean.set_data(mean)

    @running_var.setter
    def running_var(self, var):
        self.moving_variance.set_data(var)

    @property
    def bias(self):
        return self.beta

    @property
    def weight(self):
        return self.gamma

    def _set_mixed_precision_type_recursive(self, mixed_type):
        pass


class BatchNorm2d(mindspore.nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,
                 device=None, dtype=None):
        # momentum in MindSpore should be (1-momentum) in PyTorch, refer to official api mapping for more details
        if track_running_stats:
            use_batch_statistics = None
        else:
            raise NotImplementedError('"track_running_stats=False" is not supported in MindSpore.')
        if num_features == int(num_features):
            num_features = int(num_features)
        super().__init__(num_features, eps=eps, momentum=1 - momentum, affine=affine,
                         use_batch_statistics=use_batch_statistics)

    @property
    def bias(self):
        return self.beta

    @property
    def weight(self):
        return self.gamma

    @property
    def running_mean(self):
        return self.moving_mean

    @property
    def running_var(self):
        return self.moving_variance

    @running_mean.setter
    def running_mean(self, mean):
        self.moving_mean.set_data(mean)

    @running_var.setter
    def running_var(self, var):
        self.moving_variance.set_data(var)

    def _set_mixed_precision_type_recursive(self, mixed_type):
        pass


class BatchNorm1d(mindspore.nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,
                 device=None, dtype=None):
        if track_running_stats:
            use_batch_statistics = None
        else:
            raise NotImplementedError('"track_running_stats=False" is not supported in MindSpore.')
        super().__init__(num_features, eps=eps, momentum=1 - momentum, affine=affine,
                         use_batch_statistics=use_batch_statistics)

    @property
    def bias(self):
        return self.beta

    @property
    def weight(self):
        return self.gamma

    @property
    def running_mean(self):
        return self.moving_mean

    @property
    def running_var(self):
        return self.moving_variance

    def construct(self, input):
        if input.dim() == 2:
            return super().construct(input)
        elif input.dim() == 3:
            input = mindspore.ops.Transpose()(input, (2, 0, 1))
            result = []
            for data in input:
                result.append(super().construct(data))
            return mindspore.numpy.stack(result, axis=2)
        else:
            raise NotImplementedError("BatchNorm1d does not support input dim not in [2, 3].")


class SyncBatchNorm(mindspore.nn.SyncBatchNorm):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, process_group=None,
                 device=None, dtype=None):
        if track_running_stats:
            use_batch_statistics = None
        else:
            raise NotImplementedError('"track_running_stats=False" is not supported in MindSpore.')
        super().__init__(num_features, eps=eps, momentum=1 - momentum, affine=affine,
                         use_batch_statistics=use_batch_statistics, process_groups=process_group)

    @property
    def bias(self):
        return self.beta

    @property
    def weight(self):
        return self.gamma

    @property
    def running_mean(self):
        return self.moving_mean

    @property
    def running_var(self):
        return self.moving_variance


class _BatchNorm(mindspore.nn.layer.normalization._BatchNorm):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True,
                 track_running_stats: bool = True, device=None, dtype=None):
        if track_running_stats:
            use_batch_statistics = None
        else:
            raise NotImplementedError('"track_running_stats=False" is not supported in MindSpore.')
        super().__init__(num_features, eps=eps, momentum=1 - momentum, affine=affine,
                         use_batch_statistics=use_batch_statistics)

    @property
    def bias(self):
        return self.beta

    @property
    def weight(self):
        return self.gamma

    @property
    def running_mean(self):
        return self.moving_mean

    @property
    def running_var(self):
        return self.moving_variance


class Conv1d(mindspore.nn.Conv1d):
    '''
    1.The 'kernel_size' parameter in MindSpore supports only the Int type. Therefore, when 'kernel_size' is of the
    Tuple type, the position of kernel_size[0] is automatically captured.
    2.The 'device' and 'dtype' parameters in the torch.nn.Conv1d are invalid in Mindspore.nn.Conv1d.
    3.The value and policy of 'padding_mode' in MindSpore are different from those in PyTorch. Therefore, padding_mode
    is automatically set to pad.
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        if isinstance(kernel_size, tuple):
            if kernel_size:
                kernel_size = int(kernel_size[0])
            else:
                raise IndexError("The length of param 'kernel_size' must be greater than 0.")

        padding_mode = 'pad'
        super().__init__(in_channels, out_channels, kernel_size, stride, padding_mode, padding, dilation, groups, bias)

        if dtype:
            raise ValueError("The 'dtype' parameters in the torch.nn.Conv1d are invalid in mindspore.nn.Conv1d.")

    def construct(self, input):
        if input.dtype == mindspore.float64:
            input = ops.Cast()(input, mindspore.float32)
        return super().construct(input)


class Conv2d(mindspore.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        if isinstance(stride, (tuple, list)) and len(stride) == 1:
            stride = stride[0]
        if isinstance(padding, (tuple, list)) and len(padding) == 2:
            padding = (padding[0], padding[0], padding[1], padding[1])
        if in_channels == int(in_channels):
            in_channels = int(in_channels)
        if out_channels == int(out_channels):
            out_channels = int(out_channels)
        if groups == int(groups):
            groups = int(groups)
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, pad_mode='pad',
                         dilation=dilation, group=groups, has_bias=bias)
        if isinstance(kernel_size, int):
            mul_kernel_size = kernel_size * kernel_size
        else:
            mul_kernel_size = 1
            for size in kernel_size:
                mul_kernel_size *= size
        if mul_kernel_size == 0 or in_channels == 0:
            raise ValueError('Conv2d does not support in_channels == 0 or 0 in kernel_size')
        uniform_limit = math.sqrt(groups * 1.0 / (in_channels * mul_kernel_size))
        uniform_(self.weight, -uniform_limit, uniform_limit)
        if bias:
            uniform_(self.bias, -uniform_limit, uniform_limit)

    def __setattr__(self, key, value):
        if hasattr(self, key) and _check_is_new_parameter_set_to_parameter(getattr(self, key), value):
            mindspore.ops.Assign()(getattr(self, key), value)
        else:
            super().__setattr__(key, value)

    @property
    def groups(self):
        return self.group

    def construct(self, input):
        if input.dtype == mindspore.float64:
            input = ops.Cast()(input, mindspore.float32)
        return super().construct(input)


class Conv3d(mindspore.nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        if isinstance(stride, (tuple, list)) and len(stride) == 1:
            stride = stride[0]
        if isinstance(padding, (tuple, list)) and len(padding) == 3:
            padding = (padding[0], padding[0], padding[1], padding[1], padding[2], padding[2])
        if padding_mode not in ('zeros',):
            raise ValueError(f'Conv3d does not support padding_mode == "{padding_mode}". Try to use "zeros"')
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, pad_mode='pad',
                         dilation=dilation, group=groups, has_bias=bias, data_format='NCDHW')
        if isinstance(kernel_size, int):
            mul_kernel_size = kernel_size * kernel_size * kernel_size
        else:
            mul_kernel_size = 1
            for size in kernel_size:
                mul_kernel_size *= size
        if mul_kernel_size == 0 or in_channels == 0:
            raise ValueError('Conv3d does not support in_channels == 0 or 0 in kernel_size')
        uniform_limit = math.sqrt(groups * 1.0 / (in_channels * mul_kernel_size))
        uniform_(self.weight, -uniform_limit, uniform_limit)
        if bias:
            uniform_(self.bias, -uniform_limit, uniform_limit)

    def __setattr__(self, key, value):
        if hasattr(self, key) and _check_is_new_parameter_set_to_parameter(getattr(self, key), value):
            mindspore.ops.Assign()(getattr(self, key), value)
        else:
            super().__setattr__(key, value)

    @property
    def groups(self):
        return self.group

    @property
    def padding_mode(self):
        return self.pad_mode


class GroupNorm(mindspore.nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps=1e-05, affine=True, device=None, dtype=None):
        super().__init__(num_groups, num_channels, eps=eps, affine=affine)

    @property
    def bias(self):
        return self.beta

    @bias.setter
    def bias(self, bias):
        self.beta = bias

    @property
    def weight(self):
        return self.gamma

    @weight.setter
    def weight(self, weight):
        self.gamma = weight


class Linear(mindspore.nn.Dense):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, has_bias=bias)
        self.in_features = in_features
        self.out_features = out_features


class MaxPool2d(mindspore.nn.MaxPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        if dilation != 1 or return_indices:
            raise NotImplementedError("Unsupported init parameter in MaxPool2d")
        if stride is None:
            stride = kernel_size
        if kernel_size == 2 * padding + 1 or (ceil_mode and padding == 0):
            super().__init__(kernel_size=kernel_size, stride=stride, pad_mode="same")
        elif padding == 0:
            super().__init__(kernel_size=kernel_size, stride=stride, pad_mode="valid")
        else:
            raise NotImplementedError("Unsupported padding value")


class ReLU(mindspore.nn.ReLU):
    def __init__(self, inplace=False):
        super().__init__()


class GELU(mindspore.nn.GELU):
    def __init__(self):
        super().__init__(approximate=False)


class Sequential(mindspore.nn.SequentialCell):
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            super().__init__(args[0])
        else:
            super().__init__(list(args))

    def add_module(self, name, module):
        module.update_parameters_name(f'{name}.')
        self._is_dynamic_name.append(True)
        self._cells[name] = module
        self.cell_list = list(self._cells.values())


class ModuleList(mindspore.nn.CellList):
    def __init__(self, modules=None):
        if not modules:
            super().__init__([])
        else:
            if not isinstance(modules, list):
                modules = list(modules)
            super().__init__(modules)

    def extend(self, modules):
        if not isinstance(modules, list):
            modules = list(modules)
        super().extend(modules)


class LogSoftmax(mindspore.nn.Cell):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def construct(self, input):
        if self.dim is None:
            dim = 0 if input.ndim in (0, 1, 3) else 1
        else:
            dim = self.dim
        if input.ndim <= 2:
            return mindspore.ops.LogSoftmax(dim)(input)
        input = mindspore.ops.Softmax(dim)(input)
        input = mindspore.ops.functional.log(input)
        return input


class AvgPool2d(mindspore.nn.AvgPool2d):
    def __init__(self, kernel_size, stride=None, padding=0,
                 ceil_mode=False, count_include_pad=True, divisor_override=None):
        ms_stride = stride
        if ms_stride is None:
            ms_stride = kernel_size
        pad_mode = 'valid'
        if padding > 0:
            pad_mode = 'same'
        super().__init__(kernel_size=kernel_size, stride=ms_stride, pad_mode=pad_mode)


class AvgPool1d(mindspore.nn.AvgPool1d):
    """
        The input parameter ceil_mode=False, count_include_pad=True is not implemented.
        ceil_mode: When true, the output shape is calculated using ceil instead of floor
        count_include_pad: When True, zero padding is included in the average calculation
    """

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        ms_stride = stride
        if stride is None:
            ms_stride = kernel_size
        pad_mode = 'valid'
        if padding > 0:
            pad_mode = 'same'
        super().__init__(kernel_size=kernel_size, stride=ms_stride, pad_mode=pad_mode)


class Dropout(mindspore.nn.Dropout):
    def __init__(self, p=0.5, inplace=False):
        super().__init__(keep_prob=float(1 - p))


class GRU(mindspore.nn.GRU):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.,
                 bidirectional=False):
        super().__init__(input_size, hidden_size, num_layers=num_layers, has_bias=bias, batch_first=batch_first,
                         dropout=dropout, bidirectional=bidirectional)


class RNN(mindspore.nn.RNN):
    def __init__(self, input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True, batch_first=False,
                 dropout=0., bidirectional=False):
        super().__init__(input_size, hidden_size, num_layers=num_layers, nonlinearity=nonlinearity, has_bias=bias,
                         batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)


class LSTM(mindspore.nn.LSTM):
    """
    The parameter proj_size is not implemented.
    """

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.,
                 bidirectional=False, proj_size=0):
        super().__init__(input_size, hidden_size, num_layers=num_layers, has_bias=bias, batch_first=batch_first,
                         dropout=float(dropout), bidirectional=bidirectional)

    def flatten_parameters(self):
        pass

    def construct(self, x, hx=None, seq_length=None):
        if isinstance(hx, list):
            hx = tuple(hx)
        return super().construct(x, hx, seq_length)


class LSTMCell(mindspore.nn.LSTMCell):
    """
    The parameter device and dtype are not implemented.
    """

    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype=None):
        super().__init__(input_size, hidden_size, has_bias=bias)


class MaxPool1d(mindspore.nn.MaxPool1d):
    """
      The input parameter  proj_size=0，device=None, dtype=None is not implemented.
      dilation: Parameters that control the step size of an element in a window
      return_indices: If true, the maximum index and output are returned.
      ceil_mode:  When true, the output shape is calculated using ceil instead of floor
    """

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        if stride is None:
            stride = kernel_size
        if kernel_size == 2 * padding + 1:
            super().__init__(kernel_size=kernel_size, stride=stride, pad_mode="same")
        elif padding == 0:
            super().__init__(kernel_size=kernel_size, stride=stride, pad_mode="valid")
        else:
            raise NotImplementedError("Unsupported padding value")


class PixelShuffle(mindspore.nn.Cell):
    def __init__(self, upscale_factor):
        self.block_size = upscale_factor
        if self.block_size < 2:
            raise NotImplementedError(f"For 'DepthToSpace', the 'block_size' should be >= : 2")
        super().__init__()

    def construct(self, input):
        return mindspore.ops.DepthToSpace(self.block_size)(input)


class Embedding(mindspore.nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0,
                 scale_grad_by_freq=False, sparse=False, _weight=None, device=None, dtype=None):
        """
        Parameter 'padding_idx', 'max_norm', 'norm_type', 'scale_grad_by_freq', 'sparse', 'device' and 'dtype'
        is not supported.
        """
        embedding_table = 'normal' if _weight is None else _weight
        super().__init__(vocab_size=num_embeddings, embedding_size=embedding_dim, embedding_table=embedding_table)
        if _weight is None:
            normal_(self.weight, 0.0, 1.0)
        if padding_idx is not None:
            self.weight[padding_idx] = 0

    @property
    def num_embeddings(self):
        return self.vocab_size

    @property
    def embedding_dim(self):
        return self.embedding_size

    @property
    def weight(self):
        return self.embedding_table

    @classmethod
    def from_pretrained(cls, embeddings, freeze=True, padding_idx=None, max_norm=None, norm_type=2.0,
                        scale_grad_by_freq=False, sparse=False):
        """
        Parameter 'freeze', 'padding_idx', 'max_norm', 'norm_type', 'scale_grad_by_freq', 'sparse' is not supported.
        """
        num_embeddings = embeddings.shape[0]
        embedding_dim = embeddings.shape[1]
        return cls(num_embeddings, embedding_dim, _weight=embeddings)


class Flatten(mindspore.nn.Flatten):
    def __init__(self, start_dim=1, end_dim=-1):
        self.start_dim = start_dim
        self.end_dim = end_dim
        if start_dim != 1 or (end_dim != -1):
            raise ValueError('In MindSpore, only the 0th dimension will be reserved and the rest will be flattened.'
                             'Do not specify the start_dim and end_dim')
        super(Flatten, self).__init__()


class Hardshrink(mindspore.nn.HShrink):
    def __init__(self, lambd=0.5):
        super(Hardshrink, self).__init__(lambd=lambd)


class Softshrink(mindspore.nn.SoftShrink):
    def __init__(self, lambd=0.5):
        super(Softshrink, self).__init__(lambd=lambd)


class Upsample(mindspore.nn.Cell):
    """
    Only supported mode 'nearest' or 'bilinear'
    """

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super().__init__()
        self.size = size
        if isinstance(scale_factor, float):
            logger.warning(f"Upsample in MindSpore doesn't support scale_factor to be float value '{scale_factor}', "
                           f"', will converted to int value '{int(scale_factor)}'.")
            scale_factor = int(scale_factor)
        self.scale_factor = scale_factor
        self.align_corners = align_corners
        self.mode = mode
        if self.align_corners is None:
            self.align_corners = False

    def construct(self, input):
        size = input.shape
        if self.scale_factor is not None:
            size = list(sub_size * self.scale_factor for sub_size in size[len(size) - 2:])
        else:
            size = self.size
        if self.mode == 'nearest':
            upsample_ops = mindspore.ops.ResizeNearestNeighbor(size, align_corners=self.align_corners)
        else:
            upsample_ops = mindspore.ops.ResizeBilinear(size, align_corners=self.align_corners)
        return upsample_ops(input)


class Identity(mindspore.nn.Cell):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def construct(self, input):
        return ops.Identity()(input)


class Pad(mindspore.nn.Cell):
    def __init__(self, padding):
        if isinstance(padding, int):
            padding = tuple(padding for _ in range(4))
        elif isinstance(padding, (list, tuple)) and len(padding) == 4:
            padding = tuple(padding)
        else:
            raise ValueError(f'Invalid arg \'padding\': {padding}')
        self.padding = padding
        super().__init__()

    def construct(self, input):
        padding = [[0, 0] for _ in range(input.ndim)]
        padding[-1] = self.padding[0:2]
        padding[-2] = self.padding[2:4]
        padding = tuple(tuple(elem) for elem in padding)
        return ops.Pad(padding)(input)


class ZeroPad2d(Pad):
    def __init__(self, padding):
        super().__init__(padding)


class LambdaCell(mindspore.nn.Cell):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def construct(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class LayerNorm(mindspore.nn.Cell):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.epsilon = eps
        self.weight = mindspore.Parameter(initializer(
            'ones', normalized_shape), name="weight")
        self.bias = mindspore.Parameter(initializer(
            'zeros', normalized_shape), name="bias")

    def __setattr__(self, key, value):
        if hasattr(self, key) and _check_is_new_parameter_set_to_parameter(getattr(self, key), value):
            mindspore.ops.Assign()(getattr(self, key), value)
        else:
            super().__setattr__(key, value)

    def construct(self, input_x):
        return layer_norm(input_x, self.normalized_shape, self.weight, self.bias, self.epsilon)

    def _set_mixed_precision_type_recursive(self, mixed_type):
        pass


class Softmax(mindspore.nn.Softmax):
    def __init__(self, dim=None):
        if dim is None:
            dim = -1
        super().__init__(axis=dim)


class ConstantPad2d(Pad):
    def __init__(self, padding, value):
        if value != 0:
            raise NotImplementedError("value must be 0")
        self.value = value
        super().__init__(padding)


class ConvTranspose1d(mindspore.nn.Conv1dTranspose):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1,
                 bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None):
        if isinstance(stride, (tuple, list)) and len(stride) == 1:
            stride = stride[0]
        if isinstance(padding, (tuple, list)) and len(padding) == 1:
            padding = padding[0]
        if isinstance(output_padding, (tuple, list)) and len(output_padding) == 1:
            output_padding = output_padding[0]
        if isinstance(dilation, (tuple, list)) and len(dilation) == 1:
            dilation = dilation[0]
        if isinstance(kernel_size, (tuple, list)) and len(kernel_size) == 1:
            kernel_size = kernel_size[0]
        if output_padding != 0:
            warnings.warn('ConvTranspose1d does not support output_padding. `output_padding` is set it to 0')
        if padding_mode not in ('zeros'):
            raise ValueError(f'ConvTranspose1d does not support padding_mode == "{padding_mode}". Try to use "zeros"')
        super().__init__(in_channels, out_channels, kernel_size, stride, pad_mode='pad', padding=padding,
                         dilation=dilation, group=groups, has_bias=bias)
        mul_kernel_size = kernel_size
        if mul_kernel_size == 0 or in_channels == 0:
            raise ValueError('ConvTranspose1d does not support in_channels == 0 or 0 in kernel_size')

    def __setattr__(self, key, value):
        if hasattr(self, key) and _check_is_new_parameter_set_to_parameter(getattr(self, key), value):
            mindspore.ops.Assign()(getattr(self, key), value)
        else:
            super().__setattr__(key, value)

    @property
    def groups(self):
        return self.group

    @property
    def padding_mode(self):
        return self.pad_mode


class ConvTranspose2d(mindspore.nn.Conv2dTranspose):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1,
                 bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None):
        if isinstance(padding, (tuple, list)) and len(padding) == 2:
            padding = (padding[0], padding[0], padding[1], padding[1])
        super().__init__(in_channels, out_channels, kernel_size, stride, pad_mode='pad', padding=padding,
                         dilation=dilation, group=groups, has_bias=bias)


class ConvTranspose3d(mindspore.nn.Conv3dTranspose):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1,
                 bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None):
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
        if output_padding != 0:
            warnings.warn('ConvTranspose3d does not support output_padding. `output_padding` is set it to 0')
        if padding_mode not in ('zeros'):
            raise ValueError(f'ConvTranspose3d does not support padding_mode == "{padding_mode}". Try to use "zeros"')
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, pad_mode='pad',
                         dilation=dilation, group=groups, has_bias=bias, data_format='NCDHW')
        if isinstance(kernel_size, int):
            mul_kernel_size = kernel_size * kernel_size * kernel_size
        else:
            mul_kernel_size = 1
            for size in kernel_size:
                mul_kernel_size *= size
        if mul_kernel_size == 0 or in_channels == 0:
            raise ValueError('ConvTranspose3d does not support in_channels == 0 or 0 in kernel_size')

    def __setattr__(self, key, value):
        if hasattr(self, key) and _check_is_new_parameter_set_to_parameter(getattr(self, key), value):
            mindspore.ops.Assign()(getattr(self, key), value)
        else:
            super().__setattr__(key, value)

    @property
    def groups(self):
        return self.group

    @property
    def padding_mode(self):
        return self.pad_mode


class LeakyReLU(mindspore.nn.LeakyReLU):
    def __init__(self, negative_slope=0.01, inplace=False):
        super().__init__(alpha=negative_slope)


class ParameterList(mindspore.nn.Cell):
    def __init__(self, parameters: Optional[Iterable['Parameter']] = None) -> None:
        super(ParameterList, self).__init__()
        if parameters is not None:
            self += parameters

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(list(self._params.values())[idx])
        else:
            idx = self._get_abs_string_index(idx)
            return self._params[idx]

    def __setitem__(self, idx: int, param: 'Parameter') -> None:
        idx = self._get_abs_string_index(idx)
        return self.insert_param_to_cell(idx, param)

    def __setattr__(self, key: Any, value: Any) -> None:
        if not isinstance(value, mindspore.Parameter):
            warnings.warn("Setting attributes on ParameterList is not supported.")
        super(ParameterList, self).__setattr__(key, value)

    def __len__(self) -> int:
        return len(self._params)

    def __iter__(self) -> Iterator['Parameter']:
        return iter(self._params.values())

    def __iadd__(self, parameters: Iterable['Parameter']):
        return self.extend(parameters)

    def __dir__(self):
        keys = super(ParameterList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def __call__(self, input):
        raise RuntimeError('ParameterList should not be called.')

    def append(self, parameter: 'Parameter'):
        self.insert_param_to_cell(str(len(self)), parameter)
        return self

    def extend(self, parameters: Iterable['Parameter']):
        if not isinstance(parameters, collections.abc.Iterable):
            raise TypeError("ParameterList.extend should be called with an "
                            "iterable, but got " + type(parameters).__name__)
        offset = len(self)
        for i, param in enumerate(parameters):
            self.insert_param_to_cell(str(offset + i), param)
        return self

    def _get_abs_string_index(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)


class ModuleDict(mindspore.nn.Cell):
    def __init__(self, *args, **kwargs):
        auto_prefix = kwargs.get("auto_prefix") if "auto_prefix" in kwargs.keys() else True
        mindspore.nn.Cell.__init__(self, auto_prefix)
        if len(args) == 1:
            self.update(args[0])

    def __getitem__(self, key: str):
        return self._cells[key]

    def __setitem__(self, key: str, module) -> None:
        self._cells[key] = module

    def __delitem__(self, key: str) -> None:
        del self._cells[key]

    def __len__(self) -> int:
        return len(self._cells)

    def __iter__(self) -> Iterator[str]:
        return iter(self._cells)

    def __contains__(self, key: str) -> bool:
        return key in self._cells

    def clear(self) -> None:
        self._cells.clear()

    def pop(self, key: str):
        value = self[key]
        del self[key]
        return value

    def keys(self) -> Iterable[str]:
        return self._cells.keys()

    def items(self):
        return self._cells.items()

    def values(self):
        return list(self._cells.values())

    def update(self, modules) -> None:
        if not isinstance(modules, collections.abc.Iterable):
            raise TypeError("ModuleDict.update should be called with an "
                            "iterable of key/value pairs, but got " +
                            type(modules).__name__)

        if isinstance(modules, (OrderedDict, ModuleDict, collections.abc.Mapping)):
            for key, module in modules.items():
                self[key] = module
        else:
            for index, module in enumerate(modules):
                if not isinstance(module, collections.abc.Iterable):
                    raise TypeError("ModuleDict update sequence element "
                                    "#" + str(index) + " should be Iterable; is" +
                                    type(module).__name__)
                if not len(module) == 2:
                    raise ValueError("ModuleDict update sequence element "
                                     "#" + str(index) + " has length " + str(len(module)) +
                                     "; 2 is required")
                self[module[0]] = module[1]

    def construct(self, *inputs):
        raise NotImplementedError


class DistributedDataParallel(mindspore.nn.Cell):
    def __init__(self, module, device_ids=None, output_device=None, dim=0, broadcast_buffers=True,
                 process_group=None, bucket_cap_mb=25, find_unused_parameters=False, check_reduction=False):
        super().__init__()
        self.module = module

    def construct(self, *inputs, **kwargs):
        return self.module.construct(*inputs, **kwargs)


class DataParallel(mindspore.nn.Cell):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super().__init__()
        self.module = module

    def construct(self, *inputs, **kwargs):
        return self.module.construct(*inputs, **kwargs)


class SiLU(mindspore.nn.Cell):
    def __init__(self, inplace=False):
        super().__init__()
        self.sigmoid = mindspore.ops.Sigmoid()

    def construct(self, input):
        return input * self.sigmoid(input)


class ReLU6(mindspore.nn.ReLU6):
    def __init__(self, inplace=False):
        super().__init__()


class ELU(mindspore.nn.ELU):
    def __init__(self, alpha=1.0, inplace=False):
        super().__init__(alpha)


class CELU(mindspore.nn.CELU):
    def __init__(self, alpha=1.0, inplace=False):
        super().__init__(alpha)


class SELU(mindspore.nn.Cell):
    def __init__(self, inplace=False):
        super().__init__()
        self.selu = mindspore.ops.SeLU()

    def construct(self, input):
        return self.selu(input)


class Hardswish(mindspore.nn.Cell):
    def __init__(self, inplace=False):
        super().__init__()

    def construct(self, x):
        return hardswish(x)


class Hardsigmoid(mindspore.nn.HSigmoid):
    def __init__(self, inplace=False):
        super().__init__()


class PReLU(mindspore.nn.PReLU):
    def __init__(self, num_parameters=1, init=0.25, device=None, dtype=None):
        super().__init__(channel=num_parameters, w=init)
        self.dtype = dtype

    def construct(self, x):
        output = super().construct(x)
        if self.dtype:
            return output.astype(self.dtype)
        return output


class UpsamplingBilinear2d(mindspore.nn.Cell):
    def __init__(self, size=None, scale_factor=None):
        super().__init__()
        if size is not None and scale_factor is not None:
            raise ValueError(
                "size and scale_factor cannot be defined at the same time.")
        elif size is not None:
            if isinstance(size, int):
                self.size = (size, size)
            else:
                self.size = size
        elif scale_factor is not None:
            if isinstance(scale_factor, tuple):
                self.scale_factor = scale_factor
            else:
                self.scale_factor = (scale_factor, scale_factor)
        else:
            raise ValueError("size or scale_factor should be defined.")

    def construct(self, input):
        if hasattr(self, "size"):
            new_shape = self.size
        else:
            new_shape = [input.shape[2] * self.scale_factor[0],
                         input.shape[3] * self.scale_factor[1]]
        upsample_ops = mindspore.ops.ResizeBilinear(new_shape,
                                                    align_corners=True)
        return upsample_ops(input)


class MultiheadAttention(mindspore.nn.transformer.MultiHeadAttention):
    """
    Please input mindspore.nn.transformer.MultiHeadAttention args:
    batch_size, src_seq_length, tgt_seq_length.
        batch_size(int): The batch size of the input tensor.
        src_seq_length(int): The sequence length of the query vector.
        tgt_seq_length(int): The sequence length of the key and value vector.

    Examples:
        batch_size = 32
        src_seq_length = 10
        tgt_seq_length = 20

    Parameter 'add_bias_kv', 'add_zero_attn', 'kdims', 'vdims'
    is not supported.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True,
                 add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        batch_size = None
        if batch_size is None or not isinstance(batch_size, int):
            raise ValueError(
                "Please modify the value of batch_size in the previous line."
                " batch_size must be an integer, and it is the batch size of the input tensor.")

        src_seq_length = None
        if src_seq_length is None or not isinstance(src_seq_length, int):
            raise ValueError(
                "Please modify the value of src_seq_length in the previous line. "
                "src_seq_length must be an integer, and it is the sequence length of the query vector.")

        tgt_seq_length = None
        if tgt_seq_length is None or not isinstance(tgt_seq_length, int):
            raise ValueError(
                "Please modify the value of tgt_seq_length in the previous line. "
                "tgt_seq_length must be an integer, and it is the sequence length of the key and value vector.")

        super().__init__(batch_size=batch_size, src_seq_length=src_seq_length,
                         tgt_seq_length=tgt_seq_length, hidden_size=embed_dim,
                         num_heads=num_heads, compute_dtype=mindspore.float16)
        self.attention_mask = mindspore.ops.Ones()((batch_size, src_seq_length,
                                                    tgt_seq_length), mindspore.float16)

    def construct(self, query, key, value):
        output = super().construct(query.astype(mindspore.float16), key.astype(mindspore.float16),
                                   value.astype(mindspore.float16), self.attention_mask)
        return output[0].astype(query.dtype), output[1]


class Transformer(mindspore.nn.transformer.Transformer):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, activation,
                 custom_encoder, custom_decoder, layer_norm_eps, batch_first, norm_first, device=None, dtype=None):
        batch_size = None
        if batch_size is None or not isinstance(batch_size, int):
            raise ValueError(
                "Please modify the value of batch_size in the previous line."
                " batch_size must be an integer, and it is the batch size of the input tensor.")

        src_seq_length = None
        if src_seq_length is None or not isinstance(src_seq_length, int):
            raise ValueError(
                "Please modify the value of src_seq_length in the previous line. "
                "src_seq_length must be an integer, and it is the sequence length of the query vector.")

        tgt_seq_length = None
        if tgt_seq_length is None or not isinstance(tgt_seq_length, int):
            raise ValueError(
                "Please modify the value of tgt_seq_length in the previous line. "
                "tgt_seq_length must be an integer, and it is the sequence length of the key and value vector.")

        super().__init__(hidden_size=d_model, batch_size=batch_size, ffn_hidden_size=dim_feedforward,
                         src_seq_length=src_seq_length, tgt_seq_length=tgt_seq_length,
                         encoder_layers=num_encoder_layers, decoder_layers=num_decoder_layers, num_heads=nhead,
                         attention_dropout_rate=dropout, hidden_dropout_rate=dropout, hidden_act=activation,
                         post_layernorm_residual=norm_first)

    def construct(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None,
                  gt_key_padding_mask=None, memory_key_padding_mask=None) -> mindspore.Tensor:
        output = super().construct(src, src_mask, tgt, tgt_mask)
        return output


def pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True):
    if not batch_first:
        raise NotImplementedError('batch_first=False is not supported.')
    if not enforce_sorted:
        raise NotImplementedError('enforce_sorted=False is not supported.')

    max_len = lengths[0]
    if isinstance(max_len, mindspore.Tensor):
        max_len = max_len.asnumpy().item()

    if max_len == input.shape[1]:
        return input
    else:
        return input[:, :max_len]


def pad_packed_sequence(sequence, batch_first=False, padding_value=0.0, total_length=None):
    return sequence, sequence.shape


def get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def get_activation_fn(activation: str):
    if activation == "relu":
        return nn_functional.relu
    elif activation == "gelu":
        return nn_functional.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class NonDynamicallyQuantizableLinear(Linear):
    def __init__(self, in_features, out_features, bias=True,
                 device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias=bias)


def pad_sequence(sequences, batch_first=False, padding_value=0.0):
    max_seq_len = max([seq.shape[0] for seq in sequences])
    new_sequences = []
    for seq in sequences:
        if len(seq.shape) == 1:
            if seq.size == 0:
                pad_result = mindspore.ops.zeros((max_seq_len,), seq.dtype)
            else:
                pad_result = mindspore.ops.Pad(paddings=((0, max_seq_len - seq.shape[0]),))(seq)
        else:
            pad_result = nn_functional.x2ms_pad(seq, (0, 0, 0, max_seq_len - seq.shape[0]), value=padding_value)
        new_sequences.append(pad_result)
    if not batch_first:
        return mindspore.numpy.stack(new_sequences, axis=1)
    return mindspore.numpy.stack(new_sequences)


class TransformerEncoderLayer:
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=None, layer_norm_eps=1e-05,
                 batch_first=False, norm_first=False, device=None, dtype=None):
        self.hidden_size = d_model
        self.num_heads = nhead
        self.ffn_hidden_size = dim_feedforward
        self.hidden_act = activation
        self.dropout = dropout


class RealTransformerEncoder(mindspore.nn.transformer.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, batch_size, seq_length):
        ffn_hidden_size = encoder_layer.ffn_hidden_size
        num_heads = encoder_layer.num_heads
        num_layers = num_layers
        hidden_size = encoder_layer.hidden_size
        super().__init__(batch_size, num_layers, hidden_size, ffn_hidden_size, seq_length, num_heads,
                         attention_dropout_rate=0.1, hidden_dropout_rate=0.1, hidden_act="gelu",
                         post_layernorm_residual=False, layernorm_compute_type=mindspore.float32,
                         softmax_compute_type=mindspore.float32, param_init_type=mindspore.float32,
                         lambda_func=None, offset=0, use_past=False)
        self.attention_mask = mindspore.ops.Ones()((batch_size, seq_length,
                                                    seq_length), mindspore.float16)

    def construct(self, hidden_states, attention_mask=None, init_reset=True, batch_valid_length=None):
        hidden_states, present_layer = super().construct(hidden_states, self.attention_mask)
        return hidden_states, present_layer


class TransformerEncoder(mindspore.nn.Cell):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = encoder_layer
        self.num_layers = num_layers

    def construct(self, hidden_states, attention_mask=None, init_reset=True, batch_valid_length=None):
        batch_size, seq_length, _ = hidden_states.shape
        encoder = RealTransformerEncoder(self.encoder_layer, self.num_layers, batch_size, seq_length)
        result, _ = encoder(hidden_states, attention_mask)
        return result


def _check_is_new_parameter_set_to_parameter(target, value):
    return isinstance(target, mindspore.Parameter) and isinstance(value, mindspore.Parameter) \
           and value.name == 'Parameter'


class GRUCell(mindspore.nn.GRUCell):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype=None):
        super().__init__(input_size=input_size, hidden_size=hidden_size, has_bias=bias)


class RNNCell(mindspore.nn.RNNCell):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype=None):
        super(RNNCell, self).__init__(input_size, hidden_size, bias, nonlinearity)


def _trans_unfold_input(param):
    if isinstance(param, tuple) and len(param) == 2:
        return [1, param[0], param[1], 1]
    else:
        return [1, param, param, 1]


class Unfold(mindspore.nn.Unfold):
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1):
        ksize = _trans_unfold_input(kernel_size)
        m_stride = _trans_unfold_input(stride)
        m_dilation = _trans_unfold_input(dilation)
        super().__init__(ksize, m_stride, m_dilation)
        self.padding = padding

    def construct(self, input_x):
        if isinstance(self.padding, tuple) and len(self.padding) == 2:
            paddings = ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1]))
        else:
            paddings = ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))
        real_input = mindspore.ops.pad(input_x, paddings)
        result = super().construct(real_input)
        return result.reshape((result.shape[0], result.shape[1], -1))


if LooseVersion(mindspore.__version__) < LooseVersion('1.8.0'):
    from ...ms_1_7_1.nn import AdaptiveAvgPool1d, Mish
else:
    class AdaptiveAvgPool1d(mindspore.nn.AdaptiveAvgPool1d):
        def __init__(self, output_size):
            super().__init__(output_size)

        def construct(self, input):
            if input.shape[-1] % self.output_size == 0:
                return super().construct(input)
            return adaptive_avg_pool1d(input, self.output_size)


    class Mish(mindspore.nn.Mish):
        def __init__(self, inplace=False):
            super().__init__()


    class Bilinear(mindspore.nn.BiDense):
        def __init__(self, in1_features, in2_features, out_features, bias=True, device=None, dtype=None):
            super().__init__(in1_channels=in1_features, in2_channels=in2_features, out_channels=out_features,
                             has_bias=bias)


    class Hardtanh(mindspore.nn.Hardtanh):
        def __init__(self, min_val=-1.0, max_val=1.0, inplace=False, min_value=None, max_value=None):
            min_val = min_val if min_value is None else min_value
            max_val = max_val if max_value is None else max_value
            super().__init__(min_val=min_val, max_val=max_val)


    class LocalResponseNorm(mindspore.nn.LRN):
        def __init__(self, size, alpha=0.0001, beta=0.75, k=1.0):
            super(LocalResponseNorm, self).__init__(depth_radius=int(size / 2), bias=float(k), alpha=alpha, beta=beta)


    class RReLU(mindspore.nn.RReLU):
        def __init__(self, lower=0.125, upper=0.3333333333333333, inplace=False):
            super(RReLU, self).__init__(lower=lower, upper=upper)


    class Threshold(mindspore.nn.Threshold):
        def __init__(self, threshold, value, inplace=False):
            super(Threshold, self).__init__(threshold, value)
