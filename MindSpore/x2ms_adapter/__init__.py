#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
__all__ = ["save", "load", "load_state_dict", "arange", "cat", "cos", "clamp", "Device", "from_numpy", "flatten",
           "LongTensor", "matmul", "mm", "normal", "ones", "x2ms_pow", "sin", "tanh", "x2ms_tensor", "Tensor",
           "split", 'as_tensor', 'argmax', 'Generator', 'sigmoid', 'rand', 'floor', 'bernoulli', 'equal', 'var_mean',
           'randperm', 'sqrt', 'stack', 'log', 'exp', 'typename', 'is_tensor', 'randn', 'FloatTensor', 'x2ms_max',
           'x2ms_min', 'bmm', 'x2ms_abs', 'square', 'squeeze', 'unsqueeze', 'transpose', 'repeat_interleave', 'div',
           'ones_like', 'where', 'tensordot', 'meshgrid', 'roll', 'linspace', 'full', 'empty', 'x2ms_sum',
           'multinomial', 'gather', 'sort', 'topk', 'x2ms_all', 'cumsum', 'einsum', 'full_like', 'masked_select',
           'x2ms_mean', 'mul', 'isfinite', 'diag', 'acos', 'add', 'argsort', 'asin', 'atan2', 'bincount',
           'broadcast_tensors', 'chunk', 'conj', 'cosh', 'cross', 'cumprod', 'diagflat', 'x2ms_diagonal', 'eq',
           'zeros_like', 'atan', 'unique', 'triu', 'nonzero', 'log2', 'cdist', 'erf', 'softmax', 'eye', 'prod', 'norm',
           'zeros', 'lt', 'ge', 'ne', 'le', 'reshape', 'reminder', 'result_type', 'real', 'reciprocal', 'neg', 'isinf',
           'isnan', 'argmin', 'floor_divide', 'fmod', 'empty_like', 'erfc', 'erfinv', 'expm1', 'flip', 'gt',
           'bitwise_and', 'bitwise_or', 'bitwise_xor', 'bartlett_window', 'blackman_window', 'hamming_window', 'histc',
           'imag', 'ceil', 'lerp', 'log1p', 'logical_and', 'logical_not', 'logical_or', 'logical_xor', 'var', 'unbind',
           'trunc', 'true_divide', 'triu_indices', 'triu', 'tril', 'trapz', 'trapezoid', 'trace', 'tan', 'take', 'lt',
           'ge', 'ne', 'le', 'reshape', 'reminder', 'result_type', 'real', 'reciprocal', 'neg',
           'minimum', 'hann_window', 'dot', 'scatter', 'ger', 'addmm', 'BoolTensor', 'finfo', 'IntTensor',
           'get_rng_state', 'set_rng_state', 'randint', 'randn_like', 'ByteTensor', 'index_select', 'allclose', 't',
           'vstack', 'rsqrt', 'x2ms_round', 'acosh', 'addcmul', 'addcdiv', 'asinh', 'atanh', 'amax', 'amin',
           'cummax', 'cummin', 'logsumexp', 'renorm', 'xlogy', 'sign', 'sinh', 'less', 'narrow', 'tensor_zeros_like',
           'Size', 'DoubleTensor', 'cosine_similarity', 'block_diag', 'cholesky_solve', 'lu_solve', 'x2ms_any',
           'greater', 'greater_equal', 'less_equal', 'not_equal', 'multiply', 'logspace', 'tril_indices', 'vander',
           'atleast_1d', 'atleast_2d', 'atleast_3d', 'column_stack', 'rad2deg', 'outer', 'negative', 'log10',
           'count_nonzero', 'signbit', 'isposinf', 'isin', 'isneginf', 'copysign', 'deg2rad', 'diff', 'gcd',
           'heaviside']

import numbers
import numpy as np
import mindspore
import mindspore.default_config
import mindspore.nn
import mindspore.numpy
import mindspore.dataset.transforms
import mindspore.ops.functional as F
from mindspore.nn import DistributedGradReducer
from mindspore.ops import composite
from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean, _get_parallel_mode)
from mindspore.common.parameter import ParameterTuple
from mindspore.context import ParallelMode
from .core.decorator import x2ms_func_decorator
from .torch_api.save_load import save, load, load_state_dict
from .torch_api import tensor_api
from .torch_api import autograd
from .third_party_adapter import math_api
from .torch_api.nn_api import nn_cell
from .torch_api.nn_api import nn_init
from .torch_api.nn_api import nn_functional
from .core.context import x2ms_context
from .utils.util_api import logger
from .torch_api.torch_base_api import arange, cat, cos, clamp, Device, from_numpy, flatten, \
    LongTensor, matmul, mm, normal, ones, x2ms_pow, sin, tanh, x2ms_tensor, Tensor, zeros, split, as_tensor, dot, \
    x2ms_sum, argmax, Generator, sigmoid, rand, floor, bernoulli, equal, randperm, var_mean, sqrt, stack, log, exp, \
    typename, is_tensor, randn, FloatTensor, x2ms_max, x2ms_min, bmm, x2ms_abs, square, squeeze, unsqueeze, \
    transpose, repeat_interleave, div, ones_like, where, tensordot, meshgrid, roll, linspace, full, empty, \
    multinomial, gather, sort, topk, x2ms_all, cumsum, einsum, full_like, masked_select, x2ms_mean, mul, isfinite, \
    diag, acos, add, argsort, asin, atan2, bincount, broadcast_tensors, chunk, conj, cosh, cross, cumprod, \
    diagflat, x2ms_diagonal, zeros_like, atan, unique, nonzero, log2, cdist, erf, softmax, eye, prod, norm, \
    lt, ge, eq, ne, le, reshape, reminder, result_type, real, reciprocal, neg, isinf, isnan, argmin, floor_divide, \
    fmod, empty_like, erfc, erfinv, expm1, flip, gt, bitwise_and, bitwise_or, bitwise_xor, bartlett_window, \
    blackman_window, hamming_window, histc, imag, ceil, lerp, log1p, logical_and, logical_not, logical_or, \
    logical_xor, var, unbind, trunc, true_divide, triu_indices, triu, tril, trapz, trapezoid, trace, tan, take, \
    minimum, hann_window, scatter, ger, addmm, BoolTensor, finfo, IntTensor, get_rng_state, set_rng_state, randint, \
    randn_like, ByteTensor, index_select, allclose, t, vstack, rsqrt, x2ms_round, acosh, addcmul, addcdiv, asinh, \
    atanh, amax, amin, cummax, cummin, logsumexp, renorm, xlogy, sign, sinh, less, narrow, tensor_zeros_like, Size, \
    DoubleTensor, cosine_similarity, block_diag, cholesky_solve, lu_solve, x2ms_any, greater, greater_equal, \
    less_equal, not_equal, multiply, logspace, tril_indices, vander, atleast_1d, atleast_2d, atleast_3d, \
    column_stack, rad2deg, outer, negative, log10, count_nonzero, signbit, isposinf, isin, isneginf, copysign, \
    deg2rad, diff, gcd, heaviside

# overwrite Magic methods
mindspore.Tensor.__and__ = tensor_api.tensor_and
mindspore.Tensor.__or__ = tensor_api.tensor_or
mindspore.Tensor.__format__ = tensor_api.tensor_format
mindspore.Tensor.__getitem__ = tensor_api.tensor_getitem
mindspore.Tensor.__matmul__ = tensor_api.matmul
mindspore.Tensor.__setitem__ = tensor_api.tensor_setitem
mindspore.Tensor.T = tensor_api.transpose_

mindspore.Tensor.__float__ = lambda t: float(t.asnumpy())
mindspore.Tensor.__int__ = lambda t: int(t.asnumpy())

mindspore.Parameter.__iadd__ = tensor_api.parameter_iadd
mindspore.Parameter.__isub__ = tensor_api.parameter_isub
mindspore.Parameter.__imul__ = tensor_api.parameter_imul
mindspore.Parameter.__idiv__ = tensor_api.parameter_idiv

# overwrite properties
mindspore.Tensor.is_cuda = tensor_api.is_cuda
mindspore.Tensor.data = tensor_api.property_data
mindspore.Tensor.device = tensor_api.property_device
mindspore.Parameter.grad = tensor_api.grad
mindspore.Parameter.grad = tensor_api.set_grad


@property
def parameter_data(self):
    return self


@parameter_data.setter
def set_data(self, new_data):
    self.set_data(new_data)


mindspore.Parameter.data = parameter_data
mindspore.Parameter.data = set_data


def _get_calculate_shape(obj, other):
    if not isinstance(other, mindspore.Tensor):
        return obj.shape
    return np.broadcast_shapes(obj.shape, other.shape)


def _replace_tensor_calculate_func(origin_func_name, output_type=None):
    origin_func = getattr(mindspore.Tensor, origin_func_name)

    def new_func(obj, other):
        if obj.dtype == mindspore.float64:
            obj = obj.astype(mindspore.float32)
        if isinstance(other, np.ndarray):
            other = mindspore.Tensor(other, obj.dtype)
        if obj.size == 0 or (isinstance(other, mindspore.Tensor) and other.size == 0):
            if output_type is None:
                return mindspore.ops.Zeros()(_get_calculate_shape(obj, other), obj.dtype)
            else:
                return mindspore.ops.Zeros()(_get_calculate_shape(obj, other), output_type)
        return origin_func(obj, other)

    setattr(mindspore.Tensor, origin_func_name, new_func)


for func_name in ("__add__", "__sub__", "__mul__", "__truediv__", "__mod__", "__pow__"):
    _replace_tensor_calculate_func(func_name)

for func_name in ("__lt__", "__gt__", "__le__", "__ge__", "__eq__", "__ne__"):
    _replace_tensor_calculate_func(func_name, mindspore.bool_)


class GraphTrainStep(mindspore.nn.TrainOneStepCell):
    def __init__(self, network, optimizer):
        super(GraphTrainStep, self).__init__(network, optimizer)

    def call_construct(self, *inputs):
        new_input = list(mindspore.Tensor(value, dtype=mindspore.float32)
                         if not isinstance(value, mindspore.Tensor)
                         else value
                         for value in inputs)
        return self.__call__(*new_input)

    def construct(self, *inputs):
        output = self.network(*inputs)
        loss = output[0]
        model_output = output[1:]
        sens = (F.fill(loss.dtype, loss.shape, self.sens),)
        for output_value in model_output:
            sens += self._get_sens(output_value)
        grads = self.grad(self.network, self.weights)(*inputs, sens)
        grads = self.grad_reducer(grads)
        F.depend(loss, self.optimizer(grads))
        return output

    @staticmethod
    def _get_sens(value):
        if isinstance(value, mindspore.Tensor):
            return (F.fill(value.dtype, value.shape, 0),)
        if isinstance(value, list):
            sens = []
            for tensor in value:
                sens.append(F.fill(tensor.dtype, tensor.shape, 0))
            return (sens,)
        if isinstance(value, tuple):
            sens = ()
            for tensor in value:
                sens += (F.fill(tensor.dtype, tensor.shape, 0),)
            return (sens,)
        return (0,)


def add_module(obj, name, module):
    setattr(obj, name, module)


classic_cell_init = mindspore.nn.Cell.__init__


def new_cell_init(self, auto_prefix=True, flags=None):
    classic_cell_init(self, auto_prefix, flags)
    self.training = True


# same name and inherit subclass api
mindspore.nn.Cell.add_module = add_module
mindspore.nn.Cell.__init__ = new_cell_init
mindspore.nn.Cell._modules = nn_cell._modules


@property
def is_floating_point(self):
    return self in (mindspore.float16, mindspore.float32, mindspore.float64)


mindspore.dtype.typing.Number.is_floating_point = is_floating_point


def cuda_set_device(device):
    pass


def is_cuda_available():
    """
       Stub function for torch.cuda.is_available.
       get the info from default_config.
    """
    return True


def memory_cached():
    return 0.0


def memory_reserved():
    return 0.0


def max_memory_reserved(device=None):
    return 0.0


def max_memory_allocated(device=None):
    return 0.0


def memory_allocated(device=None):
    return 0.0


def get_device():
    return mindspore.context.get_context('device_target')


@x2ms_func_decorator(mindspore.nn.Cell)
def parameters(obj, *args, **kwargs):
    return get_cell_params(obj, *args, **kwargs)


def get_cell_params(cell, recurse=True):
    return iter(cell.trainable_params(recurse) + cell.untrainable_params(recurse))


@x2ms_func_decorator(mindspore.nn.Cell)
def named_parameters(model, prefix='', recurse=True):
    return list(param for param in model.parameters_and_names(prefix, recurse))


@x2ms_func_decorator(mindspore.nn.Cell)
def named_modules(model, prefix=''):
    return model.cells_and_names(prefix)


@x2ms_func_decorator(mindspore.nn.Cell)
def graph_forward(obj, *args, **kwargs):
    return obj(*args, **kwargs)


@x2ms_func_decorator(mindspore.nn.Cell)
def forward(obj, *args, **kwargs):
    return obj.construct(*args, **kwargs)


@x2ms_func_decorator(mindspore.nn.Cell)
def x2ms_train(obj, *args, **kwargs):
    if len(obj.trainable_params()) > 0:
        if obj not in x2ms_context.amp_model:
            x2ms_context.amp_model.append(obj)
    return obj.set_train(*args, **kwargs)


@x2ms_func_decorator(mindspore.nn.Cell)
def x2ms_eval(obj, *args, **kwargs):
    return obj.set_train(False)


class TrainCellWithoutOptimizer(mindspore.nn.Cell):
    def __init__(self, network, sens=1.0):
        super(TrainCellWithoutOptimizer, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.model_weights = ParameterTuple(parameters(network))
        self.grad = composite.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.parallel_flag = False
        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()
        self.parallel_flag = self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL)
        if self.parallel_flag:
            self.grads_mean = _get_gradients_mean()
            self.device_num = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.model_weights, self.grads_mean, self.device_num)

    def construct(self, *inputs):
        train_loss = self.network(*inputs)
        train_sens = F.fill(train_loss.dtype, train_loss.shape, self.sens)
        grads = self.grad(self.network, self.model_weights)(*inputs, train_sens)
        grads = self.grad_reducer(grads)
        for i, parameter in enumerate(self.model_weights):
            parameter.grad = grads[i]
        return train_loss


_wrapped_model_dict = {}


def train_one_step_cell(model, optimizer=None):
    key = id(model)
    if key in _wrapped_model_dict.keys():
        return _wrapped_model_dict.get(key)
    if x2ms_context.amp_opt_level is None or x2ms_context.amp_model is None:
        if optimizer is None:
            wrapped_model = TrainCellWithoutOptimizer(model)
        else:
            wrapped_model = mindspore.nn.TrainOneStepCell(model, optimizer)
    else:
        if isinstance(x2ms_context.loss_scale, numbers.Number) and x2ms_context.amp_opt_level != "O2":
            wrapped_model = mindspore.amp.build_train_network(model, optimizer, level=x2ms_context.amp_opt_level,
                                                              loss_scale_manager=mindspore.FixedLossScaleManager(
                                                                  x2ms_context.loss_scale))
        else:
            wrapped_model = mindspore.amp.build_train_network(model, optimizer, level=x2ms_context.amp_opt_level)
    _wrapped_model_dict[key] = wrapped_model
    return wrapped_model


def graph_train_one_step_cell(model, optimizer):
    key = id(model)
    if key in _wrapped_model_dict.keys():
        return _wrapped_model_dict.get(key)
    if x2ms_context.amp_opt_level is None or x2ms_context.amp_model is None:
        wrapped_model = GraphTrainStep(model, optimizer)
    else:
        raise NotImplementedError("Graph mode does not currently support Mixed precision")
    _wrapped_model_dict[key] = wrapped_model
    return wrapped_model


def load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True, check_hash=False):
    """
    Current not support 'model_dir', 'map_location', 'progress', 'check_hash' parameter.
    """
    logger.warning("Not support load_state_dict_from_url now")
    return {}


def to(obj, *args, **kwargs):
    if isinstance(obj, mindspore.nn.Cell):
        return _cell_to(obj, *args, **kwargs)
    elif isinstance(obj, mindspore.Tensor):
        return _tensor_to(obj, *args, **kwargs)
    else:
        return obj.to(*args, **kwargs)


def _cell_to(obj, *args, **kwargs):
    if args:
        param = args[0]
        if isinstance(param, mindspore.Type) and param in (mindspore.float16, mindspore.float32):
            return obj.to_float(dst_type=param)
        if isinstance(param, mindspore.Tensor) and param.dtype in (mindspore.float16, mindspore.float32):
            return obj.to_float(dst_type=param.dtype)
    if len(args) > 1:
        param = args[1]
        if param in (mindspore.float16, mindspore.float32):
            return obj.to_float(dst_type=param)
    if 'dtype' in kwargs.keys() and kwargs['dtype'] in (mindspore.float16, mindspore.float32):
        return obj.to_float(dst_type=kwargs['dtype'])
    if 'other' in kwargs.keys() and kwargs['other'].dtype in (mindspore.float16, mindspore.float32):
        return obj.to_float(dst_type=kwargs['other'])
    return obj


def _tensor_to(obj, *args, **kwargs):
    if args:
        param = args[0]
        if isinstance(param, mindspore.common.Type):
            return obj.astype(dtype=param)
        if isinstance(param, mindspore.Tensor):
            return obj.astype(dtype=param.dtype)
    if len(args) > 1:
        return obj.astype(dtype=args[1])
    if 'dtype' in kwargs.keys():
        return obj.astype(dtype=kwargs['dtype'])
    if 'other' in kwargs.keys():
        return obj.astype(dtype=kwargs['other'].dtype)
    return obj


def get_device_properties(device):
    return CUDADeviceProperty()


def convert_sync_batchnorm(module, process_group=None):
    return module


class CUDADeviceProperty:
    def __init__(self):
        device_target = mindspore.context.get_context('device_target')
        device_id = mindspore.context.get_context('device_id')
        self.name = f'{device_target}:{device_id}'
        self.total_memory = 0
