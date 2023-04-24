#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
from distutils.version import LooseVersion
from typing import NamedTuple
import numbers
import mindspore
import mindspore.common.initializer
import mindspore.numpy
import numpy as np
import mindspore.nn.probability.distribution as msd
import mindspore.scipy as scipy

from ..utils.util_api import float_tensor_2_bool_tensor, out_adaptor, check_input_dtype
from ..core.context import x2ms_context
from ..third_party_adapter.numpy_api import TensorNumpy

NamedtupleValuesIndices = NamedTuple("namedtuple_values_indices",
                                     [("values", mindspore.Tensor), ("indices", mindspore.Tensor)])


def scatter(input, dim, index, src, reduce=None):
    if reduce == 'multiply':
        raise NotImplementedError('Mindspore 1.9 and previous versions do not support parameter reduce="multiply".')
    reduction = 'none' if reduce is None else reduce
    if isinstance(src, numbers.Number):
        src = mindspore.numpy.full(index.shape, src, input.dtype)
    elif src.shape != index.shape:
        start_index = (0,) * src.ndim
        src = mindspore.ops.slice(src, start_index, index.shape)

    return mindspore.ops.tensor_scatter_elements(input, index, src, dim, reduction)


def index_select(input, dim, index, *, out=None):
    if index.dim() == 0:
        index = index.reshape(1)
    result = mindspore.ops.gather(input, index, axis=dim)
    return out_adaptor(result, out)


def vstack(tensors, *, out=None):
    tensors = list(tensors)
    if tensors and len(tensors[0].shape) == 1:
        for i, tensor in enumerate(tensors):
            tensors[i] = mindspore.ops.expand_dims(tensor, axis=0)
    tensors = tuple(tensors)
    return out_adaptor(mindspore.ops.concat(tensors, axis=0), out)


def amax(input, dim, keepdim=False, *, out=None):
    result = mindspore.ops.amax(input, dim, keep_dims=keepdim)
    return out_adaptor(result, out)


def amin(input, dim, keepdim=False, *, out=None):
    result = mindspore.ops.amin(input, dim, keep_dims=keepdim)
    return out_adaptor(result, out)


def cat(tensors, dim=0, *, out=None):
    if tensors and isinstance(tensors, (list, tuple)) and isinstance(tensors[0], np.ndarray):
        return TensorNumpy.create_tensor_numpy(np.concatenate(tensors, axis=dim, out=out))
    new_tensors = []
    for tensor in tensors:
        if tensor.size != 0:
            new_tensors.append(tensor)
    if len(new_tensors) == 1:
        return new_tensors[0]
    if len(new_tensors) == 0:
        new_shape = list(tensors[0].shape)
        new_shape[dim] = 0
        for tensor in tensors:
            new_shape[dim] += tensor.shape[dim]
        return mindspore.ops.Zeros()(tuple(new_shape), mindspore.float32)
    for tensor in new_tensors:
        if isinstance(tensor, mindspore.Parameter):
            return mindspore.ops.Concat(axis=dim)(new_tensors)
    return mindspore.numpy.concatenate(new_tensors, axis=dim)


def flatten(input_tensor, start_dim=0, end_dim=-1):
    shape_tuple = input_tensor.shape
    _start_dim = start_dim if start_dim >= 0 else (start_dim + input_tensor.ndim)
    _end_dim = end_dim if end_dim >= 0 else (end_dim + input_tensor.ndim)
    new_dim = 1
    for idx in range(_start_dim, _end_dim + 1):
        new_dim *= shape_tuple[idx]
    new_shape_list = []
    for i in shape_tuple[:_start_dim]:
        new_shape_list.append((i,))
    new_shape_list.append((new_dim,))
    new_shape_tuple = ()
    for i in new_shape_list:
        new_shape_tuple += i

    if 0 in new_shape_tuple:
        return mindspore.ops.Zeros()(new_shape_tuple, input_tensor.dtype)

    reshape_ops = mindspore.ops.Reshape()
    return reshape_ops(input_tensor, new_shape_tuple)


def flip(input, dims):
    return mindspore.numpy.flip(input, dims)


def from_numpy(ndarray):
    if x2ms_context.get_is_during_transform():
        return TensorNumpy.create_tensor_numpy(ndarray)
    if ndarray.dtype == np.float64:
        ndarray = ndarray.astype(np.float32)
    return mindspore.Tensor.from_numpy(np.ascontiguousarray(ndarray))


def zeros(*args, **kwargs):
    size = kwargs.pop('size', None)
    if size is not None:
        return _zeros(size, **kwargs)
    return _zeros(*args, **kwargs)


def _zeros(*size, out=None, dtype=None, layout=None, device=None, requires_grad=False):
    if x2ms_context.get_is_during_transform():
        dtype = np.float if dtype is None else dtype
        return TensorNumpy.create_tensor_numpy(np.zeros(_tuple(size), dtype=dtype))
    if dtype is None:
        dtype = mindspore.float32
    return mindspore.ops.Zeros()(_tuple(size), dtype)


def ones(*size, out=None, dtype=None, layout=None, device=None, requires_grad=False):
    if dtype is None:
        dtype = mindspore.float32
    result = mindspore.ops.Ones()(_tuple(size), dtype)
    return out_adaptor(result, out)


def arange(start, end=None, step=1, out=None, dtype=None, layout=None, device=None, requires_grad=False):
    if end is None:
        result = mindspore.numpy.arange(start, step=step, dtype=dtype)
    else:
        result = mindspore.numpy.arange(start, stop=end, step=step, dtype=dtype)
    return out_adaptor(result, out)


def x2ms_tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False):
    """
    Parameter 'device', 'pin_memory' are not supported.
    """
    if isinstance(data, list) and isinstance(data[0], mindspore.Tensor):
        temp = tuple(data)
        data = mindspore.ops.stack(temp)

    if isinstance(data, range):
        data = list(data)

    if x2ms_context.get_is_during_transform():
        if isinstance(data, np.ndarray):
            return TensorNumpy.create_tensor_numpy(data)
        return TensorNumpy.create_tensor_numpy(np.array(data))

    result = mindspore.Tensor(data)
    if dtype is not None:
        result = result.astype(dtype)
    if mindspore.context.get_context('device_target') == 'Ascend' and result.dtype == mindspore.float64:
        result = result.astype(mindspore.float32)
    if not requires_grad:
        result = mindspore.ops.stop_gradient(result)

    return result


def matmul(input, other, out=None):
    result = mindspore.ops.matmul(input, other)
    return out_adaptor(result, out)


def tanh(input, out=None):
    result = mindspore.ops.Tanh()(input)
    return out_adaptor(result, out)


def sin(input, out=None):
    result = mindspore.ops.Sin()(input)
    return out_adaptor(result, out)


def cos(input, out=None):
    result = mindspore.ops.Cos()(input)
    return out_adaptor(result, out)


def acos(input, out=None):
    result = mindspore.ops.ACos()(input)
    return out_adaptor(result, out)


def add(input, other, *, alpha=1, out=None):
    result = mindspore.ops.Add()(input, other * alpha)
    return out_adaptor(result, out)


def argsort(input, dim=-1, descending=False):
    origin_type = input.dtype
    converted_type = origin_type
    if origin_type not in (mindspore.float16, mindspore.float32):
        converted_type = mindspore.float32

    return mindspore.ops.Sort(axis=dim, descending=descending)(input.astype(converted_type))[-1]


def asin(input, *, out=None):
    result = mindspore.ops.Asin()(input)
    return out_adaptor(result, out)


def atan2(input, other, *, out=None):
    result = mindspore.ops.Atan2()(input, other)
    return out_adaptor(result, out)


def bincount(input, weights=None, minlength=0):
    return mindspore.numpy.bincount(input, weights=weights, minlength=minlength)


def broadcast_tensors(*inputs):
    return mindspore.numpy.broadcast_arrays(*inputs)


def bartlett_window(window_length, periodic=True, *, dtype=None, layout=None, device=None,
                    requires_grad=False):
    is_args_not_none = layout is not None or device is not None
    if periodic or is_args_not_none or requires_grad is not False:
        raise NotImplementedError("Parameters 'periodic', 'layout', 'device' and 'requires_grad' "
                                  "are not supported. Use default settings.")
    result = mindspore.numpy.bartlett(window_length)
    if dtype is not None:
        result.astype(dtype)
    return result


def blackman_window(window_length, periodic=True, *, dtype=None, layout=None, device=None,
                    requires_grad=False):
    is_args_not_none = layout is not None or device is not None
    if periodic or is_args_not_none or requires_grad is not False:
        raise NotImplementedError("Parameters 'periodic', 'layout', 'device' and 'requires_grad' "
                                  "are not supported. Use default settings.")
    result = mindspore.numpy.blackman(window_length)
    if dtype is not None:
        result.astype(dtype)
    return result


def hamming_window(window_length, periodic=True, *, dtype=None, layout=None, device=None,
                   requires_grad=False):
    is_args_not_none = layout is not None or device is not None
    if periodic or is_args_not_none or requires_grad is not False:
        raise NotImplementedError("Parameters 'periodic', 'layout', 'device' and 'requires_grad' "
                                  "are not supported. Use default settings.")
    result = mindspore.numpy.hamming(window_length)
    if dtype is not None:
        result.astype(dtype)
    return result


def hann_window(window_length, periodic=True, *, dtype=None, layout=None, device=None,
                requires_grad=False):
    is_args_not_none = layout is not None or device is not None
    if periodic or is_args_not_none or requires_grad is not False:
        raise NotImplementedError("Parameters 'periodic', 'layout', 'device' and 'requires_grad' "
                                  "are not supported. Use default settings.")
    result = mindspore.numpy.hanning(window_length)
    if dtype is not None:
        result.astype(dtype)
    return result


def histc(input, bins=100, min=0, max=0, *, out=None):
    input_min, input_max = input.min().asnumpy(), input.max().asnumpy()
    if min == 0 == max:
        min, max = input_min, input_max
    result = mindspore.ops.HistogramFixedWidth(bins)(input, mindspore.Tensor([min, max], input.dtype))
    if input_min < min:
        less_than_min = ((input < min).astype(input.dtype)).sum()
        result[0] = result[0] - less_than_min
    if input_max < max:
        greater_than_max = ((input > max).astype(input.dtype)).sum()
        result[-1] = result[-1] - greater_than_max
    return out_adaptor(result, out)


def imag(input):
    return mindspore.ops.Imag()(input)


def isinf(input):
    return mindspore.ops.IsInf()(input)


def isnan(input):
    return mindspore.ops.IsNan()(input)


def bitwise_and(input, other, *, out=None):
    if input.dtype == mindspore.bool_:
        bitwise_and_func = mindspore.ops.LogicalAnd()
    else:
        if input.dtype not in (mindspore.int16, mindspore.int32, mindspore.uint16):
            input = input.astype(mindspore.int32)
        bitwise_and_func = mindspore.ops.BitwiseAnd()
    result = bitwise_and_func(input, other)
    return out_adaptor(result, out)


def bitwise_or(input, other, *, out=None):
    if input.dtype == mindspore.bool_:
        bitwise_or_func = mindspore.ops.LogicalOr()
    else:
        if input.dtype not in (mindspore.int16, mindspore.int32, mindspore.uint16):
            input = input.astype(mindspore.int32)
        bitwise_or_func = mindspore.ops.BitwiseOr()
    result = bitwise_or_func(input, other)
    return out_adaptor(result, out)


def bitwise_xor(input, other, *, out=None):
    if input.dtype == mindspore.bool_:
        bitwise_xor_func = mindspore.ops.LogicalXor()
    else:
        if input.dtype not in (mindspore.int16, mindspore.int32, mindspore.uint16):
            input = input.astype(mindspore.int32)
        bitwise_xor_func = mindspore.ops.BitwiseXor()
    result = bitwise_xor_func(input, other)
    return out_adaptor(result, out)


def ceil(input, *, out=None):
    if input.dtype not in (mindspore.float32, mindspore.float16):
        input = input.astype(mindspore.float32)
    ceil_func = mindspore.ops.Ceil()
    result = ceil_func(input)
    return out_adaptor(result, out)


def chunk(input, chunks, dim=0):
    if chunks <= 0:
        raise ValueError(f"Parameter 'chunks' must be greater than 0, got {chunks}")
    return split(input, (input.shape[dim] + chunks - 1) // chunks, dim)


def conj(input):
    return mindspore.ops.Conj()(input)


def cosine_similarity(y_true, y_pred, dim=-1):
    l2_normalize_obj = mindspore.ops.L2Normalize(axis=dim)
    y_pred = l2_normalize_obj(y_pred)
    y_true = l2_normalize_obj(y_true)
    return mindspore.ops.ReduceSum()(y_true * y_pred, axis=dim)


def multiply(input, other, *, out=None):
    result = mindspore.numpy.multiply(input, other)
    return out_adaptor(result, out)


def cosh(input, *, out=None):
    result = mindspore.ops.Cosh()(input)
    return out_adaptor(result, out)


def cross(input, other, dim=None, *, out=None):
    if dim is None:
        raise NotImplementedError("MindSpore does not support param dim is None.")
    result = mindspore.numpy.cross(input, other, axis=dim)
    return out_adaptor(result, out)


def cumprod(input, dim, *, dtype=None, out=None):
    result = mindspore.ops.CumProd(exclusive=False, reverse=False)(input, dim)
    if dtype:
        result = result.astype(dtype)
    return out_adaptor(result, out)


def diagflat(input, offset=0):
    return mindspore.numpy.diagflat(input, k=offset)


def x2ms_diagonal(input, offset=0, dim1=0, dim2=1):
    return mindspore.numpy.diagonal(input, offset=offset, axis1=dim1, axis2=dim2)


def eq(input, other, *, out=None):
    result = mindspore.ops.Equal()(input, other)
    return out_adaptor(result, out)


def x2ms_pow(input, exponent, out=None):
    result = mindspore.ops.Pow()(input, exponent)
    return out_adaptor(result, out)


def clamp(input, min=None, max=None, out=None):
    if isinstance(input, np.ndarray):
        result = np.clip(input, min, max, out=out)
        return result
    if input.size == 0:
        result = input
    else:
        result = mindspore.numpy.clip(input, min, max)
    return out_adaptor(result, out)


def normal(mean, std, *, generator=None, out=None):
    if isinstance(mean, float):
        input_shape = std.shape
        mean = mindspore.Tensor(mean, mindspore.float32)
    elif isinstance(std, float):
        input_shape = mean.shape
        std = mindspore.Tensor(std, mindspore.float32)
    else:
        input_shape = mean.shape
    mean = mean.astype(mindspore.float32)
    std = std.astype(mindspore.float32)
    result = mindspore.ops.normal(input_shape, mean, std)
    return out_adaptor(result, out)


def mm(input, mat2, *, out=None):
    result = mindspore.ops.matmul(input, mat2)
    return out_adaptor(result, out)


def split(tensor, split_size_or_sections, dim=0):
    if split_size_or_sections == 0:
        raise ValueError("Input parameter split_size_or_sections cannot be 0 in split function.")
    if isinstance(split_size_or_sections, int) and tensor.shape[dim] % split_size_or_sections == 0:
        return mindspore.numpy.split(tensor, int(tensor.shape[dim] / split_size_or_sections), dim)
    if isinstance(split_size_or_sections, int) and tensor.shape[dim] % split_size_or_sections != 0:
        split_size_list = list(map(int, np.arange(0, tensor.shape[dim], split_size_or_sections)))[1:]
        return mindspore.numpy.split(tensor, split_size_list, dim)
    split_indices = np.cumsum([size for size in split_size_or_sections if size > 0]).tolist()[:-1]
    non_empty_result = mindspore.numpy.split(tensor, split_indices, axis=dim)
    if len(split_indices) == len(split_size_or_sections) - 1:
        return non_empty_result
    num = 0
    result = []
    if dim < 0:
        dim += tensor.ndim
    empty_shape = [0 if i == dim else tensor.shape[i] for i in range(len(tensor.shape))]
    empty_tensor = mindspore.ops.Zeros()(tuple(empty_shape), tensor.dtype)
    for size in split_size_or_sections:
        if size == 0:
            result.append(empty_tensor)
        else:
            result.append(non_empty_result[num])
            num += 1
    return tuple(result)


def _tuple(size):
    size = tuple(np.array(size).tolist())
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        return tuple(size[0])
    return size


def as_tensor(data, dtype=None, device=None):
    if x2ms_context.get_is_during_transform():
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        return TensorNumpy.create_tensor_numpy(data)

    if isinstance(data, np.ndarray) and data.size == 0:
        return mindspore.ops.Zeros()(data.shape, dtype)

    if isinstance(data, mindspore.Tensor) and data.dtype == dtype:
        return data
    return mindspore.Tensor(data, dtype=dtype)


def dot(input, tensor):
    support_dtype = (mindspore.float16, mindspore.float32)
    input_trans_flag, input, input_origin_type = check_input_dtype(input, support_dtype)
    tensor_trans_flag, tensor, tensor_origin_type = check_input_dtype(tensor, support_dtype)
    output = mindspore.ops.tensor_dot(input, tensor, axes=1)
    if input_origin_type and tensor_trans_flag:
        output = output.astype(input_origin_type)
    return output


def x2ms_sum(input, dim=None, keepdim=False, dtype=None, axis=None):
    input_type = input.dtype
    if dim is None and axis:
        dim = axis
    if dim is None:
        result = mindspore.ops.ReduceSum(keep_dims=keepdim)(input.astype(mindspore.float32))
    else:
        result = mindspore.ops.ReduceSum(keep_dims=keepdim)(input.astype(mindspore.float32), dim)
    if dtype:
        result = result.astype(dtype)
    elif input_type == mindspore.bool_:
        result = result.astype(mindspore.int32)
    else:
        result = result.astype(input_type)

    return result


def argmax(input, dim=None, keepdim=False):
    return input.argmax(axis=dim)


def argmin(input, dim=None, keepdim=False):
    if dim is None:
        input = input.flatten()
        axis = 0
    else:
        axis = dim
    return mindspore.ops.ArgMinWithValue(axis=axis, keep_dims=keepdim)(input)[0]


def sigmoid(data, out=None):
    if data.dtype not in (mindspore.float16, mindspore.float32):
        data = data.astype(mindspore.float32)
    if data.size == 0:
        result = mindspore.ops.Zeros()(data.shape, data.dtype)
    else:
        result = mindspore.ops.Sigmoid()(data)
    return out_adaptor(result, out)


def rand(*size, out=None, dtype=None, layout=None, device=None, requires_grad=False, generator=None):
    shape = _tuple(size)
    if dtype is None:
        dtype = mindspore.float32
    data = mindspore.Tensor(np.random.rand(*shape), dtype=dtype)
    return out_adaptor(data, out)


def floor(data, out=None):
    support_dtype = (mindspore.float16, mindspore.float32)
    trans_flag, data, origin_type = check_input_dtype(data, support_dtype)
    result = mindspore.ops.Floor()(data)
    if trans_flag:
        result = result.astype(origin_type)
    return out_adaptor(result, out)


def floor_divide(input, other, *, out=None):
    floor_divide_func = mindspore.ops.Div()
    result = mindspore.numpy.trunc(floor_divide_func(input, other))
    return out_adaptor(result, out)


def fmod(input, other, *, out=None):
    result = mindspore.ops.Mod()(input, other)
    return out_adaptor(result, out)


def gt(input, other, *, out=None):
    result = mindspore.ops.gt(input, other)
    return out_adaptor(result, out)


class Device:
    def __init__(self, *args, **kwargs):
        pass

    def __format__(self, format_spec):
        device_target = mindspore.context.get_context('device_target')
        device_id = mindspore.context.get_context('device_id')
        return f'{device_target}:{device_id}'

    @property
    def type(self):
        return mindspore.context.get_context('device_target')

    @property
    def index(self):
        return mindspore.context.get_context('device_id')


class _TypedStorage:
    def __init__(self, *args, **kwargs):
        if 'nplist' in kwargs:
            self._storage = kwargs.get('nplist')
        pass

    @property
    def storage(self):
        return self._storage or np.array([], dtype=np.float32)

    @classmethod
    def from_buffer(cls, *data, **kwargs):
        if cls == _TypedStorage:
            raise RuntimeError(
                'from_buffer: only supported for subclasses of _TypedStorage')

        if 'dtype' in kwargs or len(data) == 5:
            raise RuntimeError((
                "from_buffer: 'dtype' can only be specified in "
                "_UntypedStorage.from_buffer"))

        if len(data) == 0:
            raise TypeError(
                'function missing required argument "buffer" (pos 1)')

        _storage = np.frombuffer(data[0], dtype=cls().dtype)
        return cls(nplist=_storage)


class ByteStorage(_TypedStorage):
    @property
    def dtype(self):
        return np.uint8


class Tensor:
    def __new__(cls, *data, dtype=None, device=None, requires_grad=False, pin_memory=False):
        if x2ms_context.get_is_during_transform():
            if len(data) == 0:
                return TensorNumpy.create_tensor_numpy(np.array([], dtype=np.float32))
            if dtype is not None:
                dtype = mindspore.dtype_to_nptype(dtype)
            if isinstance(data[0], int):
                _dtype = np.float32 if dtype is None else dtype
                tensor = np.zeros(shape=data, dtype=_dtype)
            elif isinstance(data[0], _TypedStorage):
                tensor = np.array(data[0].storage, dtype=dtype)
            else:
                tensor = np.array(data[0], dtype=dtype)
            return TensorNumpy.create_tensor_numpy(tensor)
        if len(data) == 0:
            return mindspore.Tensor([], dtype=mindspore.float32)
        if isinstance(data[0], int):
            _dtype = mindspore.float32 if dtype is None else dtype
            tensor = mindspore.Tensor(shape=data, dtype=_dtype, init=mindspore.common.initializer.Zero())
            tensor.init_data()
        elif isinstance(data[0], _TypedStorage):
            tensor = mindspore.Tensor(input_data=data[0].storage, dtype=dtype)
        else:
            tensor = mindspore.Tensor(input_data=data[0], dtype=dtype)
        return tensor


class LongTensor(Tensor):
    def __new__(cls, *args, **kwargs):
        kwargs.update({'dtype': mindspore.int64})
        return super().__new__(cls, *args, **kwargs)


class ByteTensor(Tensor):
    def __new__(cls, *args, **kwargs):
        kwargs.update({'dtype': mindspore.uint8})
        return super().__new__(cls, *args, **kwargs)


class FloatTensor(Tensor):
    def __new__(cls, *args, **kwargs):
        kwargs.update({'dtype': mindspore.float32})
        return super().__new__(cls, *args, **kwargs)


class IntTensor(Tensor):
    def __new__(cls, *args, **kwargs):
        kwargs.update({'dtype': mindspore.int32})
        return super().__new__(cls, *args, **kwargs)


class BoolTensor(Tensor):
    def __new__(cls, *args, **kwargs):
        kwargs.update({'dtype': mindspore.bool_})
        return super().__new__(cls, *args, **kwargs)


class DoubleTensor(Tensor):
    def __new__(cls, *args, **kwargs):
        kwargs.update({'dtype': mindspore.float32})
        return super().__new__(cls, *args, **kwargs)


class Generator:
    def __init__(self, *args, **kwargs):
        pass

    def manual_seed(self, seed):
        pass


def bernoulli(input, *, generator=None, out=None):
    input = input.clip(1e-6, 1 - 1e-6)
    bernoulli_distribution = msd.Bernoulli(input, dtype=mindspore.float32)
    result = bernoulli_distribution.sample()
    return out_adaptor(result, out)


def equal(input, other):
    equal_ops = mindspore.ops.Equal()
    return bool(equal_ops(input, other).all())


def logspace(start, end, steps, base=10.0, *, out=None, dtype=None, layout=None, device=None, requires_grad=False):
    if layout is not None or device is not None:
        raise NotImplementedError("Parameters 'layout', 'device', are not supported. Use default settings.")
    result = mindspore.numpy.logspace(start, end, num=steps, endpoint=True, base=base, dtype=dtype, axis=0)
    if not requires_grad:
        result = mindspore.ops.stop_gradient(result)
    return out_adaptor(result, out)


def tril_indices(row, col, offset=0, *, dtype=None, device=None, layout=None):
    if device is not None or layout is not None:
        raise NotImplementedError("'device' and 'layout' are not supported in mindspore.")
    result = mindspore.numpy.stack(mindspore.numpy.tril_indices(row, offset, col))
    if dtype is not None:
        result = result.astype(dtype)
    return result


def vander(x, N=None, increasing=False):
    return mindspore.numpy.vander(x, N, increasing)


def atleast_1d(*tensors):
    if len(tensors) == 1:
        tensors = tensors[0]
    if isinstance(tensors, (list, tuple)):
        return mindspore.numpy.atleast_1d(*tensors)
    return mindspore.numpy.atleast_1d(tensors)


def atleast_2d(*tensors):
    if len(tensors) == 1:
        tensors = tensors[0]
    if isinstance(tensors, (list, tuple)):
        return mindspore.numpy.atleast_2d(*tensors)
    return mindspore.numpy.atleast_2d(tensors)


def atleast_3d(*tensors):
    if len(tensors) == 1:
        tensors = tensors[0]
    if isinstance(tensors, (list, tuple)):
        return mindspore.numpy.atleast_3d(*tensors)
    return mindspore.numpy.atleast_3d(tensors)


def column_stack(tensors, *, out=None):
    result = mindspore.numpy.column_stack(tensors)
    return out_adaptor(result, out)


def rad2deg(input, *, out=None):
    result = mindspore.numpy.rad2deg(input)
    return out_adaptor(result, out)


def outer(input, vec2, *, out=None):
    result = mindspore.numpy.outer(input, vec2)
    return out_adaptor(result, out)


def negative(input, *, out=None):
    result = mindspore.numpy.negative(input)
    return out_adaptor(result, out)


def randperm(n, *, generator=None, out=None, dtype=mindspore.int32, layout=None, device=None, requires_grad=False,
             pin_memory=False):
    if n == 0:
        result = mindspore.ops.Zeros()((0,), dtype)
    else:
        result = mindspore.ops.Randperm(max_length=n)(mindspore.Tensor([n], dtype))
    if not requires_grad:
        result = mindspore.ops.stop_gradient(result)
    return out_adaptor(result, out)


def var_mean(input, dim, unbiased, keepdim=False, *, out=None):
    _dim = tuple(dim) if isinstance(dim, list) else dim
    result = input.var(axis=_dim, keepdims=keepdim), input.mean(axis=_dim, keep_dims=keepdim)
    return out_adaptor(result, out)


def sqrt(input, *, out=None):
    result = mindspore.ops.functional.sqrt(input)
    return out_adaptor(result, out)


def stack(tensors, dim=0, out=None):
    if x2ms_context.get_is_during_transform():
        return TensorNumpy.create_tensor_numpy(np.stack(list(tensors), axis=dim))
    result = mindspore.numpy.stack(list(tensors), axis=dim)
    # see issue here: https://gitee.com/mindspore/mindspore/issues/I57FPA
    if not isinstance(result, mindspore.Tensor):
        result = mindspore.Tensor(result)
    return out_adaptor(result, out)


def log(input, out=None):
    if isinstance(input, np.ndarray):
        result = np.log(input, out=out)
        return result
    else:
        result = mindspore.ops.Log()(input)
    return out_adaptor(result, out)


def exp(input, out=None):
    result = mindspore.ops.Exp()(input)
    return out_adaptor(result, out)


def typename(obj):
    return mindspore.get_py_obj_dtype(obj)


def is_tensor(obj):
    return isinstance(obj, mindspore.Tensor)


def randn(*size, out=None, dtype=None, layout=None, device=None, requires_grad=False, generator=None):
    """
    normal distribution with mean 0 and variance 1
    """
    shape = _tuple(size)
    if generator:
        data = mindspore.ops.StandardNormal(generator.seed)(shape)
    else:
        data = mindspore.ops.StandardNormal()(shape)
    if dtype is not None:
        data = data.astype(dtype)
    return out_adaptor(data, out)


def x2ms_max(*args, **kwargs):
    if isinstance(args[0], np.ndarray):
        if len(args) >= 2 and isinstance(args[1], np.ndarray):
            return np.maximum(*args, **kwargs)
        else:
            return numpy_max(*args, **kwargs)

    if (len(args) == 2 and isinstance(args[1], mindspore.Tensor)) or "other" in kwargs:
        return maximum(*args, **kwargs)
    return x2ms_inner_max(*args, **kwargs)


def maximum(input, other, *, out=None):
    result = mindspore.numpy.maximum(input, other)
    return out_adaptor(result, out)


def numpy_max(array, dim=None, keepdim=False):
    if dim is None:
        return array.max(axis=dim, keepdims=keepdim)
    return array.max(axis=dim, keepdims=keepdim), array.argmax(axis=dim)


def x2ms_inner_max(input, dim=None, keepdim=False, *, out=None):
    if dim is None:
        origin_type = input.dtype
        convert_type = input.dtype
        if input.dtype in (mindspore.int64, mindspore.uint64, mindspore.int32, mindspore.uint32, mindspore.bool_):
            convert_type = mindspore.float32
        return input.astype(convert_type).max().astype(origin_type)
    else:
        max_ops = mindspore.ops.ArgMaxWithValue(axis=dim, keep_dims=keepdim)
        index, output = max_ops(input)
        return output, index


def x2ms_min(*args, **kwargs):
    if isinstance(args[0], np.ndarray):
        if len(args) >= 2 and isinstance(args[1], np.ndarray):
            return np.minimum(*args, **kwargs)
        else:
            return numpy_min(*args, **kwargs)
    if (len(args) == 2 and isinstance(args[1], mindspore.Tensor)) or "other" in kwargs:
        return minimum(*args, **kwargs)
    return x2ms_inner_min(*args, **kwargs)


def minimum(input, other, *, out=None):
    result = mindspore.numpy.minimum(input, other)
    return out_adaptor(result, out)


def numpy_min(array, dim=None, keepdim=False):
    if dim is None:
        return array.min(axis=dim, keepdims=keepdim)
    return array.min(axis=dim, keepdims=keepdim), array.argmin(axis=dim)


def x2ms_inner_min(input, dim=None, keepdim=False, *, out=None):
    if dim is None:
        origin_type = input.dtype
        convert_type = input.dtype
        if input.dtype in (mindspore.int64, mindspore.uint64, mindspore.bool_):
            convert_type = mindspore.float32
        return input.astype(convert_type).min().astype(origin_type)
    else:
        min_ops = mindspore.ops.ArgMinWithValue(axis=dim, keep_dims=keepdim)
        index, output = min_ops(input)
        return NamedtupleValuesIndices(output, index)


def bmm(input, mat2, *, out=None):
    batmatmul = mindspore.ops.BatchMatMul()
    result = batmatmul(input, mat2)
    return out_adaptor(result, out)


def x2ms_abs(input, *, out=None):
    result = mindspore.ops.Abs()(input)
    return out_adaptor(result, out)


def square(input, *, out=None):
    result = mindspore.ops.Square()(input)
    return out_adaptor(result, out)


def squeeze(input, dim=None, out=None):
    if dim is not None and input.shape[dim] != 1:
        result = input
    elif dim is None:
        result = mindspore.ops.Squeeze()(input)
    else:
        result = mindspore.ops.Squeeze(axis=dim)(input)
    return out_adaptor(result, out)


def unsqueeze(input, dim):
    expand_dim = mindspore.ops.ExpandDims()
    return expand_dim(input, dim)


def transpose(input, dim0, dim1):
    dim = input.dim()
    _dim0 = dim0 if dim0 >= 0 else (dim0 + dim)
    _dim1 = dim1 if dim1 >= 0 else (dim1 + dim)
    dim_list = list(range(dim))
    dim_list[_dim0] = _dim1
    dim_list[_dim1] = _dim0
    if input.dtype == mindspore.bool_:
        input = input.astype(mindspore.int32)
        result = mindspore.ops.Transpose()(input, tuple(dim_list))
        return result.astype(mindspore.bool_)
    return mindspore.ops.Transpose()(input, tuple(dim_list))


def repeat_interleave(input, repeats, dim=None, *, output_size=None):
    if dim is None:
        if input.dtype == mindspore.bool_:
            input = input.astype(mindspore.int32)
            result = mindspore.ops.repeat_elements(input.view(-1), rep=repeats)
            return result.astype(mindspore.bool_)
        return mindspore.ops.repeat_elements(input.view(-1), rep=repeats)
    if input.dtype == mindspore.bool_:
        input = input.astype(mindspore.int32)
        result = mindspore.ops.repeat_elements(input, rep=repeats, axis=dim)
        return result.astype(mindspore.bool_)
    return mindspore.ops.repeat_elements(input, rep=repeats, axis=dim)


def div(input, other, *, rounding_mode=None, out=None):
    result = mindspore.ops.Div()(input, other)
    return out_adaptor(result, out)


def ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=None):
    _input = input
    if _input.dtype == mindspore.bool_:
        if _input.size == 0:
            return mindspore.ops.Ones()(_input.shape, _input.dtype)
        _input = _input.astype(mindspore.float32)
        return mindspore.ops.OnesLike()(_input).astype(mindspore.bool_)
    return mindspore.ops.OnesLike()(_input)


def where(condition, x=None, y=None):
    if x is None or y is None:
        result = np.nonzero(condition.asnumpy())
        return tuple(mindspore.Tensor(index) for index in result)
    return mindspore.numpy.where(condition, x, y)


def tensordot(a, b, dims=2, out=None):
    result = mindspore.numpy.tensordot(a, b, axes=dims)
    return out_adaptor(result, out)


def meshgrid(*tensors):
    if len(tensors) == 1:
        return mindspore.numpy.meshgrid(*tuple(tensors[0]), indexing="ij")
    else:
        return mindspore.numpy.meshgrid(*tensors, indexing="ij")


def roll(input, shifts, dims=None):
    return mindspore.numpy.roll(input, shifts, dims)


def linspace(start, end, steps=100, out=None, dtype=None, layout=None, device=None, requires_grad=False):
    result = mindspore.numpy.linspace(start, end, steps, dtype=dtype)
    return out_adaptor(result, out)


def full(size, fill_value, *, out=None, dtype=None, layout=None, device=None, requires_grad=False):
    result = mindspore.numpy.full(size, fill_value, dtype=dtype)
    return out_adaptor(result, out)


def empty(*size, out=None, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False):
    if x2ms_context.get_is_during_transform():
        result = TensorNumpy.create_tensor_numpy(np.random.random(*size))
    else:
        result = rand(*size, dtype=dtype)
    return out_adaptor(result, out)


def empty_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=None):
    is_args_not_none = memory_format is not None or layout is not None or device is not None
    if is_args_not_none or requires_grad is not False:
        raise NotImplementedError("Parameters 'layout', 'device', 'requires_grad' and 'periodic' "
                                  "are not supported. Use default settings.")
    return mindspore.numpy.empty_like(input, dtype=dtype)


def erfc(input, *, out=None):
    if input.dtype not in (mindspore.float16, mindspore.float32):
        input = input.astype(mindspore.float32)
    erfc_func = mindspore.ops.Erfc()
    result = erfc_func(input)
    return out_adaptor(result, out)


def erfinv(input, *, out=None):
    result = mindspore.ops.Erfinv()(input)
    return out_adaptor(result, out)


def expm1(input, *, out=None):
    if input.dtype not in (mindspore.float16, mindspore.float32):
        input = input.astype(mindspore.float32)
    expm1_func = mindspore.ops.Expm1()
    result = expm1_func(input)
    return out_adaptor(result, out)


def multinomial(input, num_samples, replacement=False, *, generator=None, out=None):
    result = mindspore.ops.multinomial(inputs=input, num_sample=num_samples, replacement=replacement)
    return out_adaptor(result, out)


def gather(input, dim, index, out=None, sparse_grad=False):
    result = mindspore.ops.GatherD()(input, dim, index)
    return out_adaptor(result, out)


def sort(input, dim=-1, descending=False, out=None):
    origin_type = input.dtype
    converted_type = origin_type
    if origin_type not in (mindspore.float16, mindspore.float32):
        converted_type = mindspore.float32
    result = mindspore.ops.Sort(axis=dim, descending=descending)(input.astype(converted_type))
    return out_adaptor(result, out)


def x2ms_all(input, dim=None, keepdim=False, *, out=None):
    if input.dtype == mindspore.bool_:
        bool_input = input
    else:
        bool_input = input != 0
    if dim is None:
        return bool_input.all()
    else:
        result = mindspore.ops.ReduceAll(keepdim)(bool_input, dim)
        return out_adaptor(result, out)


def cumsum(input, dim, out=None, dtype=None):
    result = input.cumsum(axis=dim)
    return out_adaptor(result, out)


def full_like(input, fill_value, out=None, dtype=None, layout=None, device=None, requires_grad=False,
              memory_format=None):
    result = mindspore.numpy.full_like(input, fill_value, dtype=dtype)
    return out_adaptor(result, out)


def masked_select(input, mask, *, out=None):
    result = mindspore.ops.MaskedSelect()(input, mask)
    return out_adaptor(result, out)


def x2ms_mean(input, dim=None, keepdim=False, *, dtype=None, out=None):
    if dtype:
        input = input.astype(dtype)
    mean_ops = mindspore.ops.ReduceMean(keep_dims=keepdim)
    if dim is None:
        result = mean_ops(input)
    else:
        result = mean_ops(input, dim)
    return out_adaptor(result, out)


def mul(input, other, *, out=None):
    result = mindspore.ops.Mul()(input, other)
    return out_adaptor(result, out)


def topk(input, k, dim=None, largest=True, sorted=True, *, out=None):
    origin_type = input.dtype
    input = input.astype(mindspore.float32)
    if not largest:
        input *= -1
    if dim not in [-1, input.dim() - 1, None]:
        input = transpose(input, dim, input.dim() - 1)
        output = mindspore.ops.TopK(sorted=sorted)(input, k)
        values, indices = output[0].astype(origin_type), output[1]
        values, indices = transpose(values, dim, values.dim() - 1), transpose(indices, dim, indices.dim() - 1)
        if not largest:
            return values * -1, indices
        return values, indices
    if not largest:
        output = mindspore.ops.TopK(sorted=sorted)(input, k)
        values, indices = output[0].astype(origin_type), output[1]
        return values * -1, indices
    output = mindspore.ops.TopK(sorted=sorted)(input, k)
    output = (output[0].astype(origin_type), output[1])
    return output


def isfinite(input):
    return mindspore.numpy.isfinite(input)


def diag(input, diagonal=0, *, out=None):
    result = mindspore.numpy.diag(input, diagonal)
    return out_adaptor(result, out)


def zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=None):
    if x2ms_context.get_is_during_transform():
        return TensorNumpy.create_tensor_numpy(np.zeros_like(input, dtype))
    return tensor_zeros_like(input, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad,
                             memory_format=memory_format)


def tensor_zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=None):
    if input.size == 0:
        result = mindspore.ops.Zeros()(input.shape, input.dtype if dtype is None else dtype)
    else:
        result = mindspore.ops.zeros_like(input)
        if dtype:
            result = result.astype(dtype)
    if requires_grad:
        return result
    return mindspore.ops.stop_gradient(result)


def atan(input, out=None):
    if input.dtype == mindspore.float64:
        result = mindspore.ops.Atan()(input.astype(mindspore.float32)).astype(mindspore.float64)
    else:
        result = mindspore.ops.Atan()(input)
    return out_adaptor(result, out)


def unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
    """
    input type only support int64/int32 in mindspore.
    """
    if return_counts:
        raise NotImplementedError("MindSpore does not support param 'return_counts'.")
    if dim is not None:
        raise NotImplementedError("MindSpore does not support param 'dim'.")
    trans_flag = False
    origin_type = None
    if input.dtype not in (mindspore.int64, mindspore.int32):
        trans_flag = True
        origin_type = input.dtype
        input = input.astype(mindspore.int32)
    output = mindspore.numpy.unique(input, return_inverse)
    if trans_flag:
        output = output.astype(origin_type)
    sort_trans_flag = False
    sort_origin_type = None
    if sorted:
        if output.dtype not in (mindspore.float32, mindspore.float16):
            sort_trans_flag = True
            sort_origin_type = output.dtype
            output = output.astype(mindspore.float32)
        output = mindspore.ops.Sort()(output)[0]
        if sort_trans_flag:
            output = output.astype(sort_origin_type)
    return output


def triu(input, diagonal=0, out=None):
    result = mindspore.numpy.triu(input, diagonal)
    return out_adaptor(result, out)


def nonzero(input, out=None, as_tuple=False):
    if input.ndim == 0:
        if as_tuple:
            result = (mindspore.Tensor([0], dtype=mindspore.int64))
        else:
            result = mindspore.ops.zeros((1, 0), mindspore.int64)
    else:
        result = mindspore.ops.nonzero(input)
        if as_tuple:
            result = result.T
            result = tuple(result[index] for index in range(len(result)))
    return out_adaptor(result, out)


def log2(input, out=None):
    result = mindspore.numpy.log2(input)
    return out_adaptor(result, out)


def lt(input, other, *, out=None):
    result = mindspore.ops.less(input, other)
    return out_adaptor(result, out)


def ge(input, other, *, out=None):
    result = mindspore.ops.ge(input, other)
    return out_adaptor(result, out)


def ne(input, other, *, out=None):
    result = mindspore.ops.not_equal(input, other)
    return out_adaptor(result, out)


def le(input, other, *, out=None):
    result = mindspore.ops.le(input, other)
    return out_adaptor(result, out)


def norm(input, p='fro', dim=None, keepdim=False, out=None, dtype=None):
    if isinstance(p, int) or (isinstance(p, float) and int(p) == p):
        p = round(p)
    elif p == 'fro':
        p = 2
    else:
        raise TypeError(f"MindSpore does not support the {p} paradigm.")
    if dim is not None:
        norm_ops = mindspore.ops.LpNorm(axis=dim, p=p, keep_dims=keepdim)
    else:
        axis = tuple(range(input.ndim))
        norm_ops = mindspore.ops.LpNorm(axis=axis, p=p, keep_dims=keepdim)
    result = norm_ops(input)
    if dtype:
        result = result.astype(dtype)
    return out_adaptor(result, out)


def cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary'):
    """
    @parameter
    compute_mode is not implemented
    """
    if compute_mode != 'use_mm_for_euclid_dist_if_necessary':
        raise NotImplementedError('parameter compute_mode is currently not supported.')
    return mindspore.ops.Cdist(p=float(p))(x1, x2)


def erf(input, out=None):
    if input.dtype not in (mindspore.float16, mindspore.float32):
        input = input.astype(mindspore.float32)
    result = mindspore.ops.erf(input)
    return out_adaptor(result, out)


def softmax(input, dim=None, dtype=None):
    if dim is None:
        dim = -1
    result = mindspore.ops.Softmax(axis=dim)(input)
    if dtype:
        result = result.astype(dtype)

    return result


def eye(n, m=None, out=None, dtype=None, layout=None, device=None, requires_grad=False):
    """
    parameter 'layout' and 'device' is not supported in MindSpore
    """
    if dtype is None:
        dtype = mindspore.float32
    if m is None:
        m = n
    result = mindspore.ops.Eye()(n, m, dtype)

    if out is not None:
        out.assign_value(result)
        result = out

    if not requires_grad:
        return mindspore.ops.stop_gradient(result)

    return result


def _prod_reduce_all_dim(input, *, dtype=None):
    origin_type = input.dtype
    converted_type = origin_type
    if origin_type in (mindspore.int64,):
        converted_type = mindspore.float32

    result = mindspore.ops.ReduceProd()(input.astype(converted_type)).astype(origin_type)
    if dtype is not None:
        result = result.astype(dtype)
    return result


def _prod_keep_dim(input, dim, keepdim=False, *, dtype=None):
    result = mindspore.ops.ReduceProd(keep_dims=keepdim)(input, axis=dim)
    if dtype is not None:
        result = result.astype(dtype)
    return result


def prod(*args, **kwargs):
    if len(args) == 2 or 'dim' in kwargs:
        return _prod_keep_dim(*args, **kwargs)
    else:
        return _prod_reduce_all_dim(*args, **kwargs)


def reshape(input, shape):
    return mindspore.ops.Reshape()(input, tuple(shape))


def logical_and(input, other, *, out=None):
    _input = float_tensor_2_bool_tensor(input)
    _other = float_tensor_2_bool_tensor(other)
    result = mindspore.ops.LogicalAnd()(_input, _other)
    return out_adaptor(result, out)


def lerp(input, end, weight, *, out=None):
    result = mindspore.ops.Lerp()(input, end, weight)
    return out_adaptor(result, out)


def log1p(input, *, out=None):
    result = mindspore.numpy.log1p(input)
    return out_adaptor(result, out)


def logical_not(input, *, out=None):
    _input = float_tensor_2_bool_tensor(input)
    result = mindspore.numpy.logical_not(_input)
    return out_adaptor(result, out)


def logical_or(input, other, *, out=None):
    _input = float_tensor_2_bool_tensor(input)
    _other = float_tensor_2_bool_tensor(other)
    result = mindspore.ops.LogicalOr()(_input, _other)
    return out_adaptor(result, out)


def logical_xor(input, other, *, out=None):
    _input = float_tensor_2_bool_tensor(input)
    _other = float_tensor_2_bool_tensor(other)
    result = mindspore.numpy.logical_xor(_input, _other)
    return out_adaptor(result, out)


def var(input, dim, unbiased, keepdim=False, *, out=None):
    if unbiased is True:
        raise NotImplementedError('Bessel’s correction is not implemented.')
    _dim = tuple(dim) if isinstance(dim, list) else dim
    result = input.var(axis=_dim, keepdims=keepdim)
    return out_adaptor(result, out)


def unbind(input, dim=0):
    unbind_func = mindspore.ops.Unstack(axis=dim)
    return unbind_func(input)


def trunc(input, *, out=None):
    result = mindspore.numpy.trunc(input)
    return out_adaptor(result, out)


def true_divide(dividend, divisor, *, out=None):
    result = mindspore.numpy.true_divide(dividend, divisor)
    return out_adaptor(result, out)


def triu_indices(row, col, offset=0, *, dtype=None, device=None, layout=None):
    if device is not None or layout is not None:
        raise NotImplementedError("'device' and 'layout' are not supported in mindspore.")
    result = mindspore.ops.Stack()(mindspore.numpy.triu_indices(row, offset, col))
    if dtype is not None:
        result = result.astype(dtype)
    return result


def tril(input, diagonal=0, *, out=None):
    result = mindspore.numpy.tril(input, diagonal)
    return out_adaptor(result, out)


def trapz(y, x=None, *, dx=None, dim=-1):
    if x is None and dx is None:
        dx = 1.0
    return mindspore.numpy.trapz(y, x, dx, axis=dim)


def trapezoid(y, x=None, *, dx=None, dim=-1):
    if x is None and dx is None:
        dx = 1.0
    return mindspore.numpy.trapz(y, x, dx, axis=dim)


def trace(input):
    return input.trace()


def tan(input, out=None):
    result = mindspore.ops.Tan()(input)
    return out_adaptor(result, out)


def take(input, index):
    _input = mindspore.Tensor.flatten(input)
    if index.asnumpy().max() >= _input.shape[0]:
        raise IndexError(f'out of range: tried to access index {index.max().asnumpy()} on '
                         f'a tensor of {_input.shape[0]} elements.')
    result = mindspore.Tensor.take(_input, index)
    return result


def reminder(input, other, out=None):
    result = mindspore.numpy.remainder(input, other)
    return out_adaptor(result, out)


def result_type(tensor1, tensor2):
    return mindspore.numpy.result_type(tensor1, tensor2)


def real(input):
    result = mindspore.ops.Real()(input)
    return result


def reciprocal(input, *, out=None):
    result = mindspore.ops.Reciprocal()(input)
    return out_adaptor(result, out)


def neg(input, out=None):
    result = mindspore.ops.Neg()(input)
    return out_adaptor(result, out)


def ger(input, vec2, out=None):
    support_dtype = (mindspore.float16, mindspore.float32)
    trans_flag, input, origin_type = check_input_dtype(input, support_dtype)
    if input.dtype != vec2.dtype:
        vec2 = vec2.astype(input.dtype)
    result = mindspore.ops.Ger()(input, vec2)
    if trans_flag:
        result = result.astype(origin_type)
    return out_adaptor(result, out)


def addmm(input, mat1, mat2, beta=1, alpha=1, out=None):
    result = beta * input + alpha * mindspore.ops.matmul(mat1, mat2)
    return out_adaptor(result, out)


_equation_dict = {
    "multi_operate": {
        "ibnd,jbnd->ijbn": {
            "transpose": {
                "index0": (1, 2, 0, 3),
                "index1": (1, 2, 3, 0),
                "index_res": (2, 3, 0, 1)
            }
        },
        "ibnd,jbnd->bnij": {
            "transpose": {
                "index0": (1, 2, 0, 3),
                "index1": (1, 2, 3, 0)
            }
        },
        "ibnd,jnd->ijbn": {
            "broadcast": {
                "index1": 1
            },
            "transpose": {
                "index0": (1, 2, 0, 3),
                "index1": (0, 2, 3, 1),
                "index_res": (2, 3, 0, 1)
            }
        },
        "ijbn,jbnd->ibnd": {
            "transpose": {
                "index0": (2, 3, 0, 1),
                "index1": (1, 2, 0, 3),
                "index_res": (2, 0, 1, 3)
            }
        },
        "bhlt,bhtv->bhlv": {},
        "bhlk,bhtk->bhlt": {
            "transpose": {
                "index1": (0, 1, 3, 2)
            }
        },
        "ibh,hnd->ibnd": {
            "broadcast": {
                "index0": 1,
                "index1": 2
            },
            "transpose": {
                "index0": (0, 2, 1, 3),
                "index1": (2, 0, 1, 3),
                "index_res": (2, 1, 0, 3)
            }
        },
        "ibnd,snd->ibns": {
            "broadcast": {
                "index1": 0
            },
            "transpose": {
                "index0": (0, 2, 1, 3),
                "index1": (0, 2, 3, 1),
                "index_res": (0, 2, 1, 3)
            }
        },
        "ijbs,ibns->bnij": {
            "transpose": {
                "index0": (0, 2, 1, 3),
                "index1": (0, 1, 3, 2),
                "index_res": (1, 3, 0, 2)
            }
        },
        "bnij,jbnd->ibnd": {
            "transpose": {
                "index1": (1, 2, 0, 3),
                "index_res": (2, 0, 1, 3)
            }
        },
        "mbnd,mlb->lbnd": {
            "broadcast": {
                "index1": 2
            },
            "transpose": {
                "index0": (1, 2, 3, 0),
                "index1": (3, 0, 1, 2),
                "index_res": (3, 0, 1, 2)
            }
        },
        "lbnd,mlb->mbnd": {
            "broadcast": {
                "index1": 2
            },
            "transpose": {
                "index0": (1, 2, 3, 0),
                "index1": (3, 0, 2, 1),
                "index_res": (3, 0, 1, 2)
            }
        },
        "ibnd,hnd->ibh": {
            "reshape": {
                "index0": [[0, 1], -1],
                "index1": [[0], -1]
            },
            "broadcast": {
                "index1": 0
            },
            "transpose": {
                "index1": (0, 2, 1)
            }
        },
        "bhld,lrd->bhlr": {
            "broadcast": {
                "index1": 0
            },
            "transpose": {
                "index0": (0, 2, 1, 3),
                "index1": (0, 1, 3, 2),
                "index_res": (0, 2, 1, 3)
            }
        },
        "hbd,bmd->bhmd": {
            "broadcast": {
                "index0": 2,
                "index1": 1
            },
            "transpose": {
                "index0": (1, 2, 3, 0),
                "index1": (1, 0, 3, 2),
                "index_res": (0, 1, 3, 2)
            }
        },
        "bnik,bnjk->bnij": {
            "transpose": {
                "index1": (0, 1, 3, 2)
            }
        }
    },
    "single_operate": {
        "ijbn->bnij": {
            "transpose": {
                "index0": (2, 3, 0, 1)
            }
        },
        "bnij->ijbn": {
            "transpose": {
                "index0": (2, 3, 0, 1)
            }
        }
    }
}


def _handle_reshape(shape_operate, tensor0, tensor1):
    if shape_operate.get("index0") is not None:
        reshape0 = shape_operate.get("index0")
        if isinstance(reshape0[0], list):
            shape0 = [tensor0.shape[i] for i in reshape0[0]]
            reshape0 = shape0 + reshape0[1:]
        tensor0 = tensor0.reshape(*reshape0)
    if shape_operate.get("index1") is not None:
        reshape1 = shape_operate.get("index1")
        if isinstance(reshape1[0], list):
            shape1 = [tensor1.shape[i] for i in reshape1[0]]
            reshape1 = shape1 + reshape1[1:]
        tensor1 = tensor1.reshape(*reshape1)
    return tensor0, tensor1


def _handle_broadcast(broadcast_operate, tensor0, tensor1):
    if broadcast_operate.get("index0") is not None:
        tensor0 = tensor0.broadcast_to((tensor1.shape[broadcast_operate.get("index0")], *tensor0.shape))
    if broadcast_operate.get("index1") is not None:
        tensor1 = tensor1.broadcast_to((tensor0.shape[broadcast_operate.get("index1")], *tensor1.shape))
    return tensor0, tensor1


def _handle_transpose(transpose_operate, tensor0, tensor1):
    if transpose_operate.get("index0") is not None:
        tensor0 = tensor0.transpose(*transpose_operate.get("index0"))
    if transpose_operate.get("index1") is not None:
        tensor1 = tensor1.transpose(*transpose_operate.get("index1"))
    result = mindspore.ops.BatchMatMul()(tensor0, tensor1)
    if transpose_operate.get("index_res") is not None:
        return result.transpose(*transpose_operate.get("index_res"))
    return result


def einsum(equation, *operands):
    """You can refer to the example the statement below to implement other equations yourself:
        einsum("ibnd,hnd->ibh", *operands):
            tensor0 = operands[0]
            tensor1 = operands[1]
            tensor0 = tensor0.reshape(tensor0.shape[0], tensor0.shape[1], -1)
            tensor1 = tensor1.reshape(tensor1.shape[0], -1)
            tensor1 = tensor1.broadcast_to(tensor0.shape[0], *tensor1.shape)
            tensor1 = tensor1.transpose(0, 2, 1)
            result = mindspore.ops.BatchMatMul()(tensor0, tensor1)
    """
    if isinstance(operands[0], (tuple, list)):
        operands = tuple(operands[0])
    if mindspore.context.get_context('device_target') == 'GPU':
        return mindspore.ops.Einsum(equation)(operands)

    mul_operates = _equation_dict.get("multi_operate").get(equation)
    if mul_operates is not None:
        tensor0 = operands[0]
        tensor1 = operands[1]
        if mul_operates.get("reshape") is not None:
            tensor0, tensor1 = _handle_reshape(mul_operates.get("reshape"), tensor0, tensor1)
        if mul_operates.get("broadcast") is not None:
            tensor0, tensor1 = _handle_broadcast(mul_operates.get("broadcast"), tensor0, tensor1)
        if mul_operates.get("transpose") is not None:
            return _handle_transpose(mul_operates.get("transpose"), tensor0, tensor1)
        return mindspore.ops.BatchMatMul()(tensor0, tensor1)
    single_operates = _equation_dict.get("single_operate").get(equation)
    if single_operates is not None:
        tensor0 = operands[0]
        if single_operates.get("transpose") is not None:
            tensor0 = tensor0.transpose(*single_operates.get("transpose").get("index0"))
        return tensor0
    if equation == "i,d->id":
        tensor0 = operands[0].reshape(-1, 1)
        tensor1 = operands[1].reshape(1, -1)
        return mindspore.ops.MatMul()(tensor0, tensor1)
    if equation == "blh,bl->bh":
        shape1 = operands[1].shape
        tensor0 = operands[0].transpose(0, 2, 1)
        tensor1 = operands[1].reshape(shape1[0], shape1[1], -1)
        return mindspore.ops.BatchMatMul()(tensor0, tensor1).squeeze(axis=2)
    raise NotImplementedError(f"Does not support equation: {equation}, you can refer to the example in this "
                              f"function to implement it yourself")


class Finfo:
    def __init__(self, dtype):
        self.finfo_obj = np.finfo(mindspore.dtype.dtype_to_nptype(dtype))
        self.bits = self.finfo_obj.bits
        self.eps = self.finfo_obj.eps.item()
        self.min = self.finfo_obj.min.item()
        self.max = self.finfo_obj.max.item()
        self.resolution = self.finfo_obj.resolution.item()
        self.tiny = self.finfo_obj.tiny.item()


def finfo(dtype):
    return Finfo(dtype)


def get_rng_state():
    return mindspore.ops.Zeros()(0, mindspore.uint8)


def set_rng_state(new_state):
    pass


def randint(*args, **kwargs):
    has_low_high = len(args) >= 2 and isinstance(args[0], int) and isinstance(args[1], int)
    if has_low_high or (len(args) == 1 and "high" in kwargs):
        return _randint(*args, **kwargs)
    else:
        return _randint(kwargs.pop('low', 0), *args, **kwargs)


def _randint(low, high, size, generator=None, out=None, dtype=None, layout=None, device=None,
             requires_grad=False):
    minval = mindspore.Tensor(low, mindspore.int32)
    maxval = mindspore.Tensor(high, mindspore.int32)
    if not size:
        result = mindspore.ops.Zeros()(minval.shape, minval.dtype if dtype is None else dtype)
    else:
        result = mindspore.ops.UniformInt()(size, minval, maxval)
    if dtype is not None:
        result = result.astype(dtype)
    result = out_adaptor(result, out)
    if not requires_grad:
        return mindspore.ops.stop_gradient(result)
    return result


def randn_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=None):
    return randn(*input.shape, dtype=dtype, layout=layout, device=device)


def allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    return mindspore.numpy.isclose(input, other, rtol, atol, equal_nan).all()


def t(input):
    if len(input.shape) < 2:
        return input
    return mindspore.ops.transpose(input, (1, 0))


def rsqrt(input, *, out=None):
    result = mindspore.ops.Rsqrt()(input)
    return out_adaptor(result, out)


def x2ms_round(input, *, out=None):
    result = mindspore.ops.Rint()(input)
    return out_adaptor(result, out)


def acosh(input, *, out=None):
    result = mindspore.ops.Acosh()(input)
    return out_adaptor(result, out)


def addcmul(input, tensor1, tensor2, *, value=1, out=None):
    if not isinstance(value, mindspore.Tensor):
        value = mindspore.Tensor(value, mindspore.float32)
    result = mindspore.ops.Addcmul()(input, tensor1, tensor2, value)
    return out_adaptor(result, out)


def addcdiv(input, tensor1, tensor2, *, value=1, out=None):
    value = mindspore.Tensor(value, mindspore.float32)
    result = mindspore.ops.Addcdiv()(input, tensor1, tensor2, value)
    return out_adaptor(result, out)


def asinh(input, *, out=None):
    result = mindspore.ops.Asinh()(input)
    return out_adaptor(result, out)


def atanh(input, *, out=None):
    result = mindspore.ops.Atanh()(input)
    return out_adaptor(result, out)


def cummax(input, dim, *, out=None):
    result = mindspore.ops.cummax(input, dim)
    return out_adaptor(result, out)


def cummin(input, dim, *, out=None):
    result = mindspore.ops.cummin(input, dim)
    return out_adaptor(result, out)


def logsumexp(input, dim, keepdim=False, *, out=None):
    result = mindspore.ops.logsumexp(input, dim, keep_dims=keepdim)
    return out_adaptor(result, out)


def renorm(input, p, dim, maxnorm, *, out=None):
    result = mindspore.ops.renorm(input, p, dim, maxnorm)
    return out_adaptor(result, out)


def xlogy(input, other, *, out=None):
    xlogy_func = mindspore.ops.Xlogy()
    result = xlogy_func(input, other)
    if isinstance(input, mindspore.Tensor):
        result[input == 0] = 0
    if isinstance(other, mindspore.Tensor):
        result[mindspore.ops.isnan(other)] = mindspore.numpy.nan
    return out_adaptor(result, out)


def sign(input, *, out=None):
    sign_func = mindspore.ops.Sign()
    return out_adaptor(sign_func(input), out)


def sinh(input, *, out=None):
    sinh_func = mindspore.ops.Sinh()
    return out_adaptor(sinh_func(input), out)


def less(input, other, *, out=None):
    less_func = mindspore.ops.Less()
    return out_adaptor(less_func(input, other), out)


def narrow(input, dim, start, length):
    return mindspore.ops.narrow(input, dim, start, length)


def block_diag(*tensors):
    return scipy.linalg.block_diag(*tensors)


def cholesky_solve(input, input2, upper=False, *, out=None):
    result = scipy.linalg.cho_solve((input2, not upper), input)
    return out_adaptor(result, out)


def lu_solve(b, LU_data, LU_pivots, *, out=None):
    # The pivot indices tensor in mindspore is less one than torch.
    piv = LU_pivots - 1
    result = scipy.linalg.lu_solve((LU_data, piv), b)
    return out_adaptor(result, out)


def x2ms_any(input, dim=None, keepdim=False, *, out=None):
    if input.dtype == mindspore.bool_:
        bool_input = input
    else:
        bool_input = input != 0
    if dim is None:
        return bool_input.any()
    else:
        result = mindspore.ops.ReduceAny(keepdim)(bool_input, dim)
        return out_adaptor(result, out)


def greater(input, other, *, out=None):
    result = mindspore.ops.Greater()(input, other)
    return out_adaptor(result, out)


def greater_equal(input, other, *, out=None):
    result = mindspore.ops.GreaterEqual()(input, other)
    return out_adaptor(result, out)


def less_equal(input, other, *, out=None):
    result = mindspore.ops.LessEqual()(input, other)
    return out_adaptor(result, out)


def not_equal(input, other, *, out=None):
    result = mindspore.ops.NotEqual()(input, other)
    return out_adaptor(result, out)


def log10(input, *, out=None):
    result = mindspore.ops.log10(input)
    return out_adaptor(result, out)


def count_nonzero(input, dim=None):
    if dim is None:
        dim = ()
    return mindspore.ops.count_nonzero(input, dim)


def signbit(input, *, out=None):
    result = mindspore.numpy.signbit(input)
    return out_adaptor(result, out)


def isposinf(input, *, out=None):
    result = mindspore.numpy.isposinf(input)
    return out_adaptor(result, out)


def isin(elements, test_elements, *, assume_unique=False, invert=False):
    if assume_unique:
        raise NotImplementedError('Numpy argument assume_unique is not supported since the implementation'
                                  ' does not rely on the uniqueness of the input arrays.')
    return mindspore.numpy.isin(elements, test_elements, invert=invert)


def isneginf(input, *, out=None):
    result = mindspore.numpy.isneginf(input)
    return out_adaptor(result, out)


def copysign(input, other, *, out=None):
    result = mindspore.numpy.copysign(input, other)
    return out_adaptor(result, out)


def deg2rad(input, *, out=None):
    result = mindspore.numpy.deg2rad(input)
    return out_adaptor(result, out)


def diff(input, n=1, dim=-1, prepend=None, append=None):
    return mindspore.numpy.diff(input, n=n, axis=dim, prepend=prepend, append=append)


def gcd(input, other, *, out=None):
    result = mindspore.numpy.gcd(input, other)
    return out_adaptor(result, out)


def heaviside(input, values, *, out=None):
    result = mindspore.numpy.heaviside(input, values)
    return out_adaptor(result, out)


class Size:
    def __new__(cls, shape):
        return tuple(shape)


if LooseVersion(mindspore.__version__) < LooseVersion('1.8.0'):
    from ..ms_1_7_1.torch_base_api import scatter, index_select, vstack, amax, amin
