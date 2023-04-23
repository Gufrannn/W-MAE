#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
import itertools
import numbers

import mindspore

from ..utils.util_api import out_adaptor


def scatter(input, dim, index, src, reduce=None):
    def get_value(src_value, tensor_index):
        if isinstance(src_value, mindspore.Tensor):
            return src_value[tensor_index]
        if isinstance(src_value, numbers.Number):
            return src_value
        return src_value

    shape_tuple = index.shape
    numpy_index = index.asnumpy()
    list_range = []
    for shape in shape_tuple:
        list_range.append(range(shape))
    for tensor_idx_tuple in itertools.product(*list_range):
        idx_list = list(tensor_idx_tuple)
        idx_list[dim] = numpy_index[tuple(tensor_idx_tuple)].item()
        if reduce == 'multiply':
            input[tuple(idx_list)] *= get_value(src, tuple(tensor_idx_tuple))
        elif reduce == 'add':
            input[tuple(idx_list)] += get_value(src, tuple(tensor_idx_tuple))
        else:
            input[tuple(idx_list)] = get_value(src, tuple(tensor_idx_tuple))

    return input


def index_select(input, dim, index, *, out=None):
    if index.dim() == 0:
        index = index.reshape(1)
    result = mindspore.ops.gather(input, index, dim)
    return out_adaptor(result, out)


def vstack(tensors, *, out=None):
    tensors = list(tensors)
    if tensors and len(tensors[0].shape) == 1:
        for i, tensor in enumerate(tensors):
            tensors[i] = mindspore.ops.expand_dims(tensor, 0)
    tensors = tuple(tensors)
    return out_adaptor(mindspore.ops.Concat(axis=0)(tensors), out)


def amax(input, dim, keepdim=False, *, out=None):
    result = mindspore.numpy.amax(input, dim, keepdims=keepdim)
    return out_adaptor(result, out)


def amin(input, dim, keepdim=False, *, out=None):
    result = mindspore.numpy.amin(input, dim, keepdims=keepdim)
    return out_adaptor(result, out)