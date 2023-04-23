#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import mindspore
import mindspore.numpy
import numpy as np


def mean(a, *args, **kwargs):
    if isinstance(a, (list, tuple)) and len(a) > 0 and isinstance(a[0], mindspore.Tensor):
        return np_tensor_mean(a, *args, **kwargs)
    return np.mean(a, *args, **kwargs)


def np_tensor_mean(tensor_list, axis=None, dtype=None, out=None, keepdims=None, *, where=None):
    if tensor_list[0].dim() != 0:
        raise NotImplementedError("Only supports tensor with 0-dim")
    concat_tensor = mindspore.numpy.stack(tensor_list)
    return concat_tensor.astype(mindspore.float32).mean().asnumpy().astype(dtype if dtype else float).take(0)


def concatenate(arrays, *args, **kwargs):
    if isinstance(arrays, (list, tuple)) and len(arrays) > 0 and isinstance(arrays[0], mindspore.Tensor):
        return np_tensor_concatenate(arrays, *args, **kwargs)
    return np.concatenate(arrays, *args, **kwargs)


def np_tensor_concatenate(arrays, axis=None, out=None, *, dtype=None, casting=None):
    return np.concatenate(list(tensor.asnumpy() for tensor in arrays), axis)


class TensorNumpy(np.ndarray):
    @staticmethod
    def create_tensor_numpy(data):
        tensor_numpy = TensorNumpy(shape=data.shape, dtype=data.dtype)
        np.copyto(tensor_numpy, data)
        return tensor_numpy

