#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
import math

import mindspore
from mindspore.common.initializer import initializer, Constant, Normal, One, Uniform, Zero, \
    HeNormal, HeUniform, XavierUniform, Orthogonal, _calculate_fan_in_and_fan_out
from ...utils.util_api import logger


def _assign_value(tensor, init):
    value = initializer(init, tensor.shape, mindspore.float32)
    value.init_data()
    if isinstance(tensor, mindspore.Parameter):
        tensor.set_data(value)
    else:
        tensor.assign_value(value)
    return tensor


def constant_(tensor, val):
    if tensor.parent_tensor_ is not None:
        value = initializer(Constant(val), tensor.shape, mindspore.float32)
        value.init_data()
        tensor.parent_tensor_[tensor.index_of_parent_] = value
        return tensor.parent_tensor_[tensor.index_of_parent_]
    return _assign_value(tensor, Constant(val))


def normal_(tensor, mean=0.0, std=1.0):
    # see also: https://en.wikipedia.org/wiki/Normal_distribution
    return _assign_value(tensor, Normal(sigma=std, mean=mean))


def ones_(tensor):
    return _assign_value(tensor, One())


def uniform_(tensor, a=0.0, b=1.0):
    if not math.isclose(a + b, 0.0, rel_tol=1e-5):
        logger.warning(f'Uniform initializer in MindSpore does not support bound ({a}, {b}), '
                       'will use default argument.')
        return _assign_value(tensor, Uniform())
    else:
        return _assign_value(tensor, Uniform(scale=b))


def zeros_(tensor):
    return _assign_value(tensor, Zero())


def kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    return _assign_value(tensor, HeNormal(negative_slope=a, mode=mode, nonlinearity=nonlinearity))


def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    return _assign_value(tensor, HeUniform(negative_slope=a, mode=mode, nonlinearity=nonlinearity))


def xavier_uniform_(tensor, gain=1.0):
    return _assign_value(tensor, XavierUniform(gain=gain))


def xavier_normal_(tensor, gain=1.0):
    n_in, n_out = _calculate_fan_in_and_fan_out(tensor.shape)
    if n_in + n_out <= 0:
        raise ValueError(f'Invalid shape {tensor.shape} for xavier_normal.')
    else:
        std = gain * math.sqrt(2.0 / (n_in + n_out))
    return _assign_value(tensor, Normal(sigma=std))


xavier_normal = xavier_normal_


def orthogonal_(tensor, gain=1.0):
    return _assign_value(tensor, Orthogonal(gain=gain))
