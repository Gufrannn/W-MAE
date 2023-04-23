#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
import mindspore

from ..core.decorator import x2ms_func_decorator


@x2ms_func_decorator(mindspore.Tensor)
def masked_fill(obj, mask, value):
    broadcast_to = mindspore.ops.BroadcastTo(obj.shape)
    mask = mask.astype(mindspore.int32)
    reverse_mask = (mask == 0).astype(mindspore.int32)
    mask = broadcast_to(mask)
    reverse_mask = broadcast_to(reverse_mask)
    return obj * reverse_mask + mask * mindspore.Tensor(value, dtype=mindspore.float32)


@x2ms_func_decorator(mindspore.Tensor)
def new_ones(obj, size, dtype=None, device=None, requires_grad=False):
    new_tensor = mindspore.ops.ones(tuple(size), dtype if dtype else obj.dtype)
    if not requires_grad:
        return mindspore.ops.stop_gradient(new_tensor)
    return new_tensor


@x2ms_func_decorator(mindspore.Tensor)
def norm(obj, p='fro', dim=None, keepdim=False, dtype=None):
    if p not in ['fro', 2]:
        raise TypeError("MindSpore currently only supports L2 paradigm.")
    if dim is not None:
        norm_ops = mindspore.nn.Norm(axis=dim, keep_dims=keepdim)
    else:
        norm_ops = mindspore.nn.Norm(keep_dims=keepdim)
    return norm_ops(obj)
