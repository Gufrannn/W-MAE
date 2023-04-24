#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

import mindspore


class Function:
    @property
    def saved_tensors(self):
        return []

    @staticmethod
    def construct(ctx, *args, **kwargs):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        pass

    @classmethod
    def apply(cls, *args, **kwargs):
        return cls.construct(Function(), *args, **kwargs)

    def mark_dirty(self, *args):
        pass

    def mark_non_differentiable(self, *args):
        pass

    def save_for_backward(self, *tensors):
        pass

    def set_materialize_grads(self, value):
        pass

    def mark_shared_storage(self, *pairs):
        pass


class Variable(mindspore.Tensor):
    def __init__(self, data, requires_grad=False, volatile=True):
        """
        Args:
            volatile: deprecated parameter. Used in PyTorch 0.3.x.
                Same function as torch.no_grad() in later torch version > 0.4
        """
        super().__init__(data)


def vjp(func, inputs, v=None, create_graph=False, strict=False):
    if isinstance(inputs, tuple):
        return mindspore.nn.Vjp(func)(*inputs, v)
    else:
        return mindspore.nn.Vjp(func)(inputs, v)


def jvp(func, inputs, v=None, create_graph=False, strict=False):
    return mindspore.ops.jvp(func, inputs, v)
