#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
import types

import mindspore
from .context import x2ms_context


class WithLossCell(mindspore.nn.Cell):
    _instance_dict = {}

    def __new__(cls, train_obj=None, construct=None, key='times_0'):
        if cls._instance_dict.get(key, None) is None:
            cls._instance_dict[key] = super().__new__(cls)
        return cls._instance_dict.get(key)

    def __init__(self, train_obj=None, construct=None, key='times_0'):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._input = None
        self._output = None
        self.train_obj = train_obj
        self.amp_model = x2ms_context.amp_model
        self._construct_func = types.MethodType(construct, self)

    @property
    def output(self):
        return self._output

    def set_output(self, *output):
        self._output = output

    def construct(self, *args):
        return self._construct_func(*args)
