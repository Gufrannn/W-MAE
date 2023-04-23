#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
import mindspore.nn
from ..torch_api.nn_api.nn_functional import adaptive_avg_pool1d


class AdaptiveAvgPool1d(mindspore.nn.Cell):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def construct(self, input):
        return adaptive_avg_pool1d(input, self.output_size)


class Mish(mindspore.nn.Cell):
    def __init__(self, inplace=False):
        super().__init__()
        self.mish = mindspore.ops.Mish()

    def construct(self, input):
        return self.mish(input)
