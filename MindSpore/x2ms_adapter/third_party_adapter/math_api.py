#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
import mindspore


def sqrt(x):
    return mindspore.numpy.sqrt(x * mindspore.Tensor(1.0))
