#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import mindspore


def get_cell_params(cell, recurse=True):
    return iter(cell.trainable_params(recurse) + cell.untrainable_params(recurse))


def count_parameters(m, x, y):
    total_params = 0
    for item in get_cell_params(m):
        total_params += item.size
    total_params = mindspore.Tensor([total_params], dtype=mindspore.float32)
    m.total_params[0] = total_params
