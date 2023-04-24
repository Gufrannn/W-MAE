#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import mindspore


def is_instance(obj, target_type):
    return mindspore.ops.IsInstance()(obj, target_type)


def trace(func, *args, **kwargs):
    return func


def annotate(the_type, the_value):
    return the_value
