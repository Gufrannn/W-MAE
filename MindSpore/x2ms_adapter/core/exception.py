#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

class TrainBreakException(Exception):
    pass


class TrainContinueException(Exception):
    pass


class TrainReturnException(Exception):
    pass
