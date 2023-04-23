#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
from distutils.version import LooseVersion
from enum import Enum

import mindspore

if LooseVersion(mindspore.__version__) >= LooseVersion('1.8.0'):
    from mindspore.dataset.audio import ResampleMethod
    import mindspore.dataset.audio as audio
else:
    import mindspore.dataset.audio.transforms as audio


    class ResampleMethod(str, Enum):
        SINC_INTERPOLATION: str = "sinc_interpolation"
        KAISER_WINDOW: str = "kaiser_window"
