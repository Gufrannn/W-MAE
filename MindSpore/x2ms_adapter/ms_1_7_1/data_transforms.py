#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
from distutils.version import LooseVersion
from numbers import Number

import numpy as np

import mindspore
import mindspore.dataset.transforms
import mindspore.dataset.vision

from ..core.context import x2ms_context

_ms_np_type_map = {
    mindspore.float32: np.float32,
    mindspore.float64: np.float64,
    mindspore.float16: np.float16,
    mindspore.int64: np.int64,
    mindspore.int32: np.int32,
    mindspore.int8: np.int8,
    mindspore.uint8: np.uint8,
    mindspore.int16: np.int16,
}

if LooseVersion(mindspore.__version__) < LooseVersion('1.8.0'):
    ms_data_transforms_api_map = {
        'Resize': mindspore.dataset.vision.py_transforms.Resize,
        'Compose': mindspore.dataset.py_transforms.Compose,
        'RandomResizedCrop': mindspore.dataset.vision.py_transforms.RandomResizedCrop,
        'RandomHorizontalFlip': mindspore.dataset.vision.py_transforms.RandomHorizontalFlip,
        'RandomVerticalFlip': mindspore.dataset.vision.py_transforms.RandomVerticalFlip,
        'ToTensor': mindspore.dataset.vision.py_transforms.ToTensor,
        'RandomColorAdjust': mindspore.dataset.vision.py_transforms.RandomColorAdjust,
        'RandomErasing': mindspore.dataset.vision.py_transforms.RandomErasing,
        'TenCrop': mindspore.dataset.vision.py_transforms.TenCrop,
        'RandomAffine': mindspore.dataset.vision.c_transforms.RandomAffine,
        'RandomApply': mindspore.dataset.transforms.c_transforms.RandomApply,
        'RandomPosterize': mindspore.dataset.vision.c_transforms.RandomPosterize,
        'RandomGrayscale': mindspore.dataset.vision.py_transforms.RandomGrayscale,
        'RandomPerspective': mindspore.dataset.vision.py_transforms.RandomPerspective,
        'Pad': mindspore.dataset.vision.c_transforms.Pad,
        'LinearTransformation': mindspore.dataset.vision.py_transforms.LinearTransformation,
        'RandomSolarize': mindspore.dataset.vision.c_transforms.RandomSolarize,
        'RandomChoice': mindspore.dataset.transforms.py_transforms.RandomChoice,
        'AutoAugment': mindspore.dataset.vision.c_transforms.AutoAugment,
        'RandomAdjustSharpness': mindspore.dataset.vision.c_transforms.RandomAdjustSharpness,
        'RandomAutoContrast': mindspore.dataset.vision.c_transforms.RandomAutoContrast,
        'RandomEqualize': mindspore.dataset.vision.c_transforms.RandomEqualize,
        'RandomInvert': mindspore.dataset.vision.c_transforms.RandomInvert,
        'CenterCrop': mindspore.dataset.vision.py_transforms.CenterCrop,
        'RandomCrop': mindspore.dataset.vision.py_transforms.RandomCrop,
        'RandomRotation': mindspore.dataset.vision.py_transforms.RandomRotation,
        'RandomOrder': mindspore.dataset.transforms.py_transforms.RandomOrder,
        'FiveCrop': mindspore.dataset.vision.py_transforms.FiveCrop,
        'GaussianBlur': mindspore.dataset.vision.c_transforms.GaussianBlur,
        'Grayscale': mindspore.dataset.vision.py_transforms.Grayscale
    }
else:
    ms_data_transforms_api_map = {
        'Resize': mindspore.dataset.vision.Resize,
        'Compose': mindspore.dataset.transforms.Compose,
        'RandomResizedCrop': mindspore.dataset.vision.RandomResizedCrop,
        'RandomHorizontalFlip': mindspore.dataset.vision.RandomHorizontalFlip,
        'RandomVerticalFlip': mindspore.dataset.vision.RandomVerticalFlip,
        'ToTensor': mindspore.dataset.vision.ToTensor,
        'RandomColorAdjust': mindspore.dataset.vision.RandomColorAdjust,
        'RandomErasing': mindspore.dataset.vision.RandomErasing,
        'TenCrop': mindspore.dataset.vision.TenCrop,
        'RandomAffine': mindspore.dataset.vision.RandomAffine,
        'RandomApply': mindspore.dataset.transforms.RandomApply,
        'RandomPosterize': mindspore.dataset.vision.RandomPosterize,
        'RandomGrayscale': mindspore.dataset.vision.RandomGrayscale,
        'RandomPerspective': mindspore.dataset.vision.RandomPerspective,
        'Pad': mindspore.dataset.vision.Pad,
        'LinearTransformation': mindspore.dataset.vision.LinearTransformation,
        'RandomSolarize': mindspore.dataset.vision.RandomSolarize,
        'RandomChoice': mindspore.dataset.transforms.RandomChoice,
        'AutoAugment': mindspore.dataset.vision.AutoAugment,
        'RandomAdjustSharpness': mindspore.dataset.vision.RandomAdjustSharpness,
        'RandomAutoContrast': mindspore.dataset.vision.RandomAutoContrast,
        'RandomEqualize': mindspore.dataset.vision.RandomEqualize,
        'RandomInvert': mindspore.dataset.vision.RandomInvert,
        'CenterCrop': mindspore.dataset.vision.CenterCrop,
        'RandomCrop': mindspore.dataset.vision.RandomCrop,
        'RandomRotation': mindspore.dataset.vision.RandomRotation,
        'RandomOrder': mindspore.dataset.transforms.RandomOrder,
        'FiveCrop': mindspore.dataset.vision.FiveCrop,
        'GaussianBlur': mindspore.dataset.vision.GaussianBlur,
        'Grayscale': mindspore.dataset.vision.Grayscale
    }


class Normalize(mindspore.dataset.vision.py_transforms.Normalize):
    def __init__(self, mean, std, inplace=False):
        if isinstance(mean, mindspore.Tensor):
            mean = mean.asnumpy().tolist()
        if isinstance(std, mindspore.Tensor):
            std = std.asnumpy().tolist()
        if isinstance(mean, Number):
            mean = [mean]
        if isinstance(std, Number):
            std = [std]
        super().__init__(mean, std)

    def __call__(self, img):
        if x2ms_context.get_is_during_transform():
            return super().__call__(img)
        if isinstance(img, mindspore.Tensor):
            img = img.asnumpy()
        return mindspore.Tensor(super().__call__(img))


class ConvertImageDtype(mindspore.dataset.vision.py_transforms.ToType):
    def __init__(self, dtype):
        super().__init__(output_type=_ms_np_type_map.get(dtype))


def autocontrast(img):
    return mindspore.dataset.vision.py_transforms.AutoContrast()(img)


def equalize(img):
    return mindspore.dataset.vision.py_transforms.Equalize()(img)


def invert(img):
    return mindspore.dataset.vision.py_transforms.Invert()(img)
