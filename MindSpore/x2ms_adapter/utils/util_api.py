#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
import logging
import re

import mindspore
import numpy as np


class Generator:
    def __init__(self, device='cpu'):
        self.seed = 0

    def manual_seed(self, seed):
        self.seed = seed


def float_tensor_2_bool_tensor(data):
    if isinstance(data, mindspore.Tensor) and data.dtype != mindspore.bool_:
        _data = data != 0
    else:
        _data = data
    return _data


def out_adaptor(result, out):
    if out is not None:
        if isinstance(out, tuple) and isinstance(result, tuple):
            for out_item, result_item in zip(out, result):
                out_item.assign_value(result_item)
            return out
        else:
            return out.assign_value(result)
    return result


def inplace_adaptor(original_tensor, result, inplace):
    if inplace:
        return original_tensor.assign_value(result)
    return result


_NP_TO_MS_TYPE_DICT = {
    np.float16: mindspore.float16,
    np.float32: mindspore.float32,
    np.float64: mindspore.float64,
    np.int: mindspore.int32,
    np.long: mindspore.int64,
    np.bool_: mindspore.bool_,
    np.int8: mindspore.int8,
    np.int16: mindspore.int16,
    np.int32: mindspore.int32,
    np.int64: mindspore.int64,
    np.uint8: mindspore.uint8,
    np.uint16: mindspore.uint16,
    np.uint32: mindspore.uint32,
    np.uint64: mindspore.uint64,
}


def np_to_tensor(array: np.ndarray):
    if array.size == 0 and array.ndim != 1:
        return mindspore.ops.Zeros()(array.shape, _NP_TO_MS_TYPE_DICT.get(array.dtype, mindspore.float32))
    # if array.dtype is not in supported type list (such as "|S82"), convert its dtype to str.
    if type(array.dtype).type == np.bytes_:
        array = mindspore.dataset.text.to_str(array)
    return mindspore.Tensor(array)


def check_input_dtype(input_tensor, support_dtype):
    trans_flag = False
    origin_type = None
    if input_tensor.dtype not in support_dtype:
        trans_flag = True
        origin_type = input_tensor.dtype
        input_tensor = input_tensor.astype(mindspore.float32)
    return trans_flag, input_tensor, origin_type


class Logger(logging.Logger):

    def __init__(self, log_level=logging.INFO,
                 log_format='%(asctime)s [%(levelname)s] %(message)s',
                 datefmt='%Y-%m-%d %H:%M:%S'):
        super().__init__('', log_level)
        self._formatter = logging.Formatter(log_format, datefmt)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self._formatter)
        self.addHandler(console_handler)
        self.pattern_nblank = re.compile('[\r\n\f\v\t\b\u007F]')
        self.pattern_blank = re.compile(' {2,}')

    def error(self, msg, *args, **kwargs):
        super(Logger, self).error(self._format_msg(msg), *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        super(Logger, self).warning(self._format_msg(msg), *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        super(Logger, self).info(self._format_msg(msg), *args, **kwargs)

    def _format_msg(self, msg):
        msg = self.pattern_nblank.sub('', str(msg))
        msg = self.pattern_blank.sub(' ', str(msg))
        return msg


logger = Logger()
