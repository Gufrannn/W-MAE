#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

import io
import stat
import os
import time
from numbers import Number
from pathlib import Path
import mindspore
from mindspore import context

from ..utils.util_api import logger
from ..utils.adapter_check import external_input_check, external_output_check

_TMP_MODEL_NAME = "tmp_checkpoint.ckpt"


def save(obj, f, pickle_module=None, pickle_protocol=None, _use_new_zipfile_serialization=False):
    """
    Function replace torch.save
    """
    parallel_mode = context.get_auto_parallel_context("parallel_mode")
    try:
        local_rank = mindspore.communication.get_local_rank()
    except RuntimeError:
        local_rank = -1
    if parallel_mode == context.ParallelMode.DATA_PARALLEL and local_rank not in [-1, 0]:
        return

    file_name = f
    if isinstance(file_name, Path):
        file_name = str(file_name)
    if isinstance(file_name, str):
        if not file_name.endswith('.ckpt'):
            file_name += '.ckpt'
    if isinstance(f, io.BytesIO):
        file_name = str(int(time.time())) + _TMP_MODEL_NAME
    if isinstance(f, io.BufferedWriter):
        file_name = file_name.name
        if not file_name.endswith('.ckpt'):
            file_name = file_name + '.ckpt'
    external_output_check(os.path.dirname(os.path.realpath(file_name)))

    if isinstance(obj, mindspore.nn.Cell):
        mindspore.save_checkpoint(obj, file_name)
    if isinstance(obj, dict):
        _SaveLoadDict.save(obj, file_name)

    if isinstance(f, io.BytesIO):
        flags = os.O_RDONLY
        modes = stat.S_IRUSR | stat.S_IWUSR
        with os.fdopen(os.open(file_name, flags, modes), 'rb') as ckpt_file:
            f.write(ckpt_file.read())
        os.remove(file_name)


def load(f, map_location=None, pickle_module=None, **pickle_load_args):
    """
    Loads checkpoint info from a specified file.
    """
    file_name = f
    if isinstance(f, str):
        if not f.endswith('.ckpt'):
            file_name = f + '.ckpt'
    elif isinstance(f, io.BufferedReader):
        file_name = file_name.name
        if not file_name.endswith('.ckpt'):
            file_name = file_name + '.ckpt'
    elif isinstance(f, io.BytesIO):
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        modes = stat.S_IWUSR | stat.S_IRUSR
        file_name = str(int(time.time())) + _TMP_MODEL_NAME
        with os.fdopen(os.open(file_name, flags, modes), 'wb+') as ckpt_file:
            ckpt_file.write(f.getvalue())
    else:
        raise NotImplementedError('ERROR: input file object is not supported')
    external_input_check(file_name)
    load_dict = _SaveLoadDict.load(file_name)

    if isinstance(f, io.BytesIO):
        os.remove(file_name)

    return load_dict


def load_state_dict(obj, state_dict, strict=True):
    """
    Stub function for torch.nn.module.load_state_dict
    Loads parameters into network.
    The parameter strict will be set False, to avoid defects caused by deleting functions such as nn.DataParallel.
    Returns:
       List, parameters not loaded in the network.
    """
    param_not_load = []
    if isinstance(obj, mindspore.nn.Cell):
        param_not_load = mindspore.load_param_into_net(obj, state_dict, strict_load=False)

    return param_not_load


class _SaveLoadDict(object):
    SUPPORT_MEMBER_TYPE = [Number, str, mindspore.Tensor, mindspore.Parameter, dict, bool]
    _VALUE_SUFFIX = "_x2ms_value"
    _STR_SUFFIX = ".x2ms_str"
    _SAVE_HEAD = "x2ms_dict"

    @staticmethod
    def save(save_obj, file_name):
        if _SaveLoadDict._is_save_parameter_dict(save_obj):
            param_list = list({'name': k, 'data': v} for k, v in save_obj.items())
            mindspore.save_checkpoint(param_list, file_name)
        else:
            _SaveLoadDict._save_dict(save_obj, file_name)

    @staticmethod
    def load(file_name):
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"{file_name} does not exist.")
        load_dict = mindspore.load_checkpoint(file_name)
        if _SaveLoadDict._is_load_x2ms_dict(load_dict):
            load_dict = _SaveLoadDict._load_dict(load_dict)
        return load_dict

    @staticmethod
    def _is_save_parameter_dict(save_obj):
        return all(isinstance(member, mindspore.Parameter) for member in save_obj.values())

    @staticmethod
    def _is_load_x2ms_dict(load_obj):
        return _SaveLoadDict._SAVE_HEAD in load_obj.keys()

    @staticmethod
    def _save_dict(save_obj, file_name):
        param_list = []
        param_list.append({"name": _SaveLoadDict._SAVE_HEAD, "data": mindspore.Tensor(0)})
        for key, value in save_obj.items():
            for support_type in _SaveLoadDict.SUPPORT_MEMBER_TYPE:
                if isinstance(value, support_type):
                    getattr(_SaveLoadDict, f"_save_dict_{support_type.__name__.lower()}")(param_list, key, value)
                    break
        mindspore.save_checkpoint(param_list, file_name)

    @staticmethod
    def _save_dict_dict(param_list, save_name, save_obj):
        if not _SaveLoadDict._is_save_parameter_dict(save_obj):
            logger.warning(f"Does not support to saving type of {save_name}.")
            return
        param_list.append({"name": f"{save_name}.dict", "data": mindspore.Tensor(len(save_obj))})
        param_list.extend(list({'name': f"{save_name}.{k}", 'data': v} for k, v in save_obj.items()))

    @staticmethod
    def _save_dict_number(param_list, save_name, save_obj: Number):
        _SaveLoadDict._save_single_value(param_list, save_name, "number", mindspore.Tensor(save_obj))

    @staticmethod
    def _save_dict_str(param_list, save_name, save_obj):
        param_list.append({"name": f"{save_name}.str", "data": mindspore.Tensor(1)})
        param_list.append({"name": f"{save_obj}{_SaveLoadDict._STR_SUFFIX}", "data": mindspore.Tensor(0)})

    @staticmethod
    def _save_dict_tensor(param_list, save_name, save_obj: mindspore.Tensor):
        _SaveLoadDict._save_single_value(param_list, save_name, "tensor", save_obj)

    @staticmethod
    def _save_dict_parameter(param_list, save_name, save_obj: mindspore.Parameter):
        _SaveLoadDict._save_single_value(param_list, save_name, "parameter", save_obj)

    @staticmethod
    def _save_dict_bool(param_list, save_name, save_obj):
        _SaveLoadDict._save_single_value(param_list, save_name, "bool", mindspore.Tensor(save_obj))

    @staticmethod
    def _save_single_value(param_list, save_name, save_type, save_obj):
        param_list.append({"name": f"{save_name}.{save_type}", "data": mindspore.Tensor(1)})
        param_list.append({"name": f"{save_name}{_SaveLoadDict._VALUE_SUFFIX}", "data": save_obj})

    @staticmethod
    def _load_dict(load_dict):
        param_dict = {}
        param_iter = iter(load_dict)
        next(param_iter)
        try:
            while True:
                key = next(param_iter)
                length = load_dict.get(key).asnumpy().item()
                data_type = key.split(".")[-1]
                if getattr(_SaveLoadDict, f"_load_dict_{data_type}"):
                    value = getattr(_SaveLoadDict, f"_load_dict_{data_type}")(load_dict, param_iter, length, key)
                    param_dict[".".join(key.split(".")[:-1])] = value
        except StopIteration:
            return param_dict

    @staticmethod
    def _load_dict_number(load_dict, iterator, length, save_name):
        return load_dict.get(next(iterator)).asnumpy().item()

    @staticmethod
    def _load_dict_str(load_dict, iterator, length, save_name):
        result_str = next(iterator)
        return ".".join(result_str.split(".")[:-1])

    @staticmethod
    def _load_dict_bool(load_dict, iterator, length, save_name):
        return load_dict.get(next(iterator)).asnumpy().item()

    @staticmethod
    def _load_dict_tensor(load_dict, iterator, length, save_name):
        return load_dict.get(next(iterator))

    @staticmethod
    def _load_dict_parameter(load_dict, iterator, length, save_name):
        return load_dict.get(next(iterator))

    @staticmethod
    def _load_dict_dict(load_dict, iterator, length, save_name):
        result_dict = {}
        real_save_name = ".".join(save_name.split(".")[:-1])
        for _ in range(length):
            key = next(iterator)
            real_name = key[len(real_save_name) + 1:]
            parameter = load_dict.get(key)
            parameter.name = real_name
            result_dict[real_name] = parameter
        return result_dict
