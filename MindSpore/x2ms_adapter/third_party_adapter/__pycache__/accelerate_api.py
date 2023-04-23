#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
import os
import logging
from enum import Enum

import mindspore
import mindspore.context as context

from ..torch_api.nn_api import nn as x2ms_nn
from ..torch_api import torch_base_api, save_load, distributed
from ..utils.util_api import logger


class Accelerator:
    MODEL_NAME = "mindspore_model"

    def __init__(self, *args, **kwargs):
        self.is_main_process = True
        self.num_processes = 1
        self.is_local_main_process = True
        self.state = ""
        self.model = None
        self.optimizer = None
        self.distributed_type = None
        self.use_fp16 = False

    @property
    def device(self):
        return torch_base_api.Device()

    @staticmethod
    def log(values, step):
        logger.info(str(values))

    @staticmethod
    def unwrap_model(model):
        if isinstance(model, (x2ms_nn.DataParallel, x2ms_nn.DistributedDataParallel)):
            return model.module
        return model

    @staticmethod
    def save(obj, f):
        save_load.save(obj, f)

    @staticmethod
    def main_process_first():
        class MainProcessStub:
            def __enter__(self):
                pass

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        return MainProcessStub()

    @staticmethod
    def gather(tensor):
        if context.get_auto_parallel_context("parallel_mode") == context.ParallelMode.DATA_PARALLEL:
            if isinstance(tensor, mindspore.Tensor):
                return distributed.gather(Accelerator._int64_to_int32(tensor))
            elif isinstance(tensor, tuple):
                return tuple(distributed.gather(Accelerator._int64_to_int32(_tensor)) for _tensor in tensor)
            elif isinstance(tensor, list):
                return list(distributed.gather(Accelerator._int64_to_int32(_tensor)) for _tensor in tensor)
            elif isinstance(tensor, dict):
                return {key: distributed.gather(Accelerator._int64_to_int32(value)) for key, value in tensor.items()}
        return tensor

    @staticmethod
    def accumulate(model):
        class AccumulateStub:
            def __enter__(self):
                pass

            def __exit__(self, *excinfo):
                pass

        return AccumulateStub()

    @staticmethod
    def _int64_to_int32(tensor):
        if tensor.dtype == mindspore.int64:
            return tensor.astype(mindspore.int32)
        return tensor

    def pad_across_processes(self, tensor, dim=0, pad_index=0, pad_first=False):
        old_size = tensor.shape
        if dim >= len(old_size):
            return tensor

        size = mindspore.Tensor(old_size)[None]
        sizes = self.gather(size).asnumpy().tolist()

        max_size = max(_size[dim] for _size in sizes)
        if max_size == old_size[dim]:
            return tensor

        new_size = list(old_size)
        new_size[dim] = max_size
        new_tensor = mindspore.ops.zeros(tuple(new_size), tensor.dtype) + pad_index
        indices = [slice(None)] * len(new_size)
        indices[dim] = slice(max_size - old_size[dim], max_size) if pad_first else slice(0, old_size[dim])
        new_tensor[tuple(indices)] = tensor
        return new_tensor

    def prepare(self, *args):
        for arg in args:
            if isinstance(arg, mindspore.nn.Optimizer):
                self.optimizer = arg
            elif isinstance(arg, mindspore.nn.Cell):
                self.model = arg
        return args

    def print(self, *args, **kwargs):
        """
        Use in replacement of `print()` to only print once per server.
        """
        if self.is_local_main_process:
            logger.info(*args, **kwargs)

    def wait_for_everyone(self):
        pass

    def init_trackers(self, *args, **kwargs):
        pass

    def save_state(self, output_dir: str):
        if self.model is not None:
            save_load.save(self.model, os.path.join(output_dir, self.MODEL_NAME))

    def load_state(self, model_path):
        if self.model is not None:
            save_load.load_state_dict(obj=self.model,
                                      state_dict=save_load.load(os.path.join(model_path, self.MODEL_NAME)))


class Logger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg)


def get_logger(name):
    return Logger(name)


class DistributedType(Enum):
    TPU = 0
    GPU = 1
