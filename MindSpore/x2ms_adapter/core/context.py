#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
import threading
import mindspore.nn as nn
try:
    from mindspore import ms_class
except ImportError:
    def null_decorator(func):
        return func
    ms_class = null_decorator


@ms_class
class Context:
    TRANSFORM_LOCK = threading.Lock()

    def __init__(self):
        self.amp_opt_level = None
        self.clip_grad_norm = None
        self.amp_model = nn.CellList(auto_prefix=False)
        self.loss_scale = None
        self.transformer_thread_set = set()
        self.is_context_init = False

    def thread_start_transform(self):
        with self.TRANSFORM_LOCK:
            self.transformer_thread_set.add(threading.current_thread())

    def thread_end_transform(self):
        with self.TRANSFORM_LOCK:
            self.transformer_thread_set.remove(threading.current_thread())

    def get_is_during_transform(self):
        with self.TRANSFORM_LOCK:
            return threading.current_thread() in self.transformer_thread_set


x2ms_context = Context()
