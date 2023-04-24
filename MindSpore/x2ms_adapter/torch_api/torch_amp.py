#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
from ..core.context import x2ms_context
from ..utils.util_api import logger


def amp_initialize(models, optimizers=None, enabled=True, opt_level="O1", cast_model_type=None,
                   patch_torch_functions=None, keep_batchnorm_fp32=None, master_weights=None, loss_scale=None,
                   cast_model_outputs=None, num_losses=1, verbosity=1, min_loss_scale=None, max_loss_scale=2. ** 24):
    if opt_level == "O1":
        logger.warning("MindSpore does not support O1, use O2 instead.")
        x2ms_context.amp_opt_level = "O2"
    else:
        x2ms_context.amp_opt_level = opt_level
    x2ms_context.loss_scale = loss_scale
    if optimizers is None:
        return models
    return models, optimizers


def amp_state_dict(destination=None):
    return {}


def amp_master_params(optimizer):
    return optimizer.trainable_params(True)


class GradScaler(object):
    def __init__(self, init_scale=2. ** 16, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True):
        self._enabled = enabled
        if enabled:
            self._init_scale = init_scale
            self._scale = None
            self._growth_factor = growth_factor
            self._backoff_factor = backoff_factor
            self._growth_interval = growth_interval
            self._init_growth_tracker = 0
            self._growth_tracker = None
            x2ms_context.amp_opt_level = "O2"
            x2ms_context.loss_scale = init_scale

    def scale(self, outputs):
        if not self._enabled:
            return outputs

        class _ScaleResultStub:
            def backward(self, *args, **kwargs):
                pass

        return _ScaleResultStub()

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer, *args, **kwargs):
        pass

    def update(self, new_scale=None):
        pass

    def get_scale(self):
        if self._enabled:
            return self._init_scale if self._scale is None else 1.0
        else:
            return 1.0

    def get_growth_factor(self):
        return self._growth_factor

    def set_growth_factor(self, new_factor):
        self._growth_factor = new_factor

    def get_backoff_factor(self):
        return self._backoff_factor

    def set_backoff_factor(self, new_factor):
        self._backoff_factor = new_factor

    def get_growth_interval(self):
        return self._growth_interval

    def set_growth_interval(self, new_interval):
        self._growth_interval = new_interval

    def is_enabled(self):
        return self._enabled

    def state_dict(self):
        if self._enabled:
            return {
                "scale": self.get_scale(),
                "growth_factor": self._growth_factor,
                "backoff_factor": self._backoff_factor,
                "growth_interval": self._growth_interval,
                "_growth_tracker": self._get_growth_tracker(),
            }
        else:
            return {}

    def _get_growth_tracker(self):
        if self._enabled:
            return self._init_growth_tracker if self._growth_tracker is None else self._growth_tracker.item()
        else:
            return 0
