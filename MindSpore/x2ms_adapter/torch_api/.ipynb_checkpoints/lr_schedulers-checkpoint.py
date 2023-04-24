#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
import math

import mindspore.nn
import numpy as np

from ..utils.util_api import logger


class _LRScheduler:
    def __init__(self, optimizer, last_epoch=-1):
        super(_LRScheduler, self).__init__()
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch
        self.step()

    def step(self, step=None):
        self.last_epoch += 1
        for params, lr in zip(self.optimizer.param_groups, self.get_lr()):
            params['lr'] = mindspore.Tensor(lr, mindspore.float32)
        return list(param['lr'] for param in self.optimizer.param_groups)

    def get_lr(self):
        raise NotImplementedError

    def state_dict(self):
        return {}


class StepLR(_LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, verbose=False):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch != 0 and self.last_epoch % self.step_size == 0:
            return list((param['lr'] * self.gamma) for param in self.optimizer.param_groups)
        return list(param['lr'] for param in self.optimizer.param_groups)


class LambdaLR(_LRScheduler):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1, verbose=False):
        self.lr_lambda = lr_lambda
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return list((self.lr_lambda(self.last_epoch) * lr) for lr in self.base_lrs)


class CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False):
        """
        Args:
            verbose currently unsupported
        """

        min_lr = float(eta_min)
        self.lr_group = [mindspore.nn.CosineDecayLR(min_lr, float(param_group['lr']), int(T_max))
                         for param_group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)

    def construct(self, global_step):
        return self.get_lr()

    def get_lr(self):
        return [one_lr_schedule.construct(mindspore.Tensor(self.last_epoch)).asnumpy().item()
                for one_lr_schedule in self.lr_group]


class MultiStepLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1, verbose=False):
        self.milestones = milestones
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.milestones:
            return list((param['lr'] * self.gamma) for param in self.optimizer.param_groups)
        return list(param['lr'] for param in self.optimizer.param_groups)


class ReduceLROnPlateau:
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8, verbose=False):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        if isinstance(min_lr, (tuple, list)):
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)
        self.eps = eps
        self.verbose = verbose

        self.num_bad_epochs = None
        self.last_epoch = 0
        if mode == 'min':
            self.mode_worse = np.inf
        else:
            self.mode_worse = -np.inf
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics):
        self.last_epoch += 1
        current_metrics = float(metrics)
        if self._is_better(current_metrics, self.best):
            self.num_bad_epochs = 0
            self.best = current_metrics
        else:
            self.num_bad_epochs += 1

        if self.cooldown > 0:
            self.num_bad_epochs = 0
            self.cooldown -= 1

        if self.num_bad_epochs > self.patience:
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            for i, param_group in enumerate(self.optimizer.param_groups):
                old_lr = float(param_group['lr'])
                new_lr = max(old_lr * self.factor, self.min_lrs[i])
                if old_lr - new_lr <= self.eps:
                    continue
                param_group['lr'] = new_lr
                if self.verbose:
                    logger.info(f'Epoch {self.last_epoch}: reducing learning rate of group {i} to {new_lr:.4e}.')

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def _is_better(self, current, best):
        if self.mode == 'min':
            if self.threshold_mode == 'rel':
                return current < best * (1. - self.threshold)
            else:
                return current < best - self.threshold
        else:
            if self.threshold_mode == 'rel':
                return current > best * (self.threshold + 1.)
            else:
                return current > best + self.threshold


class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, gamma, last_epoch=-1, verbose=False):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return list((param['lr'] * self.gamma) for param in self.optimizer.param_groups)


class CyclicLR(_LRScheduler):
    def __init__(self, optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9,
                 last_epoch=-1, verbose=False):
        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')
        if step_size_up <= 0 or (step_size_down is not None and step_size_down <= 0):
            raise ValueError('step_size_down and step_size_up must be positive number')

        self.mode = mode
        self.gamma = gamma
        base_lrs = self._format_param('base_lr', optimizer, base_lr)

        self.max_lrs = self._format_param('max_lr', optimizer, max_lr)

        float_step_size_up = float(step_size_up)
        float_step_size_down = float(step_size_down) if step_size_down is not None else float_step_size_up
        self.total_size = float_step_size_up + float_step_size_down
        self.step_ratio = float_step_size_up / self.total_size
        self.cycle_momentum = cycle_momentum

        if scale_fn is None:
            _scale_dict = {
                'triangular': {
                    'scale_fn': lambda x: 1.,
                    'scale_mode': 'cycle'
                },
                'triangular2': {
                    'scale_fn': lambda x: 1 / (2. ** (x - 1)),
                    'scale_mode': 'cycle'
                },
                'exp_range': {
                    'scale_fn': lambda x: self.gamma ** x,
                    'scale_mode': 'iterations'
                },
            }
            self.scale_fn = _scale_dict.get(self.mode).get('scale_fn')
            self.scale_mode = _scale_dict.get(self.mode).get('scale_mode')
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        if cycle_momentum:
            self._init_momentum(optimizer, base_momentum, max_momentum, last_epoch)

        super().__init__(optimizer, last_epoch)
        self.base_lrs = base_lrs

    def get_lr(self):
        cycle = math.floor(1 + self.last_epoch / self.total_size)
        cycle_offset = 1. + self.last_epoch / self.total_size - cycle
        _offset = 1 if cycle_offset > self.step_ratio else 0
        scale_factor = (cycle_offset - _offset) / (self.step_ratio - _offset)

        lrs = []
        _scale = cycle if self.scale_mode == 'cycle' else self.last_epoch
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            base_height = (max_lr - base_lr) * scale_factor
            lr = base_lr + base_height * self.scale_fn(_scale)
            lrs.append(lr)

        if self.cycle_momentum:
            momentums = []
            for base_momentum, max_momentum in zip(self.base_momentums, self.max_momentums):
                base_height = (max_momentum - base_momentum) * scale_factor
                momentum = max_momentum - base_height * self.scale_fn(_scale)
                momentums.append(momentum)
            for param_group, momentum in zip(self.optimizer.param_groups, momentums):
                param_group['momentum'] = momentum

        return lrs

    def _format_param(self, name, optimizer, param):
        parma_groups_length = len(optimizer.param_groups)
        if isinstance(param, (list, tuple)):
            param_length = len(param)
            if param_length != parma_groups_length:
                raise ValueError("expected {} values for {}, got {}".format(
                    parma_groups_length, name, param_length))
            return param
        else:
            return [param] * parma_groups_length

    def _init_momentum(self, optimizer, base_momentum, max_momentum, last_epoch):
        if 'momentum' not in optimizer.defaults:
            raise ValueError('optimizer must support momentum with `cycle_momentum` option enabled')

        _base_momentums = self._format_param('base_momentum', optimizer, base_momentum)
        if last_epoch == -1:
            for momentum, group in zip(_base_momentums, optimizer.param_groups):
                group['momentum'] = momentum
        self.max_momentums = self._format_param('max_momentum', optimizer, max_momentum)
        self.base_momentums = [group['momentum'] for group in optimizer.param_groups]
