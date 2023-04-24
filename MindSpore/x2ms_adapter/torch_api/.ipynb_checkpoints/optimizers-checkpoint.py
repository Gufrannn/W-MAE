#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
from distutils.version import LooseVersion
from typing import Iterator
from types import GeneratorType
from collections import namedtuple

import mindspore.nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from ..core.context import x2ms_context

_X2MS_ADAM_W_OPT = mindspore.ops.composite.MultitypeFuncGraph("_X2MS_ADAM_W_OPT")


@_X2MS_ADAM_W_OPT.register("Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor",
                           "Tensor", "Tensor", "Tensor", "Bool", "Bool")
def _adam_w_op(beta1, beta2, beta1_power, beta2_power, eps, learning_rate, weight_decay, param, moment_m, moment_v,
               gradient, decay_flag, optim_filter):
    if optim_filter:
        param_fp32 = param.astype(mindspore.float32)
        m_fp32 = moment_m.astype(mindspore.float32)
        v_fp32 = moment_v.astype(mindspore.float32)
        gradient_fp32 = gradient.astype(mindspore.float32)
        _tuple_to_array = mindspore.ops.operations.TupleToArray()

        next_m = beta1 * m_fp32 + (_tuple_to_array((1.0,)).astype(mindspore.float32) - beta1) * gradient_fp32
        next_v = beta2 * v_fp32 + (_tuple_to_array((1.0,)).astype(mindspore.float32) - beta2) * \
                 mindspore.ops.pow(gradient_fp32, 2)
        next_m_div = next_m / (_tuple_to_array((1.0,)).astype(mindspore.float32) - beta1_power)
        next_v_div = next_v / (_tuple_to_array((1.0,)).astype(mindspore.float32) - beta2_power)
        update = next_m_div / (eps + mindspore.ops.sqrt(next_v_div))
        if decay_flag:
            update = weight_decay * param_fp32 + update
        update_with_lr = learning_rate * update
        next_param = param_fp32 - update_with_lr.reshape(param_fp32.shape)
        _assign = mindspore.ops.Assign()
        _depend = mindspore.ops.Depend()
        next_param = _depend(next_param, _assign(param, next_param.astype(param.dtype)))
        next_param = _depend(next_param, _assign(moment_m, next_m.astype(moment_m.dtype)))
        next_param = _depend(next_param, _assign(moment_v, next_v.astype(moment_v.dtype)))
        return next_param.astype(param.dtype)
    return gradient.astype(param.dtype)


OptimizerInfo = namedtuple('OptimizerInfo', ['instance'])


class OptimAdaptorMixIn:
    def zero_grad(self, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass


class Adam(mindspore.nn.Adam, OptimAdaptorMixIn):
    def __init__(self, params, **kwargs):
        new_params = params_dict_to_list(params)
        mindspore.nn.Adam.__init__(self, new_params, **kwargs)
        _record_args(self, kwargs, params)
        self.x2ms_param_groups = create_param_groups_modifiers(self)

    def construct(self, gradients):
        if x2ms_context.clip_grad_norm is not None:
            gradients = mindspore.ops.composite.clip_by_global_norm(gradients, x2ms_context.clip_grad_norm)
        return super().construct(gradients)


class SGD(mindspore.nn.SGD, OptimAdaptorMixIn):
    def __init__(self, params, **kwargs):
        new_params = params_dict_to_list(params)
        mindspore.nn.SGD.__init__(self, new_params, **kwargs)
        _record_args(self, kwargs, params)
        self.x2ms_param_groups = create_param_groups_modifiers(self)

    def construct(self, gradients):
        if x2ms_context.clip_grad_norm is not None:
            gradients = mindspore.ops.composite.clip_by_global_norm(gradients, x2ms_context.clip_grad_norm)
        return super().construct(gradients)


class RMSprop(mindspore.nn.RMSProp, OptimAdaptorMixIn):
    def __init__(self, params, **kwargs):
        new_params = params_dict_to_list(params)
        mindspore.nn.RMSProp.__init__(self, new_params, **kwargs)
        _record_args(self, kwargs, params)
        self.x2ms_param_groups = create_param_groups_modifiers(self)

    def construct(self, gradients):
        if x2ms_context.clip_grad_norm is not None:
            gradients = mindspore.ops.composite.clip_by_global_norm(gradients, x2ms_context.clip_grad_norm)
        return super().construct(gradients)


class Rprop(mindspore.nn.Rprop, OptimAdaptorMixIn):
    def __init__(self, params, **kwargs):
        new_params = params_dict_to_list(params)
        mindspore.nn.Rprop.__init__(self, new_params, **kwargs)
        _record_args(self, kwargs, params)
        self.x2ms_param_groups = create_param_groups_modifiers(self)

    def construct(self, gradients):
        if x2ms_context.clip_grad_norm is not None:
            gradients = mindspore.ops.composite.clip_by_global_norm(gradients, x2ms_context.clip_grad_norm)
        return super().construct(gradients)


class Adagrad(mindspore.nn.Adagrad, OptimAdaptorMixIn):
    def __init__(self, params, **kwargs):
        new_params = params_dict_to_list(params)
        mindspore.nn.Adagrad.__init__(self, new_params, **kwargs)
        _record_args(self, kwargs, params)
        self.x2ms_param_groups = create_param_groups_modifiers(self)

    def construct(self, gradients):
        if x2ms_context.clip_grad_norm is not None:
            gradients = mindspore.ops.composite.clip_by_global_norm(gradients, x2ms_context.clip_grad_norm)
        return super().construct(gradients)


class ASGD(mindspore.nn.ASGD, OptimAdaptorMixIn):
    def __init__(self, params, **kwargs):
        new_params = params_dict_to_list(params)
        mindspore.nn.ASGD.__init__(self, new_params, **kwargs)
        _record_args(self, kwargs, params)
        self.x2ms_param_groups = create_param_groups_modifiers(self)

    def construct(self, gradients):
        if x2ms_context.clip_grad_norm is not None:
            gradients = mindspore.ops.composite.clip_by_global_norm(gradients, x2ms_context.clip_grad_norm)
        return super().construct(gradients)


def _get_value(param, key):
    value = param.get(key)
    if key == 'params':
        return list(parameter for parameter in value if parameter.requires_grad)
    return value


def _parse_params(params):
    parse_keys = ['params', 'lr', 'weight_decay', 'order_params', 'grad_centralization']
    new_params = []
    for param in params:
        new_param = {}
        for key in param.keys():
            if isinstance(param[key], Iterator):
                param[key] = list(param[key])
            if key in parse_keys:
                new_param[key] = _get_value(param, key)
        new_params.append(new_param)
    return new_params


def params_dict_to_list(params):
    if isinstance(params[0], dict):
        new_params = _parse_params(params)
        return new_params
    return list(parameter for parameter in params if parameter.requires_grad)


class OptimRegister:
    def __init__(self):
        self._register_info = []
        self._lr_scheduler = None

    @staticmethod
    def _params_to_list(params):
        if isinstance(params, (GeneratorType, Iterator)):
            params = list(params)
        return params

    def adam(self, params, lr=0.001, betas=(0.9, 0.999),
             eps=1e-8, weight_decay=0, amsgrad=False):
        params = self._params_to_list(params)
        kwargs = {
            "learning_rate": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "eps": eps,
            "weight_decay": weight_decay,
        }
        optimizer_instance = Adam(params, **kwargs)
        self._register_info.append(OptimizerInfo(optimizer_instance))
        return optimizer_instance

    def sgd(self, params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        params = self._params_to_list(params)
        kwargs = {
            "learning_rate": lr,
            "momentum": momentum,
            "dampening": dampening,
            "nesterov": nesterov,
            "weight_decay": weight_decay,
        }
        optimizer_instance = SGD(params, **kwargs)
        self._register_info.append(OptimizerInfo(optimizer_instance))
        return optimizer_instance

    def rmsprop(self, params, lr=0.01, alpha=0.99, eps=1e-10, weight_decay=0, momentum=0.0, centered=False):
        params = self._params_to_list(params)
        kwargs = {
            "learning_rate": lr,
            "momentum": momentum,
            "epsilon": eps,
            "decay": alpha,
            "centered": centered,
            "weight_decay": weight_decay,
        }
        optimizer_instance = RMSprop(params, **kwargs)
        self._register_info.append(OptimizerInfo(optimizer_instance))
        return optimizer_instance

    def rprop(self, params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50)):
        params = self._params_to_list(params)
        if isinstance(step_sizes[0], int) or isinstance(step_sizes[1], int):
            step_sizes = (float(step_sizes[0]), float(step_sizes[1]))
        kwargs = {
            "learning_rate": lr,
            "etas": etas,
            "step_sizes": step_sizes,
        }
        optimizer_instance = Rprop(params, **kwargs)
        self._register_info.append(OptimizerInfo(optimizer_instance))
        return optimizer_instance

    def adagrad(self, params, lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10):
        params = self._params_to_list(params)
        kwargs = {
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "accum": float(initial_accumulator_value) + eps
        }
        optimizer_instance = Adagrad(params, **kwargs)
        self._register_info.append(OptimizerInfo(optimizer_instance))
        return optimizer_instance

    def adamw(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False):
        params = self._params_to_list(params)
        kwargs = {
            "learning_rate": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "eps": eps,
            "weight_decay": weight_decay
        }
        optimizer_instance = AdamW(params, **kwargs)
        self._register_info.append(OptimizerInfo(optimizer_instance))
        return optimizer_instance

    def asgd(self, params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0):
        params = self._params_to_list(params)
        kwargs = {
            "learning_rate": lr,
            "lambd": lambd,
            "alpha": alpha,
            "t0": t0,
            "weight_decay": weight_decay
        }
        optimizer_instance = ASGD(params, **kwargs)
        self._register_info.append(OptimizerInfo(optimizer_instance))
        return optimizer_instance

    def adadelta(self, params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0, foreach=None, *, maximize=False):
        params = self._params_to_list(params)
        kwargs = {
            "learning_rate": lr,
            "rho": rho,
            "epsilon": eps,
            "weight_decay": weight_decay
        }
        optimizer_instance = Adadelta(params, **kwargs)
        self._register_info.append(OptimizerInfo(optimizer_instance))
        return optimizer_instance

    def adamax(self, params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, foreach=None, *, maximize=False):
        params = self._params_to_list(params)
        kwargs = {
            "learning_rate": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "eps": eps,
            "weight_decay": weight_decay
        }
        optimizer_instance = Adamax(params, **kwargs)
        self._register_info.append(OptimizerInfo(optimizer_instance))
        return optimizer_instance

    def get_instance(self):
        if len(self._register_info) == 0:
            return None
        elif len(self._register_info) > 1:
            return ConcatOptimizer(list(optimizer_info.instance for optimizer_info in self._register_info))
        return self._register_info[-1].instance


def _record_args(optimizer, kwargs, params):
    if hasattr(optimizer, 'x2ms_input_kwargs'):
        return
    optimizer.x2ms_input_kwargs = kwargs
    if isinstance(params[0], dict):
        optimizer.x2ms_param_list = _list(params)
    else:
        optimizer.x2ms_param_list = [{'params': params}]
    if 'learning_rate' in kwargs:
        optimizer.initial_lr = kwargs['learning_rate']


class ConcatOptimizer(mindspore.nn.Optimizer):
    def __init__(self, optimizer_list):
        parameters = ()
        for optimizer in optimizer_list:
            parameters += optimizer.parameters
        super().__init__(learning_rate=0.1, parameters=parameters, weight_decay=0.0, loss_scale=1.0)
        self.optimizer_list = optimizer_list

    def construct(self, gradients):
        if x2ms_context.clip_grad_norm is not None:
            gradients = mindspore.ops.composite.clip_by_global_norm(gradients, x2ms_context.clip_grad_norm)
        success = ()
        start = 0
        for optimizer in self.optimizer_list:
            success += optimizer(gradients[start:(start + len(optimizer.parameters))])
            start = start + len(optimizer.parameters)
        return success


def create_param_groups_modifiers(optim):
    param_list = []
    for index, params in enumerate(optim.x2ms_param_list):
        param_list.append(OptimizerParamGroupsModifier(optim, params, index))
    return param_list


class OptimizerParamGroupsModifier:
    def __init__(self, optimizer, param, index=0):
        self.index = index
        self._optimizer = optimizer
        self.param_dict = dict(param)
        if 'lr' not in self.param_dict:
            self.param_dict['lr'] = optimizer.initial_lr
        if hasattr(optimizer, 'momentum'):
            if isinstance(optimizer.momentum, mindspore.Tensor):
                self.param_dict['momentum'] = float(optimizer.momentum.asnumpy())
            else:
                self.param_dict['momentum'] = optimizer.momentum

    def __setitem__(self, key, value):
        if key == 'lr':
            self.set_lr(value)
        elif key == 'momentum':
            self.set_momentum(value)
        else:
            self.param_dict[key] = value

    def __getitem__(self, key):
        if key == 'momentum' and hasattr(self._optimizer, 'momentum'):
            _momentum = self._optimizer.momentum
            return float(_momentum.asnumpy()) if isinstance(_momentum, mindspore.Tensor) else _momentum
        else:
            return self.param_dict.get(key)

    def __iter__(self):
        return iter(self.param_dict)

    def setdefault(self, key, default=None):
        self.param_dict.setdefault(key, default)

    def set_lr(self, value):
        if self._optimizer.is_group_lr:
            self._optimizer.learning_rate[self.index].set_data(Tensor(value, mstype.float32))
        else:
            self._optimizer.learning_rate.set_data(Tensor(value, mstype.float32))
        self.param_dict['lr'] = value

    def set_momentum(self, value):
        if hasattr(self._optimizer, 'momentum'):
            if isinstance(self._optimizer.momentum, mindspore.Tensor):
                self._optimizer.momentum.assign_value(mindspore.Tensor(value, mindspore.float32))
            else:
                self._optimizer.momentum = value
            self.param_dict['momentum'] = value


class _RequiredMindsporeCellParameter(object):
    def __repr__(self):
        return "<required parameter>"


@property
def get_param_groups(self):
    if hasattr(self, 'x2ms_param_groups'):
        return self.x2ms_param_groups
    return []


def _list(param):
    return param if isinstance(param, list) else [param]


def add_param_group(self, param_group):
    if 'lr' not in param_group:
        param_group['lr'] = self.initial_lr
    self.x2ms_param_list += _list(param_group)
    self.__init__(self.x2ms_param_list, **self.x2ms_input_kwargs)


mindspore.nn.Optimizer.param_groups = get_param_groups
mindspore.nn.Optimizer.add_param_group = add_param_group
optim_register = OptimRegister()
required = _RequiredMindsporeCellParameter()

if LooseVersion(mindspore.__version__) < LooseVersion('1.8.0'):
    class AdamW(mindspore.nn.AdamWeightDecay, OptimAdaptorMixIn):
        def __init__(self, params, **kwargs):
            new_params = params_dict_to_list(params)
            mindspore.nn.AdamWeightDecay.__init__(self, new_params, **kwargs)
            _record_args(self, kwargs, params)
            self.x2ms_param_groups = create_param_groups_modifiers(self)

        def construct(self, gradients):
            if x2ms_context.clip_grad_norm is not None:
                gradients = mindspore.ops.composite.clip_by_global_norm(gradients, x2ms_context.clip_grad_norm)
            return super().construct(gradients)
else:
    class Adamax(mindspore.nn.AdaMax, OptimAdaptorMixIn):
        def __init__(self, params, **kwargs):
            new_params = params_dict_to_list(params)
            mindspore.nn.AdaMax.__init__(self, new_params, **kwargs)
            _record_args(self, kwargs, params)
            self.x2ms_param_groups = create_param_groups_modifiers(self)

        def construct(self, gradients):
            if x2ms_context.clip_grad_norm is not None:
                gradients = mindspore.ops.composite.clip_by_global_norm(gradients, x2ms_context.clip_grad_norm)
            return super().construct(gradients)


    class Adadelta(mindspore.nn.Adadelta, OptimAdaptorMixIn):
        def __init__(self, params, **kwargs):
            new_params = params_dict_to_list(params)
            mindspore.nn.Adadelta.__init__(self, new_params, **kwargs)
            _record_args(self, kwargs, params)
            self.x2ms_param_groups = create_param_groups_modifiers(self)

        def construct(self, gradients):
            if x2ms_context.clip_grad_norm is not None:
                gradients = mindspore.ops.composite.clip_by_global_norm(gradients, x2ms_context.clip_grad_norm)
            return super().construct(gradients)


    class AdamW(mindspore.nn.AdamWeightDecay, OptimAdaptorMixIn):
        def __init__(self, params, **kwargs):
            new_params = params_dict_to_list(params)
            mindspore.nn.AdamWeightDecay.__init__(self, new_params, **kwargs)
            _record_args(self, kwargs, params)
            self.x2ms_param_groups = create_param_groups_modifiers(self)
            self.beta1_power = mindspore.Parameter(mindspore.ops.ones((1,), mindspore.float32))
            self.beta2_power = mindspore.Parameter(mindspore.ops.ones((1,), mindspore.float32))
            self._partial = mindspore.ops.operations.Partial()

        def construct(self, gradients):
            if x2ms_context.clip_grad_norm is not None:
                gradients = mindspore.ops.composite.clip_by_global_norm(gradients, x2ms_context.clip_grad_norm)
            gradients = self.flatten_gradients(gradients)
            weight_decay = self.get_weight_decay()
            learning_rate = self.get_lr()
            new_beta1_power = self.beta1_power * self.beta1
            self.beta1_power = new_beta1_power
            new_beta2_power = self.beta2_power * self.beta2
            self.beta2_power = new_beta2_power
            if self.is_group:
                if self.is_group_lr:
                    result = self.hyper_map(self._partial(_X2MS_ADAM_W_OPT, self.beta1, self.beta2, new_beta1_power,
                                                          new_beta2_power, self.eps), learning_rate, weight_decay,
                                            self._parameters, self.moments1,
                                            self.moments2, gradients, self.decay_flags, self.optim_filter)
                else:
                    result = self.hyper_map(self._partial(_X2MS_ADAM_W_OPT, self.beta1, self.beta2, new_beta1_power,
                                                          new_beta2_power, self.eps, learning_rate),
                                            weight_decay, self._parameters, self.moments1, self.moments2,
                                            gradients, self.decay_flags, self.optim_filter)
            else:
                result = self.hyper_map(self._partial(_X2MS_ADAM_W_OPT, self.beta1, self.beta2, new_beta1_power,
                                                      new_beta2_power, self.eps, learning_rate, weight_decay),
                                        self._parameters, self.moments1, self.moments2,
                                        gradients, self.decay_flags, self.optim_filter)
            if self.use_parallel:
                self.broadcast_params(result)

            return result
