#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
from collections import OrderedDict
import itertools
import inspect
from types import MethodType
import mindspore
import mindspore.nn

from ...core.decorator import x2ms_func_decorator
from ..tensor_api import copy_


@x2ms_func_decorator(mindspore.nn.Cell)
def apply(obj, fn):
    fn(obj)
    for _, cell in obj.cells_and_names():
        fn(cell)
    return


@x2ms_func_decorator(mindspore.nn.Cell)
def children(obj):
    return obj.cells()


@x2ms_func_decorator(mindspore.nn.Cell)
def modules(obj):
    return (m[1] for m in obj.cells_and_names())


@x2ms_func_decorator(mindspore.nn.Cell)
def named_children(obj):
    return obj.name_cells().items()


@x2ms_func_decorator(mindspore.nn.Cell)
def register_buffer(obj, name, tensor, persistent=True):
    if not hasattr(obj, '_buffers'):
        obj._buffers = OrderedDict()
    obj._buffers[name] = tensor

    if tensor is not None:
        setattr(obj, name, mindspore.Parameter(tensor, requires_grad=False))
    else:
        setattr(obj, name, None)


@x2ms_func_decorator(mindspore.nn.Cell)
def register_forward_hook(obj, hook):
    original_construct = obj.construct
    is_class_obj = inspect.isclass(obj)

    class ForwardHookHandler:
        def __init__(self, module, origin_construct):
            self.module = module
            self.origin_construct = origin_construct

        def remove(self):
            self.module.construct = self.origin_construct

    def new_construct(self, *args):
        inputs = args
        if is_class_obj:
            outputs = original_construct(self, *inputs)
        else:
            outputs = original_construct(*inputs)
        hook(self, inputs, outputs)
        return outputs

    if is_class_obj:
        obj.construct = new_construct
    else:
        obj.construct = MethodType(new_construct, obj)

    return ForwardHookHandler(obj, original_construct)


@x2ms_func_decorator(mindspore.nn.Cell)
def zero_grad(*args, **kwargs):
    pass
    # return None


@x2ms_func_decorator(mindspore.nn.Cell)
def private_load_from_state_dict(obj, state_dict, prefix, local_metadata, strict,
                                 missing_keys, unexpected_keys, error_msgs):
    local_name_params = itertools.chain(obj._params.items())
    local_state = {k: v.data for k, v in local_name_params if v is not None}

    for name, param in local_state.items():
        key = prefix + name
        if key in state_dict:
            new_param = state_dict[key]

            # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
            if len(param.shape) == 0 and len(new_param.shape) == 1:
                new_param = new_param[0]

            if new_param.shape != param.shape:
                # local shape should match the one in checkpoint
                error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                  'the shape in current model is {}.'
                                  .format(key, new_param.shape, param.shape))
                continue

            if isinstance(new_param, mindspore.Parameter):
                # backwards compatibility for serialized parameters
                new_param = new_param.data
            try:
                copy_(param, new_param)
            except Exception:
                error_msgs.append('While copying the parameter named "{}", '
                                  'whose dimensions in the model are {} and '
                                  'whose dimensions in the checkpoint are {}.'
                                  .format(key, param.size, new_param.size))
        elif strict:
            missing_keys.append(key)

    if not strict:
        return

    for key in state_dict.keys():
        if key.startswith(prefix):
            input_name = key[len(prefix):]
            input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
            if input_name not in obj._cells and input_name not in local_state:
                unexpected_keys.append(key)


@property
def _modules(self):
    return OrderedDict(self.name_cells())


@x2ms_func_decorator(mindspore.nn.Cell)
def state_dict(obj, *args, **kwargs):
    result = obj.parameters_dict()
    if len(result) > 0 and list(result.keys())[0].startswith("module.") and not hasattr(obj, 'module'):
        result = OrderedDict(list((k[len("module."):], v) for k, v in result.items()))
    if hasattr(obj, 'x2ms_param_groups') and 'param_groups' not in result:
        result['param_groups'] = obj.x2ms_param_groups
    return result
