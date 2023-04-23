#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
from functools import wraps


def x2ms_func_decorator(*required_type, origin_mapping_func=''):
    def x2ms_fun_decorator(func):
        @wraps(func)
        def wrapper(input_obj, *args, **kwargs):
            # check input_obj is whether the required_type
            if isinstance(input_obj, super) and isinstance(input_obj.__self__, required_type) or \
                    isinstance(input_obj, required_type):
                return func(input_obj, *args, **kwargs)
            else:
                if origin_mapping_func:
                    origin_method = origin_mapping_func
                else:
                    origin_method = func.__name__.replace('x2ms_', '')
                return getattr(input_obj, origin_method)(*args, **kwargs)

        return wrapper

    return x2ms_fun_decorator
