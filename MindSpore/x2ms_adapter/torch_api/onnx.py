#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.


def export(model, args, f, export_params=True, verbose=False, training='', input_names=None, output_names=None,
           operator_export_type=None, opset_version=None, do_constant_folding=True, dynamic_axes=None,
           keep_initializers_as_inputs=None, custom_opsets=None, export_modules_as_functions=False):
    raise NotImplementedError("Currently, The function of exporting models is not implemented..")


def is_in_onnx_export():
    return False
