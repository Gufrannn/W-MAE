#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
import mindspore


def multiscale_structural_similarity_index_measure(preds, target, gaussian_kernel=True, sigma=1.5, kernel_size=11,
                                                   reduction='elementwise_mean', data_range=None, k1=0.01, k2=0.03,
                                                   betas=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333), normalize=None):
    if not isinstance(data_range, (int, float)):
        raise TypeError(f'MindSpore currently only support data_range of float or int type. Got {data_range}')
    if not gaussian_kernel:
        raise ValueError(f'Mindspore currently only support gaussian kernel')
    _compute_op = mindspore.nn.MSSSIM(max_val=data_range, power_factors=betas, filter_size=kernel_size,
                                      filter_sigma=sigma, k1=k1, k2=k2)
    result = _compute_op(preds, target)
    if reduction == 'none':
        result = mindspore.ops.BroadcastTo((len(betas),))(result)
    return result
