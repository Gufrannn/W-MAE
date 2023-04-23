#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import mindspore
# import sklearn


def sklearn_metrics_f1_score(y_true, y_pred, **kwargs):
    if isinstance(y_true, mindspore.Tensor):
        y_true = y_true.asnumpy()
    if isinstance(y_pred, mindspore.Tensor):
        y_pred = y_pred.asnumpy()

    return sklearn.metrics.f1_score(y_true, y_pred, **kwargs)


def sklearn_metrics_precision_score(y_true, y_pred, **kwargs):
    if isinstance(y_true, mindspore.Tensor):
        y_true = y_true.asnumpy()
    if isinstance(y_pred, mindspore.Tensor):
        y_pred = y_pred.asnumpy()

    return sklearn.metrics.precision_score(y_true, y_pred, **kwargs)


def sklearn_metrics_recall_score(y_true, y_pred, **kwargs):
    if isinstance(y_true, mindspore.Tensor):
        y_true = y_true.asnumpy()
    if isinstance(y_pred, mindspore.Tensor):
        y_pred = y_pred.asnumpy()

    return sklearn.metrics.recall_score(y_true, y_pred, **kwargs)
