#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
from distutils.version import LooseVersion

import mindspore
import mindspore.nn


def auc(x, y, reorder=False):
    return mindspore.Tensor(mindspore.nn.auc(x.asnumpy(), y.asnumpy(), reorder)).astype(mindspore.float32)


_CONFUSION_NORMALIZE_MAP = {
    None: 'no_norm',
    'none': 'no_norm',
    'true': 'target',
    'pred': 'prediction',
    'all': 'all'
}


class ConfusionMatrix(mindspore.nn.ConfusionMatrix):
    def __init__(self, num_classes, normalize=None, threshold=0.5, multilabel=False, **kwargs):
        super().__init__(num_classes=num_classes, normalize=_CONFUSION_NORMALIZE_MAP.get(normalize, 'no_norm'),
                         threshold=threshold)
        self._compute_confusion_matrix = mindspore.nn.ConfusionMatrix(
            num_classes=num_classes, normalize=_CONFUSION_NORMALIZE_MAP.get(normalize, 'no_norm'), threshold=threshold)

    def __call__(self, preds, target):
        self.update(preds, target)
        return mindspore.Tensor(self._compute_confusion_matrix(preds, target))

    def compute(self):
        return mindspore.Tensor(self.eval())


class BLEUScore(mindspore.nn.BleuScore):
    def __init__(self, n_gram=4, smooth=False, **kwargs):
        super().__init__(n_gram=n_gram, smooth=smooth)
        self._compute_bleu_score = mindspore.nn.BleuScore(n_gram=n_gram, smooth=smooth)

    def __call__(self, preds, target):
        self.update(preds, target)
        _preds, _target = self._transform_input(preds, target)
        return mindspore.Tensor(self._compute_bleu_score(_preds, _target))

    @staticmethod
    def _transform_input(preds, target):
        _preds = list(pred.split(' ') for pred in preds)
        _target = []
        for target_list in target:
            sub_target = []
            for target_value in target_list:
                sub_target.append(target_value.split(' '))
            _target.append(sub_target)
        return _preds, _target

    def update(self, preds, target):
        preds, target = self._transform_input(preds, target)
        super().update(preds, target)

    def compute(self):
        return mindspore.Tensor(self.eval())


class MeanSquaredError(mindspore.nn.MSE):
    def __init__(self, squared=True, **kwargs):
        super().__init__()
        self.squared = squared
        self._compute_op = mindspore.nn.MSE()

    def __call__(self, preds, target):
        self.update(preds, target)
        result = mindspore.Tensor(self._compute_op(preds, target))
        return result if self.squared else mindspore.numpy.sqrt(result)

    def update(self, preds, target):
        super().update(preds, target)

    def compute(self):
        result = mindspore.Tensor(self.eval())
        return result if self.squared else mindspore.numpy.sqrt(result)


class MeanAbsoluteError(mindspore.nn.MAE):
    def __init__(self, **kwargs):
        super().__init__()
        self._compute_op = mindspore.nn.MAE()

    def __call__(self, preds, target):
        self.update(preds, target)
        return mindspore.Tensor(self._compute_op(preds, target))

    def update(self, preds, target):
        super().update(preds, target)

    def compute(self):
        return mindspore.Tensor(self.eval())


def structural_similarity_index_measure(preds, target, gaussian_kernel=True, sigma=1.5, kernel_size=11,
                                        reduction='elementwise_mean', data_range=None, k1=0.01, k2=0.03,
                                        return_full_image=False, return_contrast_sensitivity=False):
    if not gaussian_kernel:
        raise ValueError(f'Mindspore currently only support gaussian kernel')
    if return_full_image or return_contrast_sensitivity:
        raise ValueError(
            f'MindSpore currently does not support parameter "return_full_image" or "return_contrast_sensitivity"')
    if not isinstance(data_range, (int, float)):
        data_range = max(preds.max() - preds.min(), target.max() - target.min()).asnumpy().item()
    _compute_op = mindspore.nn.SSIM(max_val=data_range, filter_size=kernel_size,
                                    filter_sigma=sigma, k1=k1, k2=k2)
    result = _compute_op(preds, target)
    if reduction == 'elementwise_mean':
        return result.mean()
    elif reduction == 'sum':
        return result.sum()
    else:
        return result


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
        result = result.broadcast_to((len(betas),))
    return result


if LooseVersion(mindspore.__version__) < LooseVersion('1.8.0'):
    from ..ms_1_7_1.torchmetrics_ops import multiscale_structural_similarity_index_measure
