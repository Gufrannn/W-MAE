#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
import mindspore


# timm
class MixupStub:
    def __init__(self, mixup_alpha=1., cutmix_alpha=0., cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                 mode='batch', correct_lam=True, label_smoothing=0.1, num_classes=1000):
        self.num_classes = num_classes
        self.one_hot = mindspore.ops.OneHot()

    def __call__(self, x, target):
        target = self.one_hot(target.astype(mindspore.int64), self.num_classes,
                              mindspore.Tensor(1.0, dtype=mindspore.float32),
                              mindspore.Tensor(0.0, dtype=mindspore.float32))
        return x, target


# tensorwatch
class ModelStats:
    def __init__(self, model, input_shape):
        pass

    def to_html(self, buf=None):
        pass

    def to_csv(self, path_or_buf=None):
        pass

    def iloc(self):
        pass


# ptflops
def get_model_complexity_info(model, input_res, print_per_layer_stat=True, as_strings=True, verbose=False):
    pass
