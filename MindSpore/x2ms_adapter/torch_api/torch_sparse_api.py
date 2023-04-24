#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import mindspore


class FloatTensor(mindspore.Tensor):
    def __init__(self, index, value, shape=None):
        """
        MindSpore 1.9.0 most operations do not support COOTensor
        To minimize the judgment logic operations, return dense COOTensor which is a normal MindSporeTensor
        """
        if index.ndim != 2:
            raise TypeError(f'MindSpore 1.9.0 currently only support index.ndim == 2, '
                            f'but got index.ndim with {index.ndim}')
        if shape is None:
            shape0, shape1 = index.asnumpy().max(axis=1).tolist()
            shape = (shape0 + 1, shape1 + 1)

        self.sparse_index = index
        self.sparse_value = value
        self.sparse_shape = shape

        super(FloatTensor, self).__init__(
            mindspore.COOTensor(indices=index.T, values=value, shape=shape).to_dense())

    def _indices(self):
        return self.sparse_index

    def _values(self):
        return self.sparse_value


def mm(mat1, mat2):
    return mindspore.ops.matmul(mat1, mat2)

