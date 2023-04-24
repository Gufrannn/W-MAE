#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import mindspore
import mindspore.scipy.linalg as linalg

from ..utils.util_api import out_adaptor


def inv(A, *, out=None):
    if A.ndim == 2:
        result = linalg.inv(A)
    else:
        result = mindspore.ops.MatrixInverse()(A)
    return out_adaptor(result, out)


def cholesky(A, *, upper=False, out=None):
    result = linalg.cholesky(A, not upper)
    return out_adaptor(result, out)


def eigh(A, UPLO='L', *, out=None):
    result = linalg.eigh(A, lower=UPLO == 'L')
    return out_adaptor(result, out)


def lu(A, *, pivot=True, out=None):
    if pivot:
        result = linalg.lu(A)
        return out_adaptor(result, out)
    else:
        raise NotImplementedError('"pivot=False" is not supported in mindspore')


def lu_factor(A, *, pivot=True, out=None):
    if pivot:
        output, piv = linalg.lu_factor(A)
        # The pivot indices tensor in mindspore is less one than torch.
        piv += 1
        result = (output, piv)
        return out_adaptor(result, out)
    else:
        raise NotImplementedError('"pivot=False" is not supported in mindspore')


def solve_triangular(A, B, *, upper, left=True, unitriangular=False, out=None):
    if left:
        result = linalg.solve_triangular(A, B, lower=not upper, unit_diagonal=unitriangular)
    else:
        result = linalg.solve_triangular(A, B.T, lower=not upper, trans=1, unit_diagonal=unitriangular).T
    return out_adaptor(result, out)


def det(A, *, out=None):
    result = linalg.det(A)
    return out_adaptor(result, out)
