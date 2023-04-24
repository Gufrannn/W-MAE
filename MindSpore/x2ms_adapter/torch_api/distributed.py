#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import numpy as np
import mindspore
import mindspore.communication
from mindspore.ops import ReduceOp, AllReduce, Assign, AllGather


def init_process_group(backend, init_method=None, timeout=-1, world_size=-1, rank=-1, store=None,
                       group_name=''):
    pass


def get_local_rank():
    try:
        return mindspore.communication.get_local_rank()
    # only RuntimeError when is not init, we consider it not run distributedly.
    except RuntimeError:
        return -1


def get_rank():
    """
       Stub function for torch.distributed.get_rank.
       if init can not be called for twice, call release before init again
    """
    try:
        return mindspore.communication.get_rank()
    # only RuntimeError when is not init, we consider it not run distributedly.
    except RuntimeError:
        return 0


def cuda_device_count():
    """
       Stub function for torch.cuda.device_count.
       if init can not be called for twice, call release before init again
    """
    try:
        return mindspore.communication.get_group_size()
    # only RuntimeError when is not init, we consider it not run distributedly.
    except RuntimeError:
        return 1


def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False):
    if not is_initialized():
        return

    if async_op:
        raise ValueError('Async all_reduce is not supported on MindSpore')
    if group is None:
        group = mindspore.communication.GlobalComm.WORLD_COMM_GROUP

    origin_type = tensor.dtype
    trans_type_flag = False
    data = tensor
    if origin_type in (mindspore.float64, mindspore.int64):
        trans_type_flag = True
        data = tensor.astype(mindspore.float32)

    output = AllReduce(op=op, group=group)(data)
    if trans_type_flag:
        output = output.astype(origin_type)

    Assign()(tensor, output)


def reduce(tensor, dst, op=ReduceOp.SUM, group=None, async_op=False):
    if not is_initialized():
        return

    if async_op:
        raise ValueError('Async reduce is not supported on MindSpore')
    if group is None:
        group = mindspore.communication.GlobalComm.WORLD_COMM_GROUP

    rank_id = get_rank()
    if rank_id == dst:
        origin_type = tensor.dtype
        trans_type_flag = False
        data = tensor
        if origin_type in (mindspore.float64, mindspore.int64):
            trans_type_flag = True
            data = tensor.astype(mindspore.float32)

        output = AllReduce(op=op, group=group)(data)
        if trans_type_flag:
            output = output.astype(origin_type)

        tensor.assign_value(output)


@mindspore.ms_function()
def gather(data):
    return AllGather()(data)


def all_gather(tensor_list, tensor, group=None, async_op=False):
    if not is_initialized():
        return

    if async_op:
        raise ValueError('Async all_gather is not supported on MindSpore')

    origin_type = tensor.dtype
    trans_type_flag = False
    data = tensor
    tensor_num = len(tensor_list)
    if origin_type in (mindspore.float64, mindspore.int64):
        trans_type_flag = True
        data = tensor.astype(mindspore.float32)

    out_tensor = gather(data)
    if trans_type_flag:
        out_tensor = out_tensor.astype(origin_type)

    split = mindspore.ops.Split(output_num=tensor_num)
    out_tensors = split(out_tensor)
    for i in range(tensor_num):
        tensor_list[i].assign_value(out_tensors[i])


def barrier(group=None, async_op=False, device_ids=None):
    if not is_initialized():
        return
    value = get_rank()
    if async_op:
        raise ValueError('Async barrier is not supported on MindSpore')
    if group is None:
        group = mindspore.communication.GlobalComm.WORLD_COMM_GROUP

    input_x = mindspore.Tensor(np.array([[value]]).astype(np.float32))
    AllReduce(ReduceOp.SUM, group=group)(input_x)


def is_available():
    return mindspore.context.get_context('device_target') in ('GPU', 'Ascend')


def is_initialized():
    return mindspore.context.get_auto_parallel_context("parallel_mode") != mindspore.context.ParallelMode.STAND_ALONE


def is_nccl_available():
    """
    If MindSpore supports parallel mode, NCCL or HCCL is available
    """
    return mindspore.context.get_auto_parallel_context("parallel_mode") != mindspore.context.ParallelMode.STAND_ALONE


def broadcast(tensor, src, group=None, async_op=False):
    if async_op:
        raise ValueError('Async all_reduce is not supported on MindSpore')
    if group is None:
        group = mindspore.communication.GlobalComm.WORLD_COMM_GROUP

    mindspore.ops.Broadcast(src, group=group)(tensor)
