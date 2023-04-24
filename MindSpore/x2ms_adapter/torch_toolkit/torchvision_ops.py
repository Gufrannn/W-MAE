#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import mindspore
import numpy as np
from ..torch_api.torch_base_api import empty


def nms(boxes, scores, iou_threshold=0.5):
    scores = scores.view(-1)
    boxes = boxes.asnumpy()
    scores = scores.asnumpy()
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    reserved_boxes = []
    while order.size > 0:
        i = order[0]
        reserved_boxes.append(i)
        max_x1, max_y1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        min_x2, min_y2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])

        intersect_w, intersect_h = np.maximum(0.0, min_x2 - max_x1 + 1), np.maximum(0.0, min_y2 - max_y1 + 1)
        intersect_area = intersect_w * intersect_h
        area = areas[i] + areas[order[1:]] - intersect_area
        flag = area == 0
        area[flag] = 1
        ovr = intersect_area / area
        ovr[flag] = 0
        indexes = np.where(ovr <= iou_threshold)[0]
        order = order[indexes + 1]
    return mindspore.Tensor(np.array(reserved_boxes))


def roi_align(input, boxes, output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=False):
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    if not aligned:
        roi_end_mode = 0
    else:
        roi_end_mode = 1
    if boxes.size == 0:
        return mindspore.ops.Zeros()((0, input.shape[1], *output_size), mindspore.float32)
    op = mindspore.ops.ROIAlign(pooled_height=output_size[0], pooled_width=output_size[1],
                                spatial_scale=spatial_scale, sample_num=sampling_ratio, roi_end_mode=roi_end_mode)
    return op(input, boxes)


def box_area(boxes: mindspore.Tensor) -> mindspore.Tensor:
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def get_image_backend():
    return "PIL"


def batched_nms(boxes, scores, classes, iou_threshold):
    if mindspore.ops.Size()(boxes) == 0:
        return empty((0,), dtype=mindspore.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets_1 = classes.astype(boxes.dtype) * (max_coordinate + mindspore.Tensor(1).astype(boxes.dtype))
    boxes_for_nms = boxes + offsets_1[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


def is_tracing():
    return False
