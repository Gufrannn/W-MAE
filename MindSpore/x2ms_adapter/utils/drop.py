""" DropBlock, DropPath

PyTorch implementations of DropBlock and DropPath (Stochastic Depth) regularization layers.

Papers:
DropBlock: A regularization method for convolutional networks (https://arxiv.org/abs/1810.12890)

Deep Networks with Stochastic Depth (https://arxiv.org/abs/1603.09382)

Code:
DropBlock impl inspired by two Tensorflow impl that I liked:
 - https://github.com/tensorflow/tpu/blob/master/models/official/resnet/resnet_model.py#L74
 - https://github.com/clovaai/assembled-cnn/blob/master/nets/blocks.py

Hacked together by / Copyright 2020 Ross Wightman
"""
import mindspore
import mindspore.nn as nn
import x2ms_adapter
# import x2ms_adapter.nn_functional
from x2ms_adapter.torch_api.nn_api import nn_functional


def drop_block_2d(
        x, drop_prob: float = 0.1, block_size: int = 7, gamma_scale: float = 1.0,
        with_noise: bool = False, inplace: bool = False, batchwise: bool = False):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. This layer has been tested on a few training
    runs with success, but needs further validation and possibly optimization for lower runtime impact.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    # seed_drop_rate, the gamma parameter
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / (
            (W - block_size + 1) * (H - block_size + 1))

    # Forces the block to be inside the feature map.
    w_i, h_i = x2ms_adapter.meshgrid(x2ms_adapter.to(x2ms_adapter.arange(W), x.device), x2ms_adapter.to(x2ms_adapter.arange(H), x.device))
    valid_block = ((w_i >= clipped_block_size // 2) & (w_i < W - (clipped_block_size - 1) // 2)) & \
                  ((h_i >= clipped_block_size // 2) & (h_i < H - (clipped_block_size - 1) // 2))
    valid_block = x2ms_adapter.to(x2ms_adapter.reshape(valid_block, (1, 1, H, W)), dtype=x.dtype)

    if batchwise:
        # one mask for whole batch, quite a bit faster
        uniform_noise = x2ms_adapter.rand((1, C, H, W), dtype=x.dtype, device=x.device)
    else:
        uniform_noise = torch.rand_like(x)
    block_mask = x2ms_adapter.to(((2 - gamma - valid_block + uniform_noise) >= 1), dtype=x.dtype)
    block_mask = -x2ms_adapter.nn_functional.max_pool2d(
        -block_mask,
        kernel_size=clipped_block_size,  # block_size,
        stride=1,
        padding=clipped_block_size // 2)

    if with_noise:
        normal_noise = x2ms_adapter.randn((1, C, H, W), dtype=x.dtype, device=x.device) if batchwise else torch.randn_like(x)
        if inplace:
            x2ms_adapter.tensor_api.add_(x2ms_adapter.tensor_api.mul_(x, block_mask), normal_noise * (1 - block_mask))
        else:
            x = x * block_mask + normal_noise * (1 - block_mask)
    else:
        normalize_scale = x2ms_adapter.to((x2ms_adapter.tensor_api.numel(block_mask) / x2ms_adapter.tensor_api.add(x2ms_adapter.tensor_api.x2ms_sum(x2ms_adapter.to(block_mask, dtype=mindspore.float32)), 1e-7)), x.dtype)
        if inplace:
            x2ms_adapter.tensor_api.mul_(x, block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


def drop_block_fast_2d(
        x, drop_prob: float = 0.1, block_size: int = 7,
        gamma_scale: float = 1.0, with_noise: bool = False, inplace: bool = False):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. Simplied from above without concern for valid
    block mask at edges.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / (
            (W - block_size + 1) * (H - block_size + 1))

    block_mask = x2ms_adapter.empty_like(x).bernoulli_(gamma)
    block_mask = x2ms_adapter.nn_functional.max_pool2d(
        x2ms_adapter.to(block_mask, x.dtype), kernel_size=clipped_block_size, stride=1, padding=clipped_block_size // 2)

    if with_noise:
        normal_noise = x2ms_adapter.tensor_api.normal_(x2ms_adapter.empty_like(x))
        if inplace:
            x2ms_adapter.tensor_api.add_(x2ms_adapter.tensor_api.mul_(x, 1. - block_mask), normal_noise * block_mask)
        else:
            x = x * (1. - block_mask) + normal_noise * block_mask
    else:
        block_mask = 1 - block_mask
        normalize_scale = x2ms_adapter.to((x2ms_adapter.tensor_api.numel(block_mask) / x2ms_adapter.tensor_api.add(x2ms_adapter.tensor_api.x2ms_sum(x2ms_adapter.to(block_mask, dtype=mindspore.float32)), 1e-6)), dtype=x.dtype)
        if inplace:
            x2ms_adapter.tensor_api.mul_(x, block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


class DropBlock2d(nn.Cell):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf
    """

    def __init__(
            self,
            drop_prob: float = 0.1,
            block_size: int = 7,
            gamma_scale: float = 1.0,
            with_noise: bool = False,
            inplace: bool = False,
            batchwise: bool = False,
            fast: bool = True):
        super(DropBlock2d, self).__init__()
        self.drop_prob = drop_prob
        self.gamma_scale = gamma_scale
        self.block_size = block_size
        self.with_noise = with_noise
        self.inplace = inplace
        self.batchwise = batchwise
        self.fast = fast  # FIXME finish comparisons of fast vs not

    def construct(self, x):
        if not self.training or not self.drop_prob:
            return x
        if self.fast:
            return drop_block_fast_2d(
                x, self.drop_prob, self.block_size, self.gamma_scale, self.with_noise, self.inplace)
        else:
            return drop_block_2d(
                x, self.drop_prob, self.block_size, self.gamma_scale, self.with_noise, self.inplace, self.batchwise)


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        x2ms_adapter.tensor_api.div_(random_tensor, keep_prob)
    return x * random_tensor


class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def construct(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'
