#reference: https://github.com/NVlabs/AFNO-transformer
import sys, os
sys.path.append( os.path.dirname(os.path.abspath(__file__))+"/../" )

import math
from functools import partial
from re import S
from numpy.lib.arraypad import pad
import numpy as np
from mindspore.common.initializer import initializer,TruncatedNormal
from utils.pos_emb import get_2d_sincos_pos_embed_v1
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype



class Mlp(nn.Cell):
    def __init__(self, 
                 in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.drop = nn.Dropout(keep_prob = float(1 - drop))

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class MaskedAutoencoderViT(nn.Cell):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=(720,1440), patch_size=(8,8), in_chans=20, out_chans=20,
                 embed_dim=768, depth=12, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 patchify_blocks_num=10):
        super().__init__()
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.img_size =img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.patch_embed = PatchEmbed_v1(img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = mindspore.Parameter(initializer('zeros', (1, 1, embed_dim)))
        self.pos_embed = mindspore.Parameter(initializer('zeros', (1, num_patches + 1, embed_dim)), requires_grad=False)

        from easydict import EasyDict as edict
        from networks import vit
        vit_cfg = edict({
            'body_drop_path_rate': 0.0,
            'body_norm': norm_layer, 
            
            # body attention
            'attention_init': mindspore.common.initializer.XavierUniform(),
            'attention_activation': mindspore.nn.Softmax(),
            'attention_dropout_rate': 0.,
            'attention': vit.Attention, 
            'attention_init': mindspore.common.initializer.XavierUniform(),

            # body feedforward
            'feedforward_init': mindspore.common.initializer.XavierUniform(),
            'feedforward_activation': mindspore.nn.GELU(),
            'feedforward_dropout_rate': 0.,
            'feedforward': vit.FeedForward,

            'configs': edict({
            'depth': depth,
            'd_model': embed_dim,
            'dim_head': embed_dim//num_heads,
            'normalized_shape': embed_dim,
            'heads': num_heads,
            'mlp_dim': embed_dim *  mlp_ratio
            }),
        })
        print(f"vit_cfg: {vit_cfg}")
        self.blocks = vit.Transformer(vit_cfg)
        self.norm = norm_layer([embed_dim])
        # MAE decoder specifics
        self.decoder_embed = nn.Dense(embed_dim, decoder_embed_dim, has_bias=True)

        self.mask_token = mindspore.Parameter(initializer('zeros', (1, 1, decoder_embed_dim)) )

        self.decoder_pos_embed = mindspore.Parameter(initializer('zeros', (1, num_patches, decoder_embed_dim)), requires_grad=False) # fixed sin-cos embedding

        vit_cfg_dec = edict({
            'body_drop_path_rate': 0.0,
            'body_norm': norm_layer, 
            
            # body attention
            'attention_init': mindspore.common.initializer.XavierUniform(),
            'attention_activation': mindspore.nn.Softmax(),
            'attention_dropout_rate': 0.,
            'attention': vit.Attention, 
            'attention_init': mindspore.common.initializer.XavierUniform(),

            # body feedforward
            'feedforward_init': mindspore.common.initializer.XavierUniform(),
            'feedforward_activation': mindspore.nn.GELU(),
            'feedforward_dropout_rate': 0.,
            'feedforward': vit.FeedForward,

            'configs': edict({
            'depth': decoder_depth,
            'd_model': decoder_embed_dim,
            'dim_head': decoder_embed_dim//decoder_num_heads,
            'normalized_shape': decoder_embed_dim,
            'heads': decoder_num_heads,
            'mlp_dim': decoder_embed_dim *  mlp_ratio
            }),
        })
        print(f"vit_cfg: {vit_cfg}")

        self.decoder_blocks = vit.Transformer(vit_cfg_dec)

        self.decoder_norm = norm_layer([decoder_embed_dim])
        self.decoder_pred = nn.Dense(decoder_embed_dim, self.out_chans * self.patch_size[0] * self.patch_size[1], has_bias=True)  # decoder to patch

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()
    
    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed_v1(
                self.pos_embed.shape[-1], 
                ([self.img_size, self.patch_size]), cls_token=True
            ).astype(np.float32)
        self.pos_embed.set_data( ops.ExpandDims()(mindspore.Tensor(pos_embed), 0) )

        decoder_pos_embed = get_2d_sincos_pos_embed_v1(
                self.decoder_pos_embed.shape[-1],
                ([self.img_size, self.patch_size]), cls_token=False
            ).astype(np.float32)
        self.decoder_pos_embed.set_data( ops.ExpandDims()(mindspore.Tensor(decoder_pos_embed), 0))

        # initialize patch_embed like nn.Dense (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data  # conv2d
        w_init_weight = initializer('xavier_uniform', w.reshape((w.shape[0], -1)).shape, mindspore.float32)

        self.patch_embed.proj.weight.set_data(w_init_weight.reshape(w.shape))

        self.cls_token.set_data(initializer(
            mindspore.common.initializer.Normal(sigma=0.01, ), self.cls_token.shape, self.cls_token.dtype))
        self.mask_token.set_data(initializer(
            mindspore.common.initializer.Normal(sigma=0.01, ), self.mask_token.shape, self.mask_token.dtype))

        self._init_weights()

    def _init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(initializer(
                    'xavier_uniform', cell.weight.shape, cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(initializer('zeros',
                                                    cell.bias.shape,
                                                    cell.bias.dtype))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(initializer('ones',
                                                cell.gamma.shape,
                                                cell.gamma.dtype))
                cell.beta.set_data(initializer('zeros',
                                                cell.beta.shape,
                                                cell.beta.dtype))

    def patchify(self, imgs):
        p = self.patch_embed.patch_size[0]

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape((imgs.shape[0], self.in_chans, h, p, w, p))
        x = ops.transpose(x, (0,2,4,3,5,1))
        x = x.reshape((imgs.shape[0], h * w, p ** 2 * self.in_chans))
        return x

    def unpatchify(self, x):
        raise NotImplementedError

    def random_masking(self, x, ids_keep):
        tmp_index = ops.Cast()(
            mindspore.numpy.tile(ops.ExpandDims()(ids_keep, -1), (1, 1, self.embed_dim)), mindspore.int32) 

        x_masked = ops.gather_elements(x, dim=1, index=tmp_index)
        return x_masked

    def forward_encoder(self, x, ids_keep, mask, ids_restore):
        # embed patches
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        x = self.random_masking(x, ids_keep)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]  
        cls_tokens = mindspore.numpy.tile(cls_token, (x.shape[0], 1, 1))
        x = ops.Concat(axis=1)((cls_tokens, x))

        x = ops.Cast()(x, mindspore.float32)
        x = self.blocks(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)

        mask_tokens = mindspore.numpy.tile(self.mask_token, (x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1))
        mask_tokens = ops.Cast()(mask_tokens, x.dtype)
        x_ = ops.Concat(axis=1)((x[:, 1:, :], mask_tokens))

        x_ = ops.gather_elements(x_, 
                        dim=1, 
                        index=mindspore.numpy.tile(ops.ExpandDims()(ids_restore, -1), (1, 1, x.shape[2]))
                        )
        
        x = x_ + self.decoder_pos_embed

        x = ops.Cast()(x, mstype.float32)
        x = self.decoder_blocks(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)

        return x

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(axis=-1, keep_dims=True)
            var = target.var(axis=-1, keepdims=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        loss = (pred - target) ** 2
        loss = loss.mean(axis=-1)
        loss = (loss * mask).sum() / mask.sum()

        return loss

    def construct(self, imgs, ids_keep, mask, ids_restore):
        latent, mask, ids_restore = self.forward_encoder(imgs, ids_keep, mask, ids_restore)
        # return latent
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        if self.phase != "train":
            return loss, pred, mask
        else:
            return loss


class PatchEmbed_v1(nn.Cell):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=20, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)


    def tensor_transpose(self, tensor, dim0, dim1):
        dim = tensor.ndim
        _dim0 = dim0 if dim0 >= 0 else (dim0 + dim)
        _dim1 = dim1 if dim1 >= 0 else (dim1 + dim)
        dim_list = list(range(dim))
        dim_list[_dim0] = _dim1
        dim_list[_dim1] = _dim0
        return tensor.transpose(*dim_list)
    
    def flatten(self, input_tensor, start_dim=0, end_dim=-1):
        shape_tuple = input_tensor.shape
        _start_dim = start_dim if start_dim >= 0 else (start_dim + input_tensor.ndim)
        _end_dim = end_dim if end_dim >= 0 else (end_dim + input_tensor.ndim)
        new_dim = 1
        for idx in range(_start_dim, _end_dim + 1):
            new_dim *= shape_tuple[idx]
        new_shape_list = []
        for i in shape_tuple[:_start_dim]:
            new_shape_list.append((i,))
        new_shape_list.append((new_dim,))
        new_shape_tuple = ()
        for i in new_shape_list:
            new_shape_tuple += i

        if 0 in new_shape_tuple:
            return mindspore.ops.Zeros()(new_shape_tuple, input_tensor.dtype)

        reshape_ops = mindspore.ops.Reshape()
        return reshape_ops(input_tensor, new_shape_tuple)

    def construct(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.tensor_transpose(self.flatten(self.proj(x), 2), 1, 2)
        return x


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=6, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model

def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model

def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks