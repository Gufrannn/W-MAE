import sys, os
sys.path.append( os.path.dirname(os.path.abspath(__file__))+"/../" )

import math
from functools import partial
from re import S
from mindspore.common.initializer import initializer,TruncatedNormal
from utils.drop import DropPath

from utils.img_utils import PeriodicPad2d
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from easydict import EasyDict as edict
from networks.afno_one_step import afnoOneStep

class PrecipNet(afnoOneStep):
    def __init__(self, img_size=(720,1440), patch_size=(8,8), in_chans=20, out_chans=1,
                 embed_dim=768, depth=12, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 patchify_blocks_num=10):
        super().__init__(img_size, patch_size, in_chans, out_chans, 
                         embed_dim, depth, num_heads,
                        decoder_embed_dim, decoder_depth, decoder_num_heads,
                        mlp_ratio, norm_layer, norm_pix_loss, patchify_blocks_num)
        self.p_ppad = PeriodicPad2d(1)
        self.p_conv = nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, pad_mode="valid", padding=0, has_bias=True)
        self.p_act = nn.ReLU()
        self.loss_func = mindspore.nn.MSELoss(reduction='mean')

    def rearrang_v2(self, x, N_in_channels=1):
        B = x.shape[0]
        embed_dim = int(N_in_channels * (self.patch_size[0] * self.patch_size[1]))
        h = int(self.img_size[0] // (self.patch_size[0]))
        w = int(self.img_size[1] // (self.patch_size[1]))
         
        x = mindspore.ops.reshape(x, (B, h, w, embed_dim))  # (2, 4050, 5120)
        x = mindspore.ops.reshape(x, (B, h, w, self.patch_size[0], self.patch_size[1], N_in_channels))
        x = mindspore.ops.Transpose()(x, (0, 5, 1, 3, 2, 4))   # "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
        x = mindspore.ops.reshape(x, (B, N_in_channels, self.img_size[0], self.img_size[1]))
        return x
        
    def forward_loss(self, imgs, pred):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(axis=-1, keep_dims=True)
            var = target.var(axis=-1, keepdims=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        loss = (pred - target) ** 2
        loss = loss.mean(axis=-1)  # [N, L], mean loss per patch
        mask = mindspore.ops.OnesLike()(loss)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        return loss
    
    def precip_predict(self, x):
        return self.p_act(self.p_conv(self.p_ppad(x)))
        
    def construct(self, prev, future):
        latent = self.forward_encoder(prev)
        # return latent
        pred_patch = self.forward_decoder(latent)  # [bs, patch_num, p*p*1]
        # print(pred.shape)
        pred_img = self.rearrang_v2(pred_patch) 
        pred_img = self.precip_predict(pred_img)   # (bs, 1, 720, 1440)
        loss = self.loss_func(pred_img, future)
        
        if self.phase != "train":
            return loss, pred_img, None
        else:
            return loss
        
def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = PrecipNet(
        embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=6, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model

def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = PrecipNet(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model

def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = PrecipNet(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model
        
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks