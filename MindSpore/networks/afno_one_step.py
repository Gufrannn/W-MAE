import sys, os
sys.path.append( os.path.dirname(os.path.abspath(__file__))+"/../" )

from functools import partial
from utils.img_utils import PeriodicPad2d
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from networks.mae import MaskedAutoencoderViT

class afnoOneStep(MaskedAutoencoderViT):
    def __init__(self, img_size=(720,1440), patch_size=(8,8), in_chans=20, out_chans=20,
                 embed_dim=768, depth=12, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 patchify_blocks_num=10):
        super().__init__(img_size, patch_size, in_chans, out_chans,
                         embed_dim, depth, num_heads,
                        decoder_embed_dim, decoder_depth, decoder_num_heads,
                        mlp_ratio, norm_layer, norm_pix_loss, patchify_blocks_num)

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)  
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
    
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]  # [1,1,emb_dim]
        cls_tokens = mindspore.numpy.tile(cls_token, (x.shape[0], 1, 1))
        x = ops.Concat(axis=1)((cls_tokens, x))

        # apply Transformer blocks
        x = ops.Cast()(x, mindspore.float32)
        x = self.blocks(x)   # float16
        x = self.norm(x)
        return x
    
    def forward_decoder(self, x):
        # embed tokens
        x = self.decoder_embed(x)
        
        # add pos embed
        x_ = x[:, 1:, :]
        x = x_ + self.decoder_pos_embed

        x = ops.Cast()(x, mstype.float32)
        x = self.decoder_blocks(x)   # float16
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x
        
    def forward_loss(self, imgs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
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
        
    def construct(self, prev, future):
        latent = self.forward_encoder(prev)
        # return latent
        pred = self.forward_decoder(latent)  # [N, L, p*p*3]
        loss = self.forward_loss(future, pred)
        if self.phase != "train":
            return loss, pred, loss
        else:
            return loss
        
        
def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = afnoOneStep(
        embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=6, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model

def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = afnoOneStep(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model

def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = afnoOneStep(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model
        
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
        
class PrecipNet(nn.Cell):
    def __init__(self, N_out_channels, backbone):
        super().__init__()
        self.out_chans = N_out_channels
        self.backbone = backbone
        self.ppad = PeriodicPad2d(1)
        self.conv = nn.Conv2d(self.out_chans, self.out_chans, kernel_size=3, stride=1, padding=0, has_bias=True)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.backbone(x)
        x = self.ppad(x)
        x = self.conv(x)
        x = self.act(x)
        return x