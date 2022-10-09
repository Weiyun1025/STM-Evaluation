import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, Mlp
import math


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class HaloAttn(nn.Module):
    """ Halo Attention

    Paper: `Scaling Local Self-Attention for Parameter Efficient Visual Backbones`
        - https://arxiv.org/abs/2103.12731

    The internal dimensions of the attention module are controlled by the interaction of several arguments.
      * the output dimension of the module is specified by dim_out, which falls back to input dim if not set
      * the value (v) dimension is set to dim_out // num_heads, the v projection determines the output dim
      * the query and key (qk) dimensions are determined by
        * num_heads * dim_head if dim_head is not None
        * num_heads * (dim_out * attn_ratio // num_heads) if dim_head is None
      * as seen above, attn_ratio determines the ratio of q and k relative to the output if dim_head not used

    Args:
        dim (int): input dimension to the module
        dim_out (int): output dimension of the module, same as dim if not set
        feat_size (Tuple[int, int]): size of input feature_map (not used, for arg compat with bottle/lambda)
        stride: output stride of the module, query downscaled if > 1 (default: 1).
        num_heads: parallel attention heads (default: 8).
        dim_head: dimension of query and key heads, calculated from dim_out * attn_ratio // num_heads if not set
        block_size (int): size of blocks. (default: 8)
        halo_size (int): size of halo overlap. (default: 3)
        qk_ratio (float): ratio of q and k dimensions to output dimension when dim_head not set. (default: 1.0)
        qkv_bias (bool) : add bias to q, k, and v projections
        avg_down (bool): use average pool downsample instead of strided query blocks
        scale_pos_embed (bool): scale the position embedding as well as Q @ K
    """

    def __init__(self,
                 dim,
                 dim_out=None,
                 feat_size=None,
                 stride=1,
                 num_heads=8,
                 dim_head=None,
                 block_size=8,
                 halo_size=3,
                 qk_ratio=1.0,
                 qkv_bias=False,
                 avg_down=False,
                 scale_pos_embed=False):
        super().__init__()
        dim_out = dim_out or dim
        assert dim_out % num_heads == 0
        assert stride in (1, 2)
        self.num_heads = num_heads
        self.dim_head_qk = dim_head or make_divisible(dim_out * qk_ratio,
                                                      divisor=8) // num_heads
        self.dim_head_v = dim_out // self.num_heads
        self.dim_out_qk = num_heads * self.dim_head_qk
        self.dim_out_v = num_heads * self.dim_head_v
        self.scale = self.dim_head_qk**-0.5
        self.scale_pos_embed = scale_pos_embed
        self.block_size = self.block_size_ds = block_size
        self.halo_size = halo_size
        self.win_size = block_size + halo_size * 2  # neighbourhood window size

        # pre-calculate the block area
        self.q_win_s = self.block_size ** 2
        self.kv_win_s = self.win_size ** 2

        self.block_stride = 1
        use_avg_pool = False

        # FIXME not clear if this stride behaviour is what the paper intended
        # Also, the paper mentions using a 3D conv for dealing with the blocking/gather, and leaving
        # data in unfolded block form. I haven't wrapped my head around how that'd look.
        # self.kv = nn.Conv2d(dim,
        #                     self.dim_out_qk + self.dim_out_v,
        #                     1,
        #                     bias=qkv_bias)
        self.q = nn.Linear(dim, self.dim_out_qk, bias=qkv_bias)
        self.k = nn.Linear(dim, self.dim_out_qk, bias=qkv_bias)
        self.v = nn.Linear(dim, self.dim_out_v, bias=qkv_bias)


        # generate absolute position embedding from DETR
        q_pos, k_pos = self.get_abs_pos_embed(self.win_size, self.halo_size, dim=dim//2)
        self.q_pos = nn.Parameter(q_pos)
        self.k_pos = nn.Parameter(k_pos)
        self.q_pos.requires_grad = False
        self.k_pos.requires_grad = False
        #----------------------------------------------------------------

        self.pool = nn.AvgPool2d(2, 2) if use_avg_pool else nn.Identity()
        self.proj = nn.Linear(self.dim_out_v, self.dim_out_v)

        self.H, self.W = None, None
        self.mask = None

    
    def get_abs_pos_embed(self,
                          win_size,
                          halo_size,
                          dim,
                          temperature=10000,
                          normalize=True,
                          scale=2 * math.pi,
                          eps=1e-6,
                          offset=0.,):

        mask = torch.ones((win_size, win_size), dtype=torch.long)
        y_embed = mask.cumsum(0, dtype=torch.float32) # [h, w], recording the y coordinate ot each pixel
        x_embed = mask.cumsum(1, dtype=torch.float32)
        if normalize: # default True
            y_embed = (y_embed + offset) / \
                      (y_embed[-1:, :] + eps) * scale
            x_embed = (x_embed + offset) / \
                      (x_embed[:, -1:] + eps) * scale

        dim_t = torch.arange(dim, dtype=torch.float32, device=mask.device)
        dim_t = temperature**(2 * (dim_t // 2) / dim)
        pos_x = x_embed[:, :, None] / dim_t   # [h, w, num_feats]
        pos_y = y_embed[:, :, None] / dim_t

        # use `view` instead of `flatten` for dynamically exporting to ONNX
        H, W = mask.size()
        pos_x = torch.stack(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()),
            dim=3).view(H, W, -1) # [h, w, num_feats]
        pos_y = torch.stack(
            (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()),
            dim=3).view(H, W, -1) 
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1) # [2 * num_feats, win_size, win_size]

        q_pos = pos[:, halo_size:-halo_size, halo_size:-halo_size]
        k_pos = pos
    
        return q_pos.flatten(1,2).transpose(0, 1), k_pos.flatten(1,2).transpose(0,1) # []

    def forward(self, x):
        B, H, W, C = x.shape
        assert H % self.block_size == 0
        assert W % self.block_size == 0
        num_h_blocks = H // self.block_size
        num_w_blocks = W // self.block_size
        num_blocks = num_h_blocks * num_w_blocks

        #q = self.q(x)
        # unfold
        q = x.reshape(-1, num_h_blocks, self.block_size_ds, num_w_blocks,
                      self.block_size_ds, self.dim_out_qk).permute(0, 1, 3, 2, 4, 5) # [bs, h, w, wh, ww, dim]
        q = q.reshape(B, num_blocks, self.q_win_s, self.dim_out_qk) # [bs, num_blocks, win_s, dim]
        q = self.q(q + self.q_pos[None, None, :, :]) # [bs, num_blocks, win_s, dim]
        q = q.reshape(B, num_blocks, self.q_win_s, self.num_heads, -1).permute(0, 3, 1, 2, 4) # [bs, num_heads, num_blocks, win_s, head_dim]


        #kv = self.kv(x.permute(0, 3, 1, 2))
        kv = F.pad(
            x.permute(0, 3, 1, 2),
            [
                self.halo_size,
                self.halo_size,
                self.halo_size,
                self.halo_size,
            ],
        )
        kv = kv.unfold(2, self.win_size, self.block_size).unfold(
            3, self.win_size,
            self.block_size).reshape(-1, 
                                     self.dim_out_qk,
                                     num_blocks,
                                     self.kv_win_s).permute(0, 2, 3, 1) # [bs, num_b, win_s, dim]
        k = self.k(kv + self.k_pos[None, None, :, :])
        v = self.v(kv)
        k = k.reshape(B, num_blocks, self.kv_win_s, self.num_heads, -1).permute(0, 3, 1, 2, 4) # [bs, num_heads, num_blocks, win_s, head_dim]
        v = v.reshape(B, num_blocks, self.kv_win_s, self.num_heads, -1).permute(0, 3, 1, 2, 4) 

        attn = (q * self.scale) @ k.transpose(-1, -2)

        max_neg_value = -torch.finfo(attn.dtype).max
        attn.masked_fill_(self.get_mask(H, W, attn.device), max_neg_value)

        # B, num_heads, num_blocks, block_size ** 2, win_size ** 2
        attn = attn.softmax(dim=-1)

        out = attn @ v
        # B, num_heads, num_blocks, block_size ** 2, dim_head_v
        # fold
        out = out.reshape(-1, self.num_heads, num_h_blocks, num_w_blocks,
                          self.block_size_ds, self.block_size_ds,
                          self.dim_head_qk)
        out = out.permute(0, 2, 4, 3, 5, 1, 6).reshape(B, H, W, self.dim_out_v)
        out = self.proj(out)
        # B, H, W, dim_out
        return out

    def get_mask(self, H, W, device):
        if self.H == H and self.W == W and self.mask is not None:
            return self.mask

        num_h_blocks = H // self.block_size
        num_w_blocks = W // self.block_size
        num_blocks = num_h_blocks * num_w_blocks

        mask = torch.ones((1, 1, H, W), device=device)
        mask = F.pad(mask, [self.halo_size, self.halo_size, self.halo_size, self.halo_size])
        mask = mask.unfold(2, self.win_size, self.block_size)
        mask = mask.unfold(3, self.win_size, self.block_size)
        mask = mask.reshape(1, num_blocks, self.win_size * self.win_size)
        mask = mask.unsqueeze(-2)

        # 1, num_blocks, 1, win_size * win_size
        mask = mask.bool()

        self.H = H
        self.W = W
        self.mask = ~mask
        return self.mask


class HaloBlockV2(nn.Module):

    def __init__(self,
                 dim,
                 drop_path,
                 layer_scale_init_value,
                 block_size,
                 halo_size,
                 stage,
                 num_heads,
                 mlp_ratio=4.,
                 drop=0.,
                 act_layer=nn.GELU,
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm((dim, ))
        self.attn = HaloAttn(dim=dim,
                             dim_out=dim,
                             num_heads=num_heads[stage],
                             block_size=block_size,
                             halo_size=halo_size)

        self.gamma_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((1, 1, 1, dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm((dim, ))
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer,
                       drop=drop)

        self.gamma_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((1, 1, 1, dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None

    @staticmethod
    def pre_stage_transform(x):
        return x.permute(0, 2, 3, 1)

    @staticmethod
    def post_stage_transform(x):
        return x.permute(0, 3, 1, 2)

    def forward(self, x):
        # shape: (B, H, W, C)
        shortcut = x
        x = self.attn(self.norm1(x))

        if self.gamma_1 is not None:
            x = self.gamma_1 * x
        x = shortcut + self.drop_path(x)

        # FFN
        shortcut = x
        x = self.mlp(self.norm2(x))

        if self.gamma_2 is not None:
            x = self.gamma_2 * x
        x = shortcut + self.drop_path(x)

        return x
