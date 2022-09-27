"""
HaloNet with pre-generated relative positional embedding.
"""
import torch
import torch.nn.functional as F
from torch import nn
from timm.models import register_model
from timm.models.layers import DropPath, LayerNorm2d, Mlp
from timm.models.layers.halo_attn import rel_logits_1d


from timm.models.layers.helpers import make_divisible
from timm.models.layers.weight_init import trunc_normal_
from timm.models.layers.trace_utils import _assert

from ..meta_arch import MetaArch


class HaloAttention(nn.Module):
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

    def __init__(
            self, dim, dim_out=None, feat_size=None, stride=1, num_heads=8, dim_head=None, block_size=8, halo_size=3,
            qk_ratio=1.0, qkv_bias=False, avg_down=False, scale_pos_embed=False):
        super().__init__()
        dim_out = dim_out or dim
        assert dim_out % num_heads == 0
        assert stride in (1, 2)
        self.num_heads = num_heads
        self.dim_head_qk = dim_head or make_divisible(dim_out * qk_ratio, divisor=8) // num_heads
        self.dim_head_v = dim_out // self.num_heads
        self.dim_out_qk = num_heads * self.dim_head_qk
        self.dim_out_v = num_heads * self.dim_head_v
        self.scale = self.dim_head_qk ** -0.5
        self.scale_pos_embed = scale_pos_embed
        self.block_size = self.block_size_ds = block_size
        self.halo_size = halo_size
        self.win_size = block_size + halo_size * 2  # neighbourhood window size
        self.block_stride = 1
        use_avg_pool = False
        if stride > 1:
            use_avg_pool = avg_down or block_size % stride != 0
            self.block_stride = 1 if use_avg_pool else stride
            self.block_size_ds = self.block_size // self.block_stride

        # FIXME not clear if this stride behaviour is what the paper intended
        # Also, the paper mentions using a 3D conv for dealing with the blocking/gather, and leaving
        # data in unfolded block form. I haven't wrapped my head around how that'd look.
        # self.q = nn.Conv2d(dim, self.dim_out_qk, 1, stride=self.block_stride, bias=qkv_bias)
        # self.kv = nn.Conv2d(dim, self.dim_out_qk + self.dim_out_v, 1, bias=qkv_bias)

        self.q = nn.Linear(dim, self.dim_out_qk, bias=qkv_bias)
        self.kv = nn.Linear(dim, self.dim_out_qk + self.dim_out_v, bias=qkv_bias)

        # generate positional embedding
        self.pre_generate_positional_embedding(q_size=block_size, 
                                               k_size=block_size + 2*halo_size,
                                               halo_size=halo_size,
                                               num_heads=num_heads)


        self.pool = nn.AvgPool2d(2, 2) if use_avg_pool else nn.Identity()

        self.reset_parameters()

    def pre_generate_positional_embedding(self, q_size, k_size, halo_size, num_heads):
        # define a parameter table of relative position bias
        #self.relative_position_bias_table = nn.Parameter(
        #    torch.zeros((2 * self.pos_size[0] - 1) * (2 * self.pos_size[1] - 1),
        #                num_heads))  # 2*(b+h)-1 * 2*(b+h)-1, nH
        pos_size = (q_size + halo_size, q_size + halo_size)
        self.relative_position_bias_table = torch.zeros((2 * pos_size[0] - 1) * (2 * pos_size[1] - 1), num_heads)  # 2*(b+h)-1 * 2*(b+h)-1, nH
        trunc_normal_(self.relative_position_bias_table, std=.02)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(k_size)
        coords_w = torch.arange(k_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, k_size, k_size
        coords_flatten = torch.flatten(coords, 1)  # 2, k_size * k_size
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, k_size * k_size, k_size * k_size
        relative_coords = relative_coords.reshape(2, k_size, k_size, k_size,
                                                  k_size)  # 2, k_size, k_size, k_size, k_size
        relative_coords = relative_coords[:, halo_size:halo_size + q_size, halo_size:halo_size + q_size, :,
                          :]  # 2, q_size, q_size, k_size, k_size
        relative_coords = relative_coords.reshape(2, q_size ** 2, k_size ** 2)  # 2, q_size * q_size, k_size * k_size
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # q_size * q_size, k_size * k_size, 2
        relative_coords[:, :, 0] += pos_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += pos_size[1] - 1
        relative_coords[:, :, 0] *= 2 * pos_size[1] - 1 # ?
        relative_position_index = relative_coords.sum(-1)  # q_size * q_size, k_size * k_size
        #self.register_buffer("relative_position_index", relative_position_index)

        # relative position embedding
        relative_position_bias = self.relative_position_bias_table[relative_position_index.view(-1)].view(
            self.block_size ** 2, (self.block_size + 2 * self.halo_size) ** 2,
            self.num_heads)  # q_size * q_size, k_size * k_size, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, q_size * q_size, k_size * k_size
        relative_position_bias = relative_position_bias.unsqueeze(1).unsqueeze(0)
        self.relative_position_bias = nn.Parameter(relative_position_bias)


    def reset_parameters(self):
        std = self.q.weight.shape[1] ** -0.5  # fan-in
        trunc_normal_(self.q.weight, std=std)
        trunc_normal_(self.kv.weight, std=std)
        # shimin: i remove the initalization for pos embedding
        #trunc_normal_(self.pos_embed.height_rel, std=self.scale)
        #trunc_normal_(self.pos_embed.width_rel, std=self.scale)
    
    #@profile
    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H % self.block_size == 0, '')
        _assert(W % self.block_size == 0, '')
        num_h_blocks = H // self.block_size
        num_w_blocks = W // self.block_size
        num_blocks = num_h_blocks * num_w_blocks

        x = x.permute(0, 2, 3, 1).contiguous()

        q = self.q(x).permute(0, 3, 1, 2)
        # unfold
        q = q.reshape(
            -1, self.dim_head_qk,
            num_h_blocks, self.block_size_ds, num_w_blocks, self.block_size_ds).permute(0, 1, 3, 5, 2, 4)
        # B, num_heads * dim_head * block_size ** 2, num_blocks
        q = q.reshape(B * self.num_heads, self.dim_head_qk, -1, num_blocks).transpose(1, 3)
        # B * num_heads, num_blocks, block_size ** 2, dim_head

        kv = self.kv(x).permute(0, 3, 1, 2)
        # Generate overlapping windows for kv. This approach is good for GPU and CPU. However, unfold() is not
        # lowered for PyTorch XLA so it will be very slow. See code at bottom of file for XLA friendly approach.
        # FIXME figure out how to switch impl between this and conv2d if XLA being used.
        kv = F.pad(kv, [self.halo_size, self.halo_size, self.halo_size, self.halo_size])
        kv = kv.unfold(2, self.win_size, self.block_size).unfold(3, self.win_size, self.block_size).reshape(
            B * self.num_heads, self.dim_head_qk + self.dim_head_v, num_blocks, -1).permute(0, 2, 3, 1)
        
        k, v = torch.split(kv, [self.dim_head_qk, self.dim_head_v], dim=-1)
        # B * num_heads, num_blocks, win_size ** 2, dim_head_qk or dim_head_v
        #k = v = q

        relative_position_bias = self.relative_position_bias.expand(B, -1, num_blocks, -1, -1).flatten(0,1)
        if self.scale_pos_embed:
            attn = (q @ k.transpose(-1, -2) + relative_position_bias) * self.scale
        else:
            attn = (q @ k.transpose(-1, -2)) * self.scale
            attn = attn + relative_position_bias
            #_ = self.pos_embed(q)
            #attn = attn + self.pos_embed(q)
            #attn = (q @ k.transpose(-1, -2)) * self.scale + self.pos_embed(q)
        # B * num_heads, num_blocks, block_size ** 2, win_size ** 2
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 3)  # B * num_heads, dim_head_v, block_size ** 2, num_blocks
        # fold
        out = out.reshape(-1, self.block_size_ds, self.block_size_ds, num_h_blocks, num_w_blocks)
        out = out.permute(0, 3, 1, 4, 2).contiguous().view(
            B, self.dim_out_v, H // self.block_stride, W // self.block_stride)
        # B, dim_out, H // block_stride, W // block_stride
        out = self.pool(out)
        return out


class HaloBlockV2(nn.Module):
    def __init__(self, dim, drop_path, layer_scale_init_value,
                 block_size, halo_size, stage, depth, num_heads,
                 mlp_ratio=4., drop=0., act_layer=nn.GELU, norm_layer=LayerNorm2d,
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        # stride = 2 if stage > 0 and depth == 0 else 1
        stride = 1

        self.shortcut = nn.Sequential(
            nn.Conv2d(dim // 2, dim, kernel_size=1, stride=stride, bias=False),
            norm_layer(dim),
        ) if stride > 1 else nn.Identity()

        self.norm1 = norm_layer(dim // stride)
        self.attn = HaloAttention(dim=dim // stride, dim_out=dim,
                                  num_heads=num_heads[stage], stride=stride,
                                  block_size=block_size, halo_size=halo_size)

        self.gamma_1 = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1, 1)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        self.gamma_2 = nn.Parameter(layer_scale_init_value * torch.ones((1, 1, 1, dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

    #@profile
    def forward(self, x):
        # shape: B, C, H, W
        shortcut = self.shortcut(x)
        x = self.attn(self.norm1(x))

        if self.gamma_1 is not None:
            x = self.gamma_1 * x

        x = shortcut + self.drop_path(x)

        # FFN
        shortcut = x
        x = self.mlp(self.norm2(x).permute(0, 2, 3, 1).contiguous())

        if self.gamma_2 is not None:
            x = self.gamma_2 * x

        x = x.permute(0, 3, 1, 2).contiguous()
        x = shortcut + self.drop_path(x)

        return x


@register_model
def optim_halo_v2_fixed_posembed_tiny(pretrained=False, **kwargs):
    dims = [96 * 2 ** i for i in range(4)]
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]
    block_size = 7
    halo_size = 3

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=HaloBlockV2,
                     block_kwargs=dict(num_heads=num_heads,
                                       block_size=block_size,
                                       halo_size=halo_size),
                     #  downsample_type=nn.Identity,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model
