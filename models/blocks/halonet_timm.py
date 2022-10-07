import math
import torch
import torch.nn.functional as F
from torch import nn
from timm.models import register_model
from timm.models.layers import DropPath, LayerNorm2d, Mlp, to_2tuple, make_divisible
from timm.models.byobnet import _block_registry, num_groups, create_shortcut, LayerFn
from timm.models.layers.halo_attn import PosEmbedRel, trunc_normal_, _assert
from timm.models.helpers import build_model_with_cfg
from timm.models.byobnet import ByoModelCfg, ByoBlockCfg, ByobNet
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
            qk_ratio=1.0, qkv_bias=False, avg_down=False, scale_pos_embed=False, padding_value=0,
            out_proj=False):
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
        self.padding_value = padding_value
        use_avg_pool = False
        if stride > 1:
            use_avg_pool = avg_down or block_size % stride != 0
            self.block_stride = 1 if use_avg_pool else stride
            self.block_size_ds = self.block_size // self.block_stride

        # FIXME not clear if this stride behaviour is what the paper intended
        # Also, the paper mentions using a 3D conv for dealing with the blocking/gather, and leaving
        # data in unfolded block form. I haven't wrapped my head around how that'd look.
        self.q = nn.Conv2d(dim, self.dim_out_qk, 1, stride=self.block_stride, bias=qkv_bias)
        self.kv = nn.Conv2d(dim, self.dim_out_qk + self.dim_out_v, 1, bias=qkv_bias)

        self.pos_embed = PosEmbedRel(
            block_size=self.block_size_ds, win_size=self.win_size, dim_head=self.dim_head_qk, scale=self.scale)

        self.pool = nn.AvgPool2d(2, 2) if use_avg_pool else nn.Identity()
        self.to_out = nn.Conv2d(dim, self.dim_out_v, 1) if out_proj else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        std = self.q.weight.shape[1] ** -0.5  # fan-in
        trunc_normal_(self.q.weight, std=std)
        trunc_normal_(self.kv.weight, std=std)
        trunc_normal_(self.pos_embed.height_rel, std=self.scale)
        trunc_normal_(self.pos_embed.width_rel, std=self.scale)

    @profile
    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H % self.block_size == 0, '')
        _assert(W % self.block_size == 0, '')
        num_h_blocks = H // self.block_size
        num_w_blocks = W // self.block_size
        num_blocks = num_h_blocks * num_w_blocks

        q = self.q(x)
        # unfold
        q = q.reshape(
            -1, self.dim_head_qk,
            num_h_blocks, self.block_size_ds, num_w_blocks, self.block_size_ds).permute(0, 1, 3, 5, 2, 4)
        # B, num_heads * dim_head * block_size ** 2, num_blocks
        q = q.reshape(B * self.num_heads, self.dim_head_qk, -1, num_blocks).transpose(1, 3)
        # B * num_heads, num_blocks, block_size ** 2, dim_head

        kv = self.kv(x)
        # Generate overlapping windows for kv. This approach is good for GPU and CPU. However, unfold() is not
        # lowered for PyTorch XLA so it will be very slow. See code at bottom of file for XLA friendly approach.
        # FIXME figure out how to switch impl between this and conv2d if XLA being used.
        kv = F.pad(kv, [self.halo_size, self.halo_size, self.halo_size, self.halo_size],
                   value=self.padding_value)
        kv = kv.unfold(2, self.win_size, self.block_size).unfold(3, self.win_size, self.block_size).reshape(
            B * self.num_heads, self.dim_head_qk + self.dim_head_v, num_blocks, -1).permute(0, 2, 3, 1)
        k, v = torch.split(kv, [self.dim_head_qk, self.dim_head_v], dim=-1)
        # B * num_heads, num_blocks, win_size ** 2, dim_head_qk or dim_head_v

        if self.scale_pos_embed:
            attn = q @ k.transpose(-1, -2)
            attn = attn + self.pos_embed(q)
            attn = attn * self.scale
        else:
            attn = (q @ k.transpose(-1, -2)) * self.scale
            attn = attn + self.pos_embed(q)
        # B * num_heads, num_blocks, block_size ** 2, win_size ** 2
        if math.isnan(self.padding_value):
            attn = torch.where(torch.isnan(attn), -torch.inf, attn)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 3)  # B * num_heads, dim_head_v, block_size ** 2, num_blocks
        # fold
        out = out.reshape(-1, self.block_size_ds, self.block_size_ds, num_h_blocks, num_w_blocks)
        out = out.permute(0, 3, 1, 4, 2).contiguous().view(
            B, self.dim_out_v, H // self.block_stride, W // self.block_stride)
        # B, dim_out, H // block_stride, W // block_stride
        out = self.to_out(out)
        out = self.pool(out)
        return out


class HaloBlock(nn.Module):
    """ ResNet-like Bottleneck Block - 1x1 - optional kxk - self attn - 1x1
    """

    def __init__(self, dim, drop_path, layer_scale_init_value, stage, depth, num_heads,
                 block_size, halo_size, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 **kwargs):
        super().__init__()
        stride = 2 if stage > 0 and depth == 0 else 1

        self.shortcut = nn.Sequential(
            nn.Conv2d(dim // 2, dim, kernel_size=1, stride=stride, bias=False),
            norm_layer(dim),
        ) if stride > 1 else nn.Identity()

        self.conv1_1x1 = nn.Sequential(
            nn.Conv2d(dim // stride, dim, kernel_size=1, bias=False),
            norm_layer(dim),
            act_layer(),
        )

        self.self_attn = nn.Sequential(
            HaloAttention(dim=dim, dim_out=dim,
                          num_heads=num_heads[stage], stride=stride,
                          block_size=block_size, halo_size=halo_size),
            norm_layer(dim),
            act_layer(),
        )

        self.conv3_1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            norm_layer(dim),
            act_layer(),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.act = act_layer()

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1, 1)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1_1x1(x)
        x = self.self_attn(x)
        x = self.conv3_1x1(x)
        x = self.drop_path(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = shortcut + x
        return self.act(x)


class HaloBlockV2(nn.Module):
    def __init__(self, dim, drop_path, layer_scale_init_value, padding_value,
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
                                  block_size=block_size, halo_size=halo_size,
                                  padding_value=padding_value)

        self.gamma_1 = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1, 1)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        self.gamma_2 = nn.Parameter(layer_scale_init_value * torch.ones((1, 1, 1, dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

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


class HaloStem(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, norm_layer, act_layer, **kwargs):
        super().__init__()
        self. stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=7, stride=2, padding=3, bias=False),
            norm_layer(out_channels),
            act_layer(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        img_size = to_2tuple(img_size)
        self.grid_size = (img_size[0] // 2, img_size[1] // 2)

    def forward(self, x):
        return self.stem(x)


class ModifiedSelfAttnBlock(nn.Module):
    """ ResNet-like Bottleneck Block - 1x1 - optional kxk - self attn - 1x1
    """

    def __init__(
            self, in_chs, out_chs, kernel_size=3, stride=1, dilation=(1, 1), bottle_ratio=(1., 1.), group_size=None,
            downsample='avg', extra_conv=False, linear_out=False, bottle_in=False, post_attn_na=True,
            feat_size=None, layers: LayerFn = None, drop_block=None, drop_path_rate=0.):
        super().__init__()
        assert layers is not None

        if isinstance(bottle_ratio, float):
            bottle_ratio = (bottle_ratio, bottle_ratio)

        mid_chs_1 = make_divisible(out_chs / bottle_ratio[1])
        mid_chs_2 = make_divisible(out_chs / bottle_ratio[1] * bottle_ratio[0])
        groups = num_groups(group_size, mid_chs_1)

        self.shortcut = create_shortcut(
            downsample, in_chs=in_chs, out_chs=out_chs, stride=stride, dilation=dilation,
            apply_act=False, layers=layers)

        self.conv1_1x1 = layers.conv_norm_act(in_chs, mid_chs_1, 1)
        if extra_conv:
            self.conv2_kxk = layers.conv_norm_act(
                mid_chs_1, mid_chs_1, kernel_size, stride=stride, dilation=dilation[0],
                groups=groups, drop_layer=drop_block)
            stride = 1  # striding done via conv if enabled
        else:
            self.conv2_kxk = nn.Identity()
        opt_kwargs = {} if feat_size is None else dict(feat_size=feat_size)

        num_heads = 4 if mid_chs_1 == 64 else 8
        self.self_attn = layers.self_attn(dim=mid_chs_1,
                                          dim_out=mid_chs_2,
                                          dim_head=mid_chs_1 // num_heads,
                                          num_heads=num_heads,
                                          stride=stride, **opt_kwargs)
        self.post_attn = layers.norm_act(mid_chs_2) if post_attn_na else nn.Identity()
        self.conv3_1x1 = layers.conv_norm_act(mid_chs_2, out_chs, 1, apply_act=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.act = nn.Identity() if linear_out else layers.act(inplace=True)

    def init_weights(self, zero_init_last: bool = False):
        if zero_init_last and self.shortcut is not None:
            nn.init.zeros_(self.conv3_1x1.bn.weight)
        if hasattr(self.self_attn, 'reset_parameters'):
            self.self_attn.reset_parameters()

    def forward(self, x):
        shortcut = x
        x = self.conv1_1x1(x)
        x = self.conv2_kxk(x)
        x = self.self_attn(x)
        x = self.post_attn(x)
        x = self.conv3_1x1(x)
        x = self.drop_path(x)
        if self.shortcut is not None:
            x = x + self.shortcut(shortcut)
        return self.act(x)


_block_registry['self_attn'] = ModifiedSelfAttnBlock


def _create_halonet(b, h, rv, rb, l3, df=None, pretrained=False, **kwargs):
    model_cfg = ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='self_attn', d=3, c=64 * rb, s=1, gs=0, br=(rv, rb)),
            ByoBlockCfg(type='self_attn', d=3, c=128 * rb, s=2, gs=0, br=(rv, rb)),
            ByoBlockCfg(type='self_attn', d=l3, c=256 * rb, s=2, gs=0, br=(rv, rb)),
            ByoBlockCfg(type='self_attn', d=3, c=512 * rb, s=2, gs=0, br=(rv, rb)),
        ),
        stem_chs=64,
        stem_type='7x7',
        stem_pool='maxpool',

        num_features=df,
        self_attn_layer='halo',
        self_attn_kwargs=dict(block_size=7, halo_size=h),
    )

    return build_model_with_cfg(ByobNet, 'halonet', pretrained,
                                model_cfg=model_cfg,
                                feature_cfg=dict(flatten_sequential=True),
                                **kwargs)


@register_model
def halonet_h0(pretrained=False, layer_scale_init_value=0., **kwargs):
    return _create_halonet(b=8, h=3, rv=1., rb=0.5, l3=7, **kwargs)


@register_model
def halonet_h1(pretrained=False, layer_scale_init_value=0., **kwargs):
    return _create_halonet(b=8, h=3, rv=1., rb=1., l3=10, **kwargs)


@register_model
def halonet_h2(pretrained=False, layer_scale_init_value=0., **kwargs):
    return _create_halonet(b=8, h=3, rv=1., rb=1.25, l3=11, **kwargs)


@register_model
def halonet_h3(pretrained=False, layer_scale_init_value=0., **kwargs):
    return _create_halonet(b=10, h=3, rv=1., rb=1.5, l3=12, df=1024, **kwargs)


@register_model
def halonet_h4(pretrained=False, layer_scale_init_value=0., **kwargs):
    # return _create_halonet(b=12, h=2, rv=1., rb=3., l3=12, df=1280, **kwargs)
    return _create_halonet(b=12, h=3, rv=1., rb=3., l3=12, df=1280, **kwargs)


@register_model
def halonet_h5(pretrained=False, layer_scale_init_value=0., **kwargs):
    # return _create_halonet(b=14, h=2, rv=2.5, rb=2., l3=23, df=1536, **kwargs)
    return _create_halonet(b=14, h=3, rv=2.5, rb=2., l3=23, df=1536, **kwargs)


@register_model
def halonet_h6(pretrained=False, layer_scale_init_value=0., **kwargs):
    return _create_halonet(b=8, h=4, rv=3., rb=2.75, l3=24, df=1536, **kwargs)


@register_model
def halonet_h7(pretrained=False, layer_scale_init_value=0., **kwargs):
    return _create_halonet(b=10, h=3, rv=4., rb=3.5, l3=26, df=2048, **kwargs)
