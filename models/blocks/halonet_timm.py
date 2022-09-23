import torch
from torch import nn
from timm.models import register_model
from timm.models.layers import DropPath, LayerNorm2d, Mlp, to_2tuple, make_divisible
from timm.models.byobnet import _block_registry, num_groups, create_shortcut, LayerFn
from timm.models.layers.halo_attn import HaloAttn as HaloAttention
from timm.models.helpers import build_model_with_cfg
from timm.models.byobnet import ByoModelCfg, ByoBlockCfg, ByobNet
from ..meta_arch import MetaArch


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

    def forward(self, x):
        # shape: B, C, H, W
        shortcut = self.shortcut(x)
        x = self.attn(self.norm1(x))

        if self.gamma_1 is not None:
            x = self.gamma_1 * x

        x = shortcut + self.drop_path(x)

        # FFN
        shortcut = x
        x = self.mlp(self.norm2(x).permute(0, 2, 3, 1))

        if self.gamma_2 is not None:
            x = self.gamma_2 * x

        x = x.permute(0, 3, 1, 2)
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
    return _create_halonet(b=12, h=2, rv=1., rb=3., l3=12, df=1280, **kwargs)


@register_model
def halonet_h5(pretrained=False, layer_scale_init_value=0., **kwargs):
    return _create_halonet(b=14, h=2, rv=2.5, rb=2., l3=23, df=1536, **kwargs)


@register_model
def halonet_h6(pretrained=False, layer_scale_init_value=0., **kwargs):
    return _create_halonet(b=8, h=4, rv=3., rb=2.75, l3=24, df=1536, **kwargs)


@register_model
def halonet_h7(pretrained=False, layer_scale_init_value=0., **kwargs):
    return _create_halonet(b=10, h=3, rv=4., rb=3.5, l3=26, df=2048, **kwargs)
