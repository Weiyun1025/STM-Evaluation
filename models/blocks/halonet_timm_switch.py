import torch
from torch import nn
from timm.models import register_model
from timm.models.layers import DropPath, LayerNorm2d, Mlp, to_2tuple, make_divisible
from timm.models.byobnet import _block_registry, num_groups, create_shortcut, LayerFn
from timm.models.layers.halo_attn import HaloAttn as HaloAttention
from timm.models.helpers import build_model_with_cfg
from timm.models.byobnet import ByoModelCfg, ByoBlockCfg, ByobNet
from ..meta_arch import MetaArch


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
        halo_size = 0 if depth % 2 == 0 else halo_size

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
        x = self.mlp(self.norm2(x).permute(0, 2, 3, 1).contiguous())

        if self.gamma_2 is not None:
            x = self.gamma_2 * x

        x = x.permute(0, 3, 1, 2).contiguous()
        x = shortcut + self.drop_path(x)

        return x
