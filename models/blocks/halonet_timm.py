import torch
from torch import nn
from timm.models import register_model
from timm.models.layers import DropPath, LayerNorm2d, Mlp, to_2tuple
from timm.models.layers.halo_attn import HaloAttn as HaloAttention
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
        stride = 2 if stage > 0 and depth == 0 else 1

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


# @register_model
# def official_halo_timm_tiny(pretrained=False, **kwargs):
#     dims = [64 * 2 ** i for i in range(4)]
#     depths = [3, 3, 10, 3]
#     num_heads = [8, 8, 8, 8]
#     block_size = 7
#     halo_size = 3

#     model = MetaArch(img_size=224,
#                      depths=depths,
#                      dims=dims,
#                      stem_type=HaloStem,
#                      block_type=HaloBlock,
#                      block_kwargs=dict(num_heads=num_heads, block_size=block_size, halo_size=halo_size),
#                      norm_layer=nn.BatchNorm2d,
#                      downsample_type=nn.Identity,
#                      **kwargs)

#     if pretrained:
#         raise NotImplementedError()

#     return model
