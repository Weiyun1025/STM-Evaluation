import torch
from torch import nn
from timm.models.layers import DropPath, Mlp
from ..meta_arch import LayerNorm2d


class PoolformerBlock(nn.Module):
    def __init__(self, dim, drop_path, layer_scale_init_value, pool_size=3, **kwargs):
        super().__init__()
        self.dw_norm = LayerNorm2d(dim, eps=1e-6)
        self.dwconv = nn.AvgPool2d(kernel_size=pool_size,
                                   stride=1,
                                   padding=pool_size//2,
                                   count_include_pad=False)

        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        self.gamma_1 = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1, 1)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        # pointwise/1x1 convs, implemented with linear layers
        self.pw_norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma_2 = nn.Parameter(layer_scale_init_value * torch.ones((1, 1, 1, dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(self.dw_norm(x))
        if self.gamma_1 is not None:
            x = self.gamma_1 * x
        x = shortcut + self.drop_path(x)

        # (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1).contiguous()

        shortcut = x
        x = self.pw_norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma_2 is not None:
            x = self.gamma_2 * x
        x = shortcut + self.drop_path(x)

        # (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x


class PoolformerV2Block(nn.Module):
    def __init__(self, dim, drop_path, layer_scale_init_value, pool_size=3, **kwargs):
        super().__init__()
        self.dim = dim
        mlp_ratio = 4
        self.mlp_ratio = mlp_ratio

        self.norm1 = LayerNorm2d(dim)
        self.pool = nn.AvgPool2d(kernel_size=pool_size,
                                 stride=1,
                                 padding=pool_size//2,
                                 count_include_pad=False)

        self.gamma_1 = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1, 1)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = LayerNorm2d(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=nn.GELU)

        self.gamma_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.pool(x)
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
