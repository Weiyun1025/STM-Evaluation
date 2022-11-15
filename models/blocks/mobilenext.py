import torch
from torch import nn
from timm.models.layers import DropPath
from ..meta_arch import LayerNorm2d


class MobileNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path, layer_scale_init_value, mlp_ratio=4., **kwargs):
        super().__init__()
        self.dwconv1 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, int(mlp_ratio * dim))
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.dwconv2 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1, 1)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv1(x)

        # (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm1(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # (N, H, W, C) -> (N, C, H, W)
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.dwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x
        x = shortcut + self.drop_path(x)

        return x


class MobileNeXtV2Block(nn.Module):
    def __init__(self,
                 dim,
                 drop_path,
                 layer_scale_init_value,
                 mixer_ratio=0.5,
                 mlp_ratio=4.,
                 **kwargs):
        super().__init__()
        mid_dim = int(mixer_ratio * dim)
        self.mixer = nn.Sequential(
            LayerNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, mid_dim, kernel_size=1),
            nn.Conv2d(mid_dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim),
        )
        self.gamma_1 = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1, 1)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        mid_dim = int(mlp_ratio * dim)
        self.ffn = nn.Sequential(
            LayerNorm2d(dim),
            nn.Conv2d(dim, mid_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mid_dim, dim, kernel_size=1),
        )
        self.gamma_2 = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1, 1)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.mixer(x)
        if self.gamma_1 is not None:
            x = self.gamma_1 * x
        x = shortcut + self.drop_path(x)

        shortcut = x
        x = self.ffn(x)
        if self.gamma_2 is not None:
            x = self.gamma_2 * x
        x = shortcut + self.drop_path(x)

        return x
