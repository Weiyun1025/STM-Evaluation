import torch
from torch import nn
from timm.models import register_model
from timm.models.layers import DropPath, LayerNorm2d, to_2tuple
from ..meta_arch import MetaArch


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path, layer_scale_init_value, **kwargs):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x.permute(0, 2, 3, 1)
        x = self.dwconv(x)
        # (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = shortcut + self.drop_path(x)

        # (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)

        return x


class ConvNeXtV2Block(nn.Module):
    # double res
    def __init__(self, dim, drop_path, layer_scale_init_value, **kwargs):
        super().__init__()
        self.dw_norm = LayerNorm2d(dim, eps=1e-6)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv

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
        x = x.permute(0, 2, 3, 1)

        shortcut = x
        x = self.pw_norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma_2 is not None:
            x = self.gamma_2 * x
        x = shortcut + self.drop_path(x)

        # (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)

        return x


class ConvNeXtV3Block(nn.Module):
    # double res + in/out proj
    def __init__(self, dim, drop_path, layer_scale_init_value, **kwargs):
        super().__init__()
        self.dw_norm = LayerNorm2d(dim, eps=1e-6)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        # add an output proj for dw conv in V3
        self.dw_input_proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.dw_out_proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)

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
        x = self.dwconv(self.dw_input_proj(self.dw_norm(x)))
        x = self.dw_out_proj(x)
        if self.gamma_1 is not None:
            x = self.gamma_1 * x
        x = shortcut + self.drop_path(x)

        # (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)

        shortcut = x
        x = self.pw_norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma_2 is not None:
            x = self.gamma_2 * x
        x = shortcut + self.drop_path(x)

        # (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)

        return x


class ConvNeXtV4Block(nn.Module):
    # double res + out proj
    def __init__(self, dim, drop_path, layer_scale_init_value, **kwargs):
        super().__init__()
        self.dw_norm = LayerNorm2d(dim, eps=1e-6)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        # add an output proj for dw conv in V3
        self.dw_out_proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)

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
        x = self.dw_out_proj(x)
        if self.gamma_1 is not None:
            x = self.gamma_1 * x
        x = shortcut + self.drop_path(x)

        # (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)

        shortcut = x
        x = self.pw_norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma_2 is not None:
            x = self.gamma_2 * x
        x = shortcut + self.drop_path(x)

        # (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)

        return x


class ConvNeXtV3SingleResBlock(nn.Module):
    # single res + in/out proj
    def __init__(self, dim, drop_path, layer_scale_init_value, **kwargs):
        super().__init__()
        self.dw_norm = LayerNorm2d(dim, eps=1e-6)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        # add an output proj for dw conv in V3
        self.dw_input_proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.dw_out_proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)

        # pointwise/1x1 convs, implemented with linear layers
        self.pw_norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((1, 1, 1, dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(self.dw_input_proj(self.dw_norm(x)))
        x = self.dw_out_proj(x)

        # (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.pw_norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        # (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(x)
        return x


class ConvNeXtV5Block(nn.Module):
    # double res + in/out proj
    def __init__(self, dim, drop_path, layer_scale_init_value, bias=False, **kwargs):
        super().__init__()
        self.dw_norm = LayerNorm2d(dim, eps=1e-6)
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=7, padding=3, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        self.gamma_1 = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1, 1)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        # pointwise/1x1 convs, implemented with linear layers
        self.pw_norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

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
        x = x.permute(0, 2, 3, 1)

        shortcut = x
        x = self.pwconv(self.pw_norm(x))
        if self.gamma_2 is not None:
            x = self.gamma_2 * x
        x = shortcut + self.drop_path(x)

        # (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return x


class ConvNeXtV6Block(nn.Module):
    # double res + in/out proj
    def __init__(self, dim, drop_path, layer_scale_init_value, **kwargs):
        super().__init__()
        self.pre_pwnorm = LayerNorm2d(dim, eps=1e-6)
        self.pre_pwconv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)

        self.gamma_1 = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1, 1)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.dwnorm = LayerNorm2d(dim, eps=1e-6)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        # pointwise/1x1 convs, implemented with linear layers
        self.post_pwnorm = nn.LayerNorm(dim, eps=1e-6)
        self.post_pwconv = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

        self.gamma_2 = nn.Parameter(layer_scale_init_value * torch.ones((1, 1, 1, dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.pre_pwconv(self.pre_pwnorm(x))
        if self.gamma_1 is not None:
            x = self.gamma_1 * x
        x = shortcut + self.drop_path(x)

        x = self.dwconv(self.dwnorm(x))

        # (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)

        shortcut = x
        x = self.post_pwconv(self.post_pwnorm(x))
        if self.gamma_2 is not None:
            x = self.gamma_2 * x
        x = shortcut + self.drop_path(x)

        # (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return x


class ConvNeXtStem(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, **kwargs):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=4),
            LayerNorm2d(out_channels, eps=1e-6)
        )
        self.grid_size = (img_size[0] // 4, img_size[1] // 4)

    def forward(self, x):
        return self.stem(x)


class ConvNeXtDownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        self.reduction = nn.Sequential(
            LayerNorm2d(in_channels, eps=1e-6),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.reduction(x)


class ConvNeXtHead(nn.Module):
    def __init__(self, in_features, num_classes, **kwargs):
        super().__init__()
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.head(torch.mean(x, dim=(-2, -1)))


@ register_model
def official_convnext_tiny(pretrained=False, **kwargs):
    # difference with official implementation: norm before mean, leads to 0.4% dropping
    model = MetaArch(img_size=224,
                     depths=[3, 3, 9, 3],
                     dims=[96, 192, 384, 768],
                     stem_type=ConvNeXtStem,
                     block_type=ConvNeXtBlock,
                     downsample_type=ConvNeXtDownsampleLayer,
                     extra_transform=False,
                     norm_every_stage=False,
                     norm_after_avg=True,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model
