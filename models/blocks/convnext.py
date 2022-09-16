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
        x_input = x
        x = self.dwconv(x)
        # (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        # (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)

        x = x_input + self.drop_path(x)
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
        # difference with official implementation: norm before mean
        return self.head(torch.mean(x, dim=(-2, -1)))


@ register_model
def convnext_tiny(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[3, 3, 9, 3],
                     dims=[96, 192, 384, 768],
                     stem_type=ConvNeXtStem,
                     block_type=ConvNeXtBlock,
                     downsample_type=ConvNeXtDownsampleLayer,
                     head_type=ConvNeXtHead,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model
