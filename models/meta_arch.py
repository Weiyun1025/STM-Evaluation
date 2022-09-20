import torch
from torch import nn
from timm.models.layers import LayerNorm2d, to_2tuple, trunc_normal_

# TODO: 检查所有norm的位置


class Stem(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, norm_layer, act_layer, ratio=0.5, **kwargs):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.grid_size = (img_size[0] // 4, img_size[1] // 4)

        # input_shape: B x C x H x W
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, int(out_channels * ratio),
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            norm_layer(int(out_channels * ratio)),
            act_layer(),
            nn.Conv2d(int(out_channels * ratio), out_channels,
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            norm_layer(out_channels)
        )

    def forward(self, x):
        return self.stem(x)


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, **kwargs):
        super().__init__()

        # input_shape: B x C x H x W
        self.reduction = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=(3, 3),
                      stride=(2, 2),
                      padding=(1, 1),
                      bias=False),
            norm_layer(out_channels),
        )

    def forward(self, x):
        return self.reduction(x)


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, in_channels, out_channels, img_size, patch_size, norm_layer, norm_first, **kwargs):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.pre_norm = norm_layer(in_channels) if norm_first and norm_layer else nn.Identity()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size)
        self.post_norm = norm_layer(out_channels) if not norm_first and norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."

        x = self.pre_norm(x)
        x = self.proj(x)
        x = self.post_norm(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.dim = in_channels
        self.out_dim = out_channels or 2 * in_channels
        self.norm = nn.LayerNorm(4 * in_channels)
        self.reduction = nn.Linear(4 * in_channels, self.out_dim, bias=False)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        # x = x.view(B, H, W, C)
        x = x.permute(0, 2, 3, 1)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        x = x.permute(0, 3, 1, 2)
        return x


class MetaArch(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self,
                 img_size=224,
                 in_channels=3,
                 num_classes=1000,
                 depths=(3, 3, 9, 3),
                 dims=(96, 192, 384, 768),
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 stem_type=Stem,
                 stem_kwargs={},
                 block_type=None,
                 block_kwargs={},
                 downsample_type=DownsampleLayer,
                 downsample_kwargs={},
                 extra_transform=True,
                 extra_transform_ratio=1.5,
                 norm_layer=LayerNorm2d,
                 act_layer=nn.GELU,
                 **kwargs,
                 ):
        super().__init__()

        # stem + downsample_layers
        stem = stem_type(in_channels=in_channels, out_channels=dims[0], img_size=img_size,
                         norm_layer=norm_layer, norm_first=False, act_layer=act_layer, **stem_kwargs)
        # H, W
        self.patch_grid = stem.grid_size
        self.downsample_layers = nn.ModuleList([stem])
        for i in range(3):
            downsample_layer = downsample_type(in_channels=dims[i], out_channels=dims[i+1],
                                               norm_layer=norm_layer, norm_first=True,
                                               img_size=(self.patch_grid[0] // (2 ** i), self.patch_grid[1] // (2 ** i)),
                                               **downsample_kwargs)
            self.downsample_layers.append(downsample_layer)

        # blocks
        cur = 0
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        self.stage_norms = nn.ModuleList()
        for i, (depth, dim) in enumerate(zip(depths, dims)):
            stage = nn.Sequential(
                *[block_type(dim=dim, drop_path=dp_rates[cur + j], stage=i, depth=j,
                             input_resolution=(self.patch_grid[0] // (2 ** i), self.patch_grid[1] // (2 ** i)),
                             layer_scale_init_value=layer_scale_init_value,
                             **block_kwargs)
                  for j in range(depth)]
            )
            self.stages.append(stage)
            self.stage_norms.append(norm_layer(dim))
            cur += depths[i]

        self.conv_head = nn.Sequential(
            nn.Conv2d(dims[-1], int(dims[-1] * extra_transform_ratio), 1, 1, 0, bias=False),
            nn.BatchNorm2d(int(dims[-1] * extra_transform_ratio)),
            act_layer()
        ) if extra_transform else nn.Identity()

        self.avg_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # norm_layer(mid_features),
            nn.Flatten(1),
        )

        if num_classes > 0:
            features = int(dims[-1] * extra_transform_ratio) if extra_transform else dims[-1]
            self.head = nn.Linear(features, num_classes)
        else:
            self.head = nn.Identity()

        self.apply(self._init_weights)

    @ torch.jit.ignore
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @ torch.jit.ignore
    def no_weight_decay(self):
        # from swin v1
        no_weight_decay = {'absolute_pos_embed'}
        for name, _ in self.named_parameters():
            if 'relative_position_bias_table' in name:
                no_weight_decay.add(name)

        return no_weight_decay

    def forward_features(self, x):
        # shape: (B, C, H, W)
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            x = self.stage_norms[i](x)

        x = self.conv_head(x)
        x = self.avg_head(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
