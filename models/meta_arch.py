import torch
from torch import nn
from timm.models.layers import LayerNorm2d, to_2tuple, trunc_normal_

# TODO: params: ape
# TODO: 检查所有norm的位置


class Stem(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, norm_layer, act_layer, ratio=0.5, **kwargs):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.grid_size = (img_size[0] // 4, img_size[1] // 4)

        # input_shape: B x C x H x W
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, int(out_channels * ratio), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            act_layer(),
            nn.Conv2d(int(out_channels * ratio), out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            norm_layer(out_channels)
        )

    def forward(self, x):
        return self.stem(x)


# TODO: ConvNeXtDownsample norm before conv2d
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
            norm_layer(out_channels)
        )

    def forward(self, x):
        return self.reduction(x)


class PredictionHead(nn.Module):
    def __init__(self, in_features, num_classes, norm_layer, act_layer, ratio=1.5, **kwargs):
        super().__init__()

        # input_shape: B x C x H x W
        self.conv_head = nn.Sequential(
            nn.Conv2d(in_features, int(in_features * ratio), 1, 1, 0, bias=False),
            norm_layer(int(in_features * ratio)),
            act_layer()
        )
        self.cls_head = nn.Linear(int(in_features * ratio), num_classes)

    def forward(self, x):
        assert len(x.shape) == 4, 'input shape: BxCxHxW'

        x = self.conv_head(x)
        x = torch.mean(x, dim=(-2, -1))
        x = self.cls_head(x)
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
                 head_type=PredictionHead,
                 head_kwargs={},
                 norm_layer=LayerNorm2d,
                 act_layer=nn.GELU,
                 **kwargs,
                 ):
        super().__init__()

        # stem + downsample_layers
        stem = stem_type(in_channels=in_channels, out_channels=dims[0], img_size=img_size,
                         norm_layer=norm_layer, act_layer=act_layer, **stem_kwargs)
        # H, W
        self.patch_grid = stem.grid_size
        self.downsample_layers = nn.ModuleList([stem])
        for i in range(3):
            downsample_layer = downsample_type(in_channels=dims[i], out_channels=dims[i+1],
                                               norm_layer=norm_layer, **downsample_kwargs)
            self.downsample_layers.append(downsample_layer)

        # blocks
        cur = 0
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        for i, (depth, dim) in enumerate(zip(depths, dims)):
            stage = nn.Sequential(
                *[block_type(dim=dim, drop_path=dp_rates[cur + j], stage=i, depth=j,
                             input_resolution=(self.patch_grid[0] // (2 ** i), self.patch_grid[1] // (2 ** i)),
                             layer_scale_init_value=layer_scale_init_value,
                             **block_kwargs)
                  for j in range(depth)]
            )
            self.stages.append(stage)
            cur += depths[i]
        self.norm = norm_layer(dims[-1])

        if num_classes > 0:
            self.head = head_type(dims[-1], num_classes,
                                  norm_layer=norm_layer,
                                  act_layer=act_layer,
                                  **head_kwargs)
        else:
            self.head = nn.Identity()

        self.apply(self._init_weights)

    @torch.jit.ignore
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
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
        return self.norm(x)

    def forward_head(self, x):
        # (B, C, H, W) -> (B, num_classes)
        return self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
