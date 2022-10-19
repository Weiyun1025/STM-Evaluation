import math
import torch
from torch import nn
from timm.models.layers import LayerNorm2d, to_2tuple, trunc_normal_
from .blocks.dcn_v3 import DCNv3Block

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
                 norm_every_stage=True,
                 norm_after_avg=False,
                 act_layer=nn.GELU,
                 deform_points=9,
                 deform_padding=True,
                 **kwargs,
                 ):
        super().__init__()
        self.depths = depths
        self.block_type = block_type
        self.deform_points = deform_points
        self.deform_padding = deform_padding

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
                *[block_type(dim=dim, drop_path=dp_rates[cur + j], stage=i, depth=j, total_depth=cur+j,
                             input_resolution=(self.patch_grid[0] // (2 ** i), self.patch_grid[1] // (2 ** i)),
                             layer_scale_init_value=layer_scale_init_value,
                             **block_kwargs)
                  for j in range(depth)]
            )
            self.stages.append(stage)
            self.stage_norms.append(norm_layer(dim) if norm_every_stage else nn.Identity())
            cur += depths[i]
        self.stage_end_norm = nn.Identity() if norm_every_stage or norm_after_avg else norm_layer(dims[-1])

        self.conv_head = nn.Sequential(
            nn.Conv2d(dims[-1], int(dims[-1] * extra_transform_ratio), 1, 1, 0, bias=False),
            nn.BatchNorm2d(int(dims[-1] * extra_transform_ratio)),
            act_layer()
        ) if extra_transform else nn.Identity()

        features = int(dims[-1] * extra_transform_ratio) if extra_transform else dims[-1]
        self.avg_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            norm_layer(features) if norm_after_avg else nn.Identity(),
            nn.Flatten(1),
        )

        if num_classes > 0:
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
        deform = self.block_type is DCNv3Block
        stage_outputs = [] # store intermediate feature maps for ERF analysis
        if deform:
            deform_inputs = self._deform_inputs(x)

        # shape: (B, C, H, W)
        for i in range(4):
            x = self.downsample_layers[i](x)
            if hasattr(self.block_type, 'pre_stage_transform'):
                x = self.block_type.pre_stage_transform(x)
            x = self.stages[i](x if not deform else (x, deform_inputs[i]))
            if hasattr(self.block_type, 'post_stage_transform'):
                x = self.block_type.post_stage_transform(x)
            x = x[0] if deform else x
            x = self.stage_norms[i](x)
            stage_outputs.append(x)

        x = self.stage_end_norm(x)

        x = self.conv_head(x)
        x = self.avg_head(x)
        return x, stage_outputs

    def forward(self, x):
        x, stage_output = self.forward_features(x)
        x = self.head(x)
        return x, stage_output

    def target_layers(self):
        return [self.stages[-1][-1]]

    def _deform_inputs(self, x):
        b, c, h, w = x.shape
        deform_inputs = []
        if self.deform_padding:
            padding = int(math.sqrt(self.deform_points) // 2)
        else:
            padding = int(0)

        # for i in range(sum(self.depths)):
        for i in range(len(self.depths)):
            spatial_shapes = torch.as_tensor(
                [(h // pow(2, i + 2) + 2 * padding,
                    w // pow(2, i + 2) + 2 * padding)],
                dtype=torch.long, device=x.device)
            level_start_index = torch.cat(
                (spatial_shapes.new_zeros((1,)),
                    spatial_shapes.prod(1).cumsum(0)[:-1]))
            reference_points = self._get_reference_points(
                [(h // pow(2, i + 2) + 2 * padding,
                    w // pow(2, i + 2) + 2 * padding)],
                device=x.device, padding=padding)
            deform_inputs.append(
                [reference_points, spatial_shapes, level_start_index,
                    (h // pow(2, i + 2), w // pow(2, i + 2))])

        return deform_inputs

    def _get_reference_points(self, spatial_shapes, device, padding=0):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(padding + 0.5, H_ - padding - 0.5,
                               int(H_ - 2 * padding),
                               dtype=torch.float32, device=device),
                torch.linspace(padding + 0.5, W_ - padding - 0.5,
                               int(W_ - 2 * padding),
                               dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / H_
            ref_x = ref_x.reshape(-1)[None] / W_
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None]

        return reference_points



