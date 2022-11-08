import math
import torch
import torch.nn.functional as F

from torch import nn
from timm.models.layers import to_2tuple, trunc_normal_
from .cls import ClassBlock
from .blocks.dcn_v3 import DCNv3Block, DCNv3SingleResBlock


class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """

    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


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


class MetaArchV2(nn.Module):
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
                 norm_layer=LayerNorm2d,
                 norm_every_stage=True,
                 act_layer=nn.GELU,
                 deform_points=9,
                 deform_padding=True,
                 cls_type=ClassBlock,
                 cls_kwargs={},
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
        self.cls_attn = nn.ModuleList()
        self.cls = nn.Parameter(torch.randn(1, 1, dims[0]))
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

            self.add_module(f'cls_attn_{i}', cls_type(dim=dim, **cls_kwargs))
            self.add_module(f'cls_proj_{i}', nn.Linear(in_features=dim,
                                                       out_features=dims[i+1] if i+1 < len(depths) else num_classes))

            cur += depths[i]

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
        deform = self.block_type is DCNv3Block or self.block_type is DCNv3SingleResBlock
        if deform:
            deform_inputs = self._deform_inputs(x)

        # shape: (B, C, H, W)
        cls_token = self.cls.repeat(x.shape[0], 1, 1)
        for i in range(4):
            x = self.downsample_layers[i](x)
            if hasattr(self.block_type, 'pre_stage_transform'):
                x = self.block_type.pre_stage_transform(x)
            x = self.stages[i](x if not deform else (x, deform_inputs[i]))
            if hasattr(self.block_type, 'post_stage_transform'):
                x = self.block_type.post_stage_transform(x)
            x = x[0] if deform else x
            x = self.stage_norms[i](x)

            if hasattr(self, f'cls_attn_{i}'):
                x_tokens = x.flatten(2).permute(0, 2, 1).contiguous()
                cls_token = getattr(self, f'cls_attn_{i}')(cls_token, x_tokens)
                cls_token = getattr(self, f'cls_proj_{i}')(cls_token)

        return cls_token.squeeze(1)

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x

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
