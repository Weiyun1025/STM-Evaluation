import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import register_model
from timm.models.layers import to_2tuple
from timm.models.layers import DropPath, Mlp


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """

    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x.permute(0, 2, 3,
                                      1), self.normalized_shape, self.weight,
                            self.bias, self.eps).permute(0, 3, 1, 2)


class Stem(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 img_size,
                 norm_layer,
                 act_layer,
                 ratio=0.5,
                 **kwargs):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.grid_size = (img_size[0] // 4, img_size[1] // 4)

        # input_shape: B x C x H x W
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels,
                      int(out_channels * ratio),
                      kernel_size=(3, 3),
                      stride=(2, 2),
                      padding=(1, 1)), norm_layer(int(out_channels * ratio)),
            act_layer(),
            nn.Conv2d(int(out_channels * ratio),
                      out_channels,
                      kernel_size=(3, 3),
                      stride=(2, 2),
                      padding=(1, 1)), norm_layer(out_channels))

    def forward(self, x):
        return self.stem(x)


class DownsampleLayer(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer, **kwargs):
        super().__init__()

        # input_shape: B x C x H x W
        self.reduction = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=(3, 3),
                      stride=(2, 2),
                      padding=(1, 1),
                      bias=False),
            norm_layer(out_channels),
        )

    def forward(self, x):
        return self.reduction(x)


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

    def __init__(
        self,
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
        stem = stem_type(in_channels=in_channels,
                         out_channels=dims[0],
                         img_size=img_size,
                         norm_layer=norm_layer,
                         norm_first=False,
                         act_layer=act_layer,
                         **stem_kwargs)
        # H, W
        self.patch_grid = stem.grid_size
        self.downsample_layers = nn.ModuleList([stem])
        for i in range(3):
            downsample_layer = downsample_type(
                in_channels=dims[i],
                out_channels=dims[i + 1],
                norm_layer=norm_layer,
                norm_first=True,
                img_size=(self.patch_grid[0] // (2**i),
                          self.patch_grid[1] // (2**i)),
                **downsample_kwargs)
            self.downsample_layers.append(downsample_layer)

        # blocks
        cur = 0
        dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        self.stages = nn.ModuleList()
        self.stage_norms = nn.ModuleList()
        for i, (depth, dim) in enumerate(zip(depths, dims)):
            stage = nn.Sequential(*[
                block_type(dim=dim,
                           drop_path=dp_rates[cur + j],
                           stage=i,
                           depth=j,
                           total_depth=cur + j,
                           input_resolution=(self.patch_grid[0] //
                                             (2**i), self.patch_grid[1] //
                                             (2**i)),
                           layer_scale_init_value=layer_scale_init_value,
                           **block_kwargs) for j in range(depth)
            ])
            self.stages.append(stage)
            self.stage_norms.append(
                nn.LayerNorm((dim, )) if norm_every_stage else nn.Identity())
            cur += depths[i]

        self.conv_head = nn.Sequential(
            nn.Conv2d(dims[-1],
                      int(dims[-1] * extra_transform_ratio),
                      1,
                      1,
                      0,
                      bias=False),
            nn.BatchNorm2d(int(dims[-1] * extra_transform_ratio)),
            act_layer()) if extra_transform else nn.Identity()

        features = int(dims[-1] *
                       extra_transform_ratio) if extra_transform else dims[-1]
        self.avg_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            norm_layer(features)
            if norm_after_avg and not norm_every_stage else nn.Identity(),
            nn.Flatten(1),
        )

        if num_classes > 0:
            self.head = nn.Linear(features, num_classes)
        else:
            self.head = nn.Identity()

    def forward_features(self, x):
        # shape: (B, C, H, W)
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = x.permute(0, 2, 3, 1)
            x = self.stages[i](x)
            x = self.stage_norms[i](x)
            x = x.permute(0, 3, 1, 2)

        x = self.conv_head(x)
        x = self.avg_head(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def rel_logits_1d(q, rel_k, permute_mask):
    """ Compute relative logits along one dimension

    As per: https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2
    Originally from: `Attention Augmented Convolutional Networks` - https://arxiv.org/abs/1904.09925

    Args:
        q: (batch, height, width, dim)
        rel_k: (dim, 2 * window - 1)
        permute_mask: permute output dim according to this
    """
    B, H, W, dim = q.shape
    rel_size = rel_k.shape[1]
    win_size = (rel_size + 1) // 2

    x = q.matmul(rel_k)
    x = x.view(-1, W, rel_size)

    # pad to shift from relative to absolute indexing
    x_pad = F.pad(x, [0, 1]).flatten(1)
    x_pad = F.pad(x_pad, [0, rel_size - W])

    # reshape and slice out the padded elements
    x_pad = x_pad.view(-1, W + 1, rel_size)
    x = x_pad[:, :W, win_size - 1:]

    x = x.view(B, H, 1, W, win_size)
    return x.permute(permute_mask)


class PosEmbedRel(nn.Module):
    """ Relative Position Embedding
    As per: https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2
    Originally from: `Attention Augmented Convolutional Networks` - https://arxiv.org/abs/1904.09925

    """

    def __init__(self, block_size, win_size, dim_head, scale):
        """
        Args:
            block_size (int): block size
            win_size (int): neighbourhood window size
            dim_head (int): attention head dim
            scale (float): scale factor (for init)
        """
        super().__init__()
        self.block_size = block_size
        self.dim_head = dim_head
        self.height_rel = nn.Parameter(
            torch.randn(dim_head, win_size * 2 - 1) * scale)
        self.width_rel = nn.Parameter(
            torch.randn(dim_head, win_size * 2 - 1) * scale)

    def forward(self, q):
        B, num_heads, num_blocks, HW, _ = q.shape

        # relative logits in width dimension.
        q = q.reshape(-1, self.block_size, self.block_size, self.dim_head)
        rel_logits_w = rel_logits_1d(q,
                                     self.width_rel,
                                     permute_mask=(0, 1, 3, 2, 4))

        # relative logits in height dimension.
        q = q.transpose(1, 2)
        rel_logits_h = rel_logits_1d(q,
                                     self.height_rel,
                                     permute_mask=(0, 3, 1, 4, 2))

        rel_logits = rel_logits_h.contiguous() + rel_logits_w.contiguous()
        rel_logits = rel_logits.view(B, num_heads, num_blocks, HW, -1)
        return rel_logits


class SWinPosEmbedRel(nn.Module):

    def __init__(self, block_size, win_size, num_heads) -> None:
        super().__init__()
        self.block_size = block_size
        self.win_size = win_size
        self.num_heads = num_heads
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size - 1) * (2 * win_size - 1), num_heads))
        self.register_buffer(
            "relative_position_index",
            self._get_relative_position_index(win_size, win_size, block_size,
                                              block_size))

    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.block_size**2, self.win_size**2, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0).unsqueeze(2)

    def _get_relative_position_index(self, win_h, win_w, block_h, block_w):
        # get pair-wise relative position index for each token inside the window
        coords = torch.stack(
            torch.meshgrid(
                [torch.arange(win_h), torch.arange(win_w)],
                indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :,
                                         None] - coords_flatten[:,
                                                                None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += win_h - 1  # shift to start from 0
        relative_coords[:, :, 1] += win_w - 1
        relative_coords[:, :, 0] *= 2 * win_w - 1
        relative_coords = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        _sh, _sw = (win_h - block_h) // 2, (win_w - block_w) // 2
        relative_coords = relative_coords.reshape(win_h, win_w, -1)
        relative_coords = relative_coords[_sh:_sh + block_h, _sw:_sw + block_w]
        relative_coords = relative_coords.reshape(block_h * block_w,
                                                  win_h * win_w)
        return relative_coords.contiguous()

    def forward(self, ):
        return self._get_rel_pos_bias()


class HaloAttn(nn.Module):
    """ Halo Attention

    Paper: `Scaling Local Self-Attention for Parameter Efficient Visual Backbones`
        - https://arxiv.org/abs/2103.12731

    The internal dimensions of the attention module are controlled by the interaction of several arguments.
      * the output dimension of the module is specified by dim_out, which falls back to input dim if not set
      * the value (v) dimension is set to dim_out // num_heads, the v projection determines the output dim
      * the query and key (qk) dimensions are determined by
        * num_heads * dim_head if dim_head is not None
        * num_heads * (dim_out * attn_ratio // num_heads) if dim_head is None
      * as seen above, attn_ratio determines the ratio of q and k relative to the output if dim_head not used

    Args:
        dim (int): input dimension to the module
        dim_out (int): output dimension of the module, same as dim if not set
        feat_size (Tuple[int, int]): size of input feature_map (not used, for arg compat with bottle/lambda)
        stride: output stride of the module, query downscaled if > 1 (default: 1).
        num_heads: parallel attention heads (default: 8).
        dim_head: dimension of query and key heads, calculated from dim_out * attn_ratio // num_heads if not set
        block_size (int): size of blocks. (default: 8)
        halo_size (int): size of halo overlap. (default: 3)
        qk_ratio (float): ratio of q and k dimensions to output dimension when dim_head not set. (default: 1.0)
        qkv_bias (bool) : add bias to q, k, and v projections
        avg_down (bool): use average pool downsample instead of strided query blocks
        scale_pos_embed (bool): scale the position embedding as well as Q @ K
    """

    def __init__(self,
                 dim,
                 dim_out=None,
                 feat_size=None,
                 stride=1,
                 num_heads=8,
                 dim_head=None,
                 block_size=8,
                 halo_size=3,
                 qk_ratio=1.0,
                 qkv_bias=False,
                 avg_down=False,
                 scale_pos_embed=False):
        super().__init__()
        dim_out = dim_out or dim
        assert dim_out % num_heads == 0
        assert stride in (1, 2)
        self.num_heads = num_heads
        self.dim_head_qk = dim_head or make_divisible(dim_out * qk_ratio,
                                                      divisor=8) // num_heads
        self.dim_head_v = dim_out // self.num_heads
        self.dim_out_qk = num_heads * self.dim_head_qk
        self.dim_out_v = num_heads * self.dim_head_v
        self.scale = self.dim_head_qk**-0.5
        self.scale_pos_embed = scale_pos_embed
        self.block_size = self.block_size_ds = block_size
        self.halo_size = halo_size
        self.win_size = block_size + halo_size * 2  # neighbourhood window size
        self.block_stride = 1
        use_avg_pool = False

        # FIXME not clear if this stride behaviour is what the paper intended
        # Also, the paper mentions using a 3D conv for dealing with the blocking/gather, and leaving
        # data in unfolded block form. I haven't wrapped my head around how that'd look.
        self.kv = nn.Conv2d(dim,
                            self.dim_out_qk + self.dim_out_v,
                            1,
                            bias=qkv_bias)
        self.q = nn.Linear(dim, self.dim_out_qk, bias=qkv_bias)

        # self.pos_embed = PosEmbedRel(block_size=self.block_size_ds,
        #                              win_size=self.win_size,
        #                              dim_head=self.dim_head_qk,
        #                              scale=self.scale)
        self.swin_pos_embed = SWinPosEmbedRel(block_size=self.block_size_ds,
                                              win_size=self.win_size,
                                              num_heads=num_heads)

        self.pool = nn.AvgPool2d(2, 2) if use_avg_pool else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, H, W, C = x.shape
        assert H % self.block_size == 0
        assert W % self.block_size == 0
        num_h_blocks = H // self.block_size
        num_w_blocks = W // self.block_size
        num_blocks = num_h_blocks * num_w_blocks

        q = self.q(x)
        # unfold
        q = q.reshape(-1, num_h_blocks, self.block_size_ds, num_w_blocks,
                      self.block_size_ds, self.num_heads,
                      self.dim_head_qk).permute(0, 5, 1, 3, 2, 4, 6)
        # B, num_heads, num_h_blocks, num_w_blocks, block_size_ds, block_size_ds, dim_head_qk
        q = q.reshape(-1, self.num_heads, num_blocks, self.block_size**2,
                      self.dim_head_qk)
        # B, num_heads, num_blocks, block_size ** 2, dim_head

        kv = self.kv(x.permute(0, 3, 1, 2))
        kv = F.pad(
            kv,
            [
                self.halo_size,
                self.halo_size,
                self.halo_size,
                self.halo_size,
            ],
            mode='constant',
            value=-torch.inf,
        )
        kv = kv.unfold(2, self.win_size, self.block_size).unfold(
            3, self.win_size,
            self.block_size).reshape(-1, self.num_heads,
                                     self.dim_head_qk + self.dim_head_v,
                                     num_blocks,
                                     self.win_size**2).permute(0, 1, 3, 4, 2)
        k, v = torch.split(kv, [self.dim_head_qk, self.dim_head_v], dim=-1)
        k = k.reshape(-1, self.num_heads, num_blocks, self.win_size,
                      self.win_size, self.dim_head_qk)
        v = v.reshape(-1, self.num_heads, num_blocks, self.win_size,
                      self.win_size, self.dim_head_v)
        # _L = (self.block_size + 0)
        # _S = (self.win_size - _L) // 2
        # k = k[:, :, :, _S:_S + _L, _S:_S + _L, :].flatten(3, 4)
        # v = v[:, :, :, _S:_S + _L, _S:_S + _L, :].flatten(3, 4)
        k = k.flatten(3, 4)
        v = v.flatten(3, 4)

        if self.scale_pos_embed:
            # attn = (q @ k.transpose(-1, -2) + self.pos_embed(q)) * self.scale
            attn = (q @ k.transpose(-1, -2)) * self.scale
        else:
            # attn = (q @ k.transpose(-1, -2)) * self.scale + self.pos_embed(q)
            attn = (q * self.scale) @ k.transpose(-1, -2)
            # pos = self.swin_pos_embed()
            # attn = pos + attn

        # B, num_heads, num_blocks, block_size ** 2, win_size ** 2
        attn = attn.softmax(dim=-1)

        out = attn @ v
        # B, num_heads, num_blocks, block_size ** 2, dim_head_v
        # fold
        out = out.reshape(-1, self.num_heads, num_h_blocks, num_w_blocks,
                          self.block_size_ds, self.block_size_ds,
                          self.dim_head_qk)
        out = out.permute(0, 2, 4, 3, 5, 1, 6).reshape(B, H, W, self.dim_out_v)
        out = self.proj(out)
        # B, H, W, dim_out
        return out


class HaloBlockV2(nn.Module):

    def __init__(self,
                 dim,
                 drop_path,
                 layer_scale_init_value,
                 block_size,
                 halo_size,
                 stage,
                 num_heads,
                 mlp_ratio=4.,
                 drop=0.,
                 act_layer=nn.GELU,
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm((dim, ))
        self.attn = HaloAttn(dim=dim,
                             dim_out=dim,
                             num_heads=num_heads[stage],
                             block_size=block_size,
                             halo_size=halo_size)

        self.gamma_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((1, 1, 1, dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm((dim, ))
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer,
                       drop=drop)

        self.gamma_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((1, 1, 1, dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        # shape: (B, H, W, C)
        shortcut = x
        x = self.attn(self.norm1(x))

        if self.gamma_1 is not None:
            x = self.gamma_1 * x
        x = shortcut + self.drop_path(x)

        # FFN
        shortcut = x
        x = self.mlp(self.norm2(x))

        if self.gamma_2 is not None:
            x = self.gamma_2 * x
        x = shortcut + self.drop_path(x)

        return x


@register_model
def conv_halo_opt_base(pretrained=False, **kwargs):
    dims = [128 * 2**i for i in range(4)]
    depths = [2, 2, 18, 2]
    num_heads = [4, 8, 16, 32]
    block_size = 7
    halo_size = 2

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=HaloBlockV2,
                     block_kwargs=dict(num_heads=num_heads,
                                       block_size=block_size,
                                       halo_size=halo_size),
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def halo_opt_stage(pretrained=False, **kwargs):
    dims = [128 * 2**i for i in range(4)]
    depths = [2, 2, 18, 2]
    num_heads = [4, 8, 16, 32]
    block_size = 7
    halo_size = 0
    model = torch.nn.Sequential(*[
        HaloBlockV2(512, 0, 1e-6, block_size, halo_size, 2, num_heads)
        for idx in range(18)
    ])

    if pretrained:
        raise NotImplementedError()

    return model
