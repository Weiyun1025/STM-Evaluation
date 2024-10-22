from typing import Tuple

import torch
from torch import nn
from timm.models.layers import DropPath, Mlp, to_2tuple, _assert
from timm.models.swin_transformer_v2 import WindowAttention, window_partition, window_reverse


class SwinV2Block(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pretraining.
    """

    def __init__(
            self, dim, drop_path, layer_scale_init_value,
            input_resolution, stage, depth, num_heads, window_size,
            mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0,
            **kwargs):
        super().__init__()
        num_heads = num_heads[stage]
        shift_size = 0 if (depth % 2 == 0) else window_size // 2

        self.dim = dim
        self.input_resolution = to_2tuple(input_resolution)
        self.num_heads = num_heads
        ws, ss = self._calc_window_shift(window_size, shift_size)
        self.window_size: Tuple[int, int] = ws
        self.shift_size: Tuple[int, int] = ss
        self.window_area = self.window_size[0] * self.window_size[1]
        self.mlp_ratio = mlp_ratio

        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.gamma_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.gamma_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        if any(self.shift_size):
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            cnt = 0
            for h in (
                    slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None)):
                for w in (
                        slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_area)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def _calc_window_shift(self, target_window_size, target_shift_size) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        target_window_size = to_2tuple(target_window_size)
        target_shift_size = to_2tuple(target_shift_size)
        window_size = [r if r <= w else w for r, w in zip(self.input_resolution, target_window_size)]
        shift_size = [0 if r <= w else s for r, w, s in zip(self.input_resolution, window_size, target_shift_size)]
        return tuple(window_size), tuple(shift_size)

    def _attn(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        _assert(L == H * W, "input feature has wrong size")
        x = x.view(B, H, W, C)

        # cyclic shift
        has_shift = any(self.shift_size)
        if has_shift:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, self.input_resolution)  # B H' W' C

        # reverse cyclic shift
        if has_shift:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1).contiguous()

        shortcut = x
        x = self.norm1(self._attn(x))
        if self.gamma_1 is not None:
            x = self.gamma_1 * x
        x = shortcut + self.drop_path1(x)

        shortcut = x
        x = self.norm2(self.mlp(x))
        if self.gamma_2 is not None:
            x = self.gamma_2 * x
        x = shortcut + self.drop_path2(x)

        return x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
