import torch
import torch.nn.functional as F
from torch import nn
from timm.models.layers import DropPath, Mlp, make_divisible
from timm.models.layers.halo_attn import PosEmbedRel, trunc_normal_, _assert


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        if self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

        raise NotImplementedError(self.data_format)


class HaloAttn(nn.Module):
    """ Improvements:
    1. attn mask
    2. out proj
    3. query scale
    4. query-related position
    5. layout optim

    TODO: mask Tensor.masked_fill_ or Addidion ?
    TODO: layout optim
    TODO: query-free pos embed
    TODO: query-addition pos embed

    Args:
        nn (_type_): _description_
    """

    def __init__(
            self,
            dim,
            dim_out=None,
            num_heads=8,
            dim_head=None,
            block_size=8,
            halo_size=3,
            qk_ratio=1.0,
            qkv_bias=False,
            scale_pos_embed=False):
        super().__init__()
        dim_out = dim_out or dim
        assert dim_out % num_heads == 0
        self.num_heads = num_heads
        self.dim_head_qk = dim_head or make_divisible(dim_out * qk_ratio, divisor=8) // num_heads
        self.dim_head_v = dim_out // self.num_heads
        self.dim_out_qk = num_heads * self.dim_head_qk
        self.dim_out_v = num_heads * self.dim_head_v
        self.scale = self.dim_head_qk ** -0.5
        self.scale_pos_embed = scale_pos_embed
        self.block_size = self.block_size_ds = block_size
        self.halo_size = halo_size
        self.win_size = block_size + halo_size * 2  # neighbourhood window size
        self.block_stride = 1

        # FIXME not clear if this stride behaviour is what the paper intended
        # Also, the paper mentions using a 3D conv for dealing with the blocking/gather, and leaving
        # data in unfolded block form. I haven't wrapped my head around how that'd look.
        self.to_q = nn.Conv2d(dim, self.dim_out_qk, 1, stride=self.block_stride, bias=qkv_bias)
        self.to_kv = nn.Conv2d(dim, self.dim_out_qk + self.dim_out_v, 1, bias=qkv_bias)

        self.pos_embed = PosEmbedRel(block_size=self.block_size_ds,
                                     win_size=self.win_size,
                                     dim_head=self.dim_head_qk,
                                     scale=self.scale)

        self.to_out = nn.Conv2d(self.dim_out_v, self.dim_out_v, 1)
        self.reset_parameters()

        self.H, self.W = None, None
        self.mask = None

    def reset_parameters(self):
        std = self.to_q.weight.shape[1] ** -0.5  # fan-in
        trunc_normal_(self.to_q.weight, std=std)
        trunc_normal_(self.to_kv.weight, std=std)
        trunc_normal_(self.pos_embed.height_rel, std=self.scale)
        trunc_normal_(self.pos_embed.width_rel, std=self.scale)

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H % self.block_size == 0, '')
        _assert(W % self.block_size == 0, '')
        num_h_blocks = H // self.block_size
        num_w_blocks = W // self.block_size
        num_blocks = num_h_blocks * num_w_blocks

        q = self.to_q(x)
        # unfold
        q = q.reshape(
            -1, self.dim_head_qk,
            num_h_blocks, self.block_size_ds, num_w_blocks, self.block_size_ds).permute(0, 1, 3, 5, 2, 4)
        # B, num_heads * dim_head * block_size ** 2, num_blocks
        q = q.reshape(B * self.num_heads, self.dim_head_qk, -1, num_blocks).transpose(1, 3)
        # B * num_heads, num_blocks, block_size ** 2, dim_head

        kv = self.to_kv(x)
        # Generate overlapping windows for kv. This approach is good for GPU and CPU. However, unfold() is not
        # lowered for PyTorch XLA so it will be very slow. See code at bottom of file for XLA friendly approach.
        # FIXME figure out how to switch impl between this and conv2d if XLA being used.
        kv = F.pad(kv, [self.halo_size, self.halo_size, self.halo_size, self.halo_size])
        kv = kv.unfold(2, self.win_size, self.block_size).unfold(3, self.win_size, self.block_size).reshape(
            B * self.num_heads, self.dim_head_qk + self.dim_head_v, num_blocks, -1).permute(0, 2, 3, 1)
        k, v = torch.split(kv, [self.dim_head_qk, self.dim_head_v], dim=-1)
        # B * num_heads, num_blocks, win_size ** 2, dim_head_qk or dim_head_v

        # if self.scale_pos_embed:
        #     attn = (q @ k.transpose(-1, -2) + self.pos_embed(q)) * self.scale
        # else:
        #     attn = (q @ k.transpose(-1, -2)) * self.scale + self.pos_embed(q)

        attn = ((q * self.scale) @ k.transpose(-1, -2)) + self.pos_embed(q)

        max_neg_value = -torch.finfo(attn.dtype).max
        attn.masked_fill_(self.get_mask(H, W, attn.device), max_neg_value)

        # B * num_heads, num_blocks, block_size ** 2, win_size ** 2
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 3)  # B * num_heads, dim_head_v, block_size ** 2, num_blocks
        # fold
        out = out.reshape(-1, self.block_size_ds, self.block_size_ds, num_h_blocks, num_w_blocks)
        out = out.permute(0, 3, 1, 4, 2).contiguous().view(
            B, self.dim_out_v, H // self.block_stride, W // self.block_stride)
        # B, dim_out, H // block_stride, W // block_stride
        out = self.to_out(out)
        return out

    def get_mask(self, H, W, device):
        if self.H == H and self.W == W and self.mask is not None:
            return self.mask

        num_h_blocks = H // self.block_size
        num_w_blocks = W // self.block_size
        num_blocks = num_h_blocks * num_w_blocks

        mask = torch.ones((1, 1, H, W), device=device)
        mask = F.pad(mask, [self.halo_size, self.halo_size, self.halo_size, self.halo_size])
        mask = mask.unfold(2, self.win_size, self.block_size)
        mask = mask.unfold(3, self.win_size, self.block_size)
        mask = mask.reshape(1, num_blocks, self.win_size * self.win_size)
        mask = mask.unsqueeze(-2)

        # 1, num_blocks, 1, win_size * win_size
        mask = mask.bool()

        self.H = H
        self.W = W
        self.mask = ~mask
        return self.mask


class HaloBlockV2(nn.Module):
    def __init__(self,
                 dim,
                 drop_path,
                 layer_scale_init_value,
                 block_size,
                 halo_size,
                 num_heads,
                 stage,
                 mlp_ratio=4.,
                 drop=0.,
                 act_layer=nn.GELU,
                 norm_layer=LayerNorm,
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = HaloAttn(dim=dim,
                             dim_out=dim,
                             num_heads=num_heads[stage],
                             block_size=block_size,
                             halo_size=halo_size)

        self.gamma_1 = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1, 1)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer,
                       drop=drop)

        self.gamma_2 = nn.Parameter(layer_scale_init_value * torch.ones((1, 1, 1, dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        # shape: B, C, H, W
        shortcut = x
        x = self.attn(self.norm1(x))

        if self.gamma_1 is not None:
            x = self.gamma_1 * x

        x = shortcut + self.drop_path(x)

        # FFN
        shortcut = x
        x = self.mlp(self.norm2(x).permute(0, 2, 3, 1))

        if self.gamma_2 is not None:
            x = self.gamma_2 * x

        x = x.permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(x)

        return x
