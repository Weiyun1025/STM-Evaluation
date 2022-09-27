import torch
from torch import nn
from timm.models import register_model
from timm.models.layers import DropPath, LayerNorm2d, Mlp, to_2tuple, make_divisible, trunc_normal_
import torch.nn.functional as F


class HaloBlockV3(nn.Module):
    def __init__(self, dim, drop_path, layer_scale_init_value, block_size, halo_size, stage, num_heads,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., head_dim=None, act_layer=nn.GELU,
                 norm_layer=LayerNorm2d, **kwargs):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = HaloAttentionV3(dim=dim, num_heads=num_heads[stage], block_size=block_size, halo_size=halo_size,
                                    qkv_bias=qkv_bias)

        self.gamma_1 = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1, 1)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

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


class HaloAttentionV3(nn.Module):
    def __init__(self, dim, num_heads=8, block_size=8, halo_size=3, qkv_bias=True, stride=1):
        super().__init__()
        #
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0
        self.dim_head = dim // num_heads
        self.scale = self.dim_head ** -0.5
        self.stride = stride
        #

        self.block_size = block_size
        self.halo_size = halo_size
        self.win_size = block_size + halo_size * 2  # neighbourhood window size

        # ------------- relative position encoding -------------
        q_size = block_size
        k_size = block_size + 2 * halo_size
        self.pos_size = (block_size + halo_size, block_size + halo_size)
        # define a parameter table of relative position bias
        #self.relative_position_bias_table = nn.Parameter(
        #    torch.zeros((2 * self.pos_size[0] - 1) * (2 * self.pos_size[1] - 1),
        #                num_heads))  # 2*(b+h)-1 * 2*(b+h)-1, nH
        self.relative_position_bias_table = torch.zeros((2 * self.pos_size[0] - 1) * (2 * self.pos_size[1] - 1), num_heads)  # 2*(b+h)-1 * 2*(b+h)-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(k_size)
        coords_w = torch.arange(k_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, k_size, k_size
        coords_flatten = torch.flatten(coords, 1)  # 2, k_size * k_size
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, k_size * k_size, k_size * k_size
        relative_coords = relative_coords.reshape(2, k_size, k_size, k_size,
                                                  k_size)  # 2, k_size, k_size, k_size, k_size
        relative_coords = relative_coords[:, halo_size:halo_size + q_size, halo_size:halo_size + q_size, :,
                          :]  # 2, q_size, q_size, k_size, k_size
        relative_coords = relative_coords.reshape(2, q_size ** 2, k_size ** 2)  # 2, q_size * q_size, k_size * k_size
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # q_size * q_size, k_size * k_size, 2
        relative_coords[:, :, 0] += self.pos_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.pos_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.pos_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # q_size * q_size, k_size * k_size
        #self.register_buffer("relative_position_index", relative_position_index)

        # relative position embedding
        relative_position_bias = self.relative_position_bias_table[relative_position_index.view(-1)].view(
            self.block_size ** 2, (self.block_size + 2 * self.halo_size) ** 2,
            self.num_heads)  # q_size * q_size, k_size * k_size, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, q_size * q_size, k_size * k_size
        relative_position_bias = relative_position_bias.unsqueeze(1).unsqueeze(0)
        self.relative_position_bias = nn.Parameter(relative_position_bias)
        #self.register_buffer("relative_position_bias", relative_position_bias)
        

        # ----- qkv project -----
        self.q = nn.Conv2d(dim, dim, 1, stride=stride, bias=qkv_bias)
        self.kv = nn.Conv2d(dim, 2 * dim, 1, stride=1, bias=qkv_bias)

        self.reset_parameters()

    def reset_parameters(self):
        std = self.q.weight.shape[1] ** -0.5  # fan-in
        trunc_normal_(self.q.weight, std=std)
        trunc_normal_(self.kv.weight, std=std)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    #@profile
    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.block_size == 0 and W % self.block_size == 0
        num_h_blocks = H // self.block_size
        num_w_blocks = W // self.block_size
        num_blocks = num_h_blocks * num_w_blocks

        # unfold
        q = self.q(x).permute(0, 3, 1, 2)
        q = q.reshape(
            -1, self.dim_head,
            num_h_blocks, self.block_size_ds, num_w_blocks, self.block_size_ds).permute(0, 1, 3, 5, 2, 4)
        # B, num_heads * dim_head * block_size ** 2, num_blocks
        q = q.reshape(B * self.num_heads, self.dim_head, -1, num_blocks).transpose(1, 3)
        # B * num_heads, num_blocks, block_size ** 2, dim_head

        kv = self.kv(x).permute(0, 3, 1, 2)
        # Generate overlapping windows for kv. This approach is good for GPU and CPU. However, unfold() is not
        # lowered for PyTorch XLA so it will be very slow. See code at bottom of file for XLA friendly approach.
        # FIXME figure out how to switch impl between this and conv2d if XLA being used.
        kv = F.pad(kv, [self.halo_size, self.halo_size, self.halo_size, self.halo_size])
        kv = kv.unfold(2, self.win_size, self.block_size).unfold(3, self.win_size, self.block_size).reshape(
            B * self.num_heads, self.dim_head_qk + self.dim_head_v, num_blocks, -1).permute(0, 2, 3, 1)
        
        k, v = torch.split(kv, [self.dim_head_qk, self.dim_head_v], dim=-1)

        

        # add relative position
        relative_position_bias = self.relative_position_bias.expand(B, -1, num_blocks, -1, -1)
        if self.scale_pos_embed:
            attn = (q @ k.transpose(-1, -2) + relative_position_bias) * self.scale
        else:
            attn = (q @ k.transpose(-1, -2)) * self.scale
            #_ = self.pos_embed(q)
            attn = attn + relative_position_bias
        
         # B * num_heads, num_blocks, block_size ** 2, win_size ** 2
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 3)  # B * num_heads, dim_head_v, block_size ** 2, num_blocks
        # fold
        out = out.reshape(-1, self.block_size_ds, self.block_size_ds, num_h_blocks, num_w_blocks)
        out = out.permute(0, 3, 1, 4, 2).contiguous().view(
            B, self.dim_out_v, H // self.block_stride, W // self.block_stride)
        # B, dim_out, H // block_stride, W // block_stride
        #out = self.pool(out)

        '''
        attn_logits = (q @ k.transpose(-1, -2)) * self.scale  # FIXME should usual attn scale be applied?
        attn_logits = attn_logits.reshape(B, self.num_heads, num_blocks, self.block_size ** 2, self.win_size ** 2)
        attn_logits = attn_logits + relative_position_bias  # B, num_heads, num_blocks, block_size ** 2, win_size ** 2
        attn_logits = attn_logits.reshape(B * self.num_heads, num_blocks, self.block_size ** 2,
                                          self.win_size ** 2)  # B * num_heads, num_blocks, block_size ** 2, win_size ** 2
        attn_out = attn_logits.softmax(dim=-1)
        attn_out = (attn_out @ v).transpose(1, 3)  # B * num_heads, dim_v // num_heads, block_size ** 2, num_blocks
        attn_out = F.fold(
            attn_out.reshape(B, -1, num_blocks),
            (H // self.stride, W // self.stride),
            kernel_size=self.block_size // self.stride, stride=self.block_size // self.stride)
        # B, dim_out, H // stride, W // stride
        '''
        return out

