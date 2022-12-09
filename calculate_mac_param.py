import argparse

import numpy as np
import torch
from thop import profile
import timm
from timm.models import create_model
import models


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    return parser.parse_args()


def sum_submodule_ops(module):
    ops = 0
    for m in module.children():
        if not getattr(m, '__thop_visit', False):
            ops += sum_submodule_ops(m)
        ops += m.total_ops
    module.__thop_visit = True
    return ops


def timm_halo_attn_posemb(module, input, output):
    module.total_ops += input[0].numel(
    ) / module.dim_head * module.width_rel.numel()
    module.total_ops += input[0].numel(
    ) / module.dim_head * module.height_rel.numel()
    module.total_ops += output.numel()


def haloblock(module, input, output):
    if module.gamma_1 is not None:
        module.total_ops += output.numel()
    if module.gamma_2 is not None:
        module.total_ops += output.numel()
    module.total_ops += output.numel() * 2
    module.total_ops += sum_submodule_ops(module)


def timm_halo_attn_haloatnn(module, input, output):
    # B, C, H, W = input[0].shape
    B, H, W, C = input[0].shape
    num_h_blocks = H // module.block_size
    num_w_blocks = W // module.block_size
    num_blocks = num_h_blocks * num_w_blocks
    num_heads = module.num_heads
    block_size = module.block_size
    dim_head_qk = module.dim_head_qk
    win_size = module.win_size
    dim_head_v = module.dim_head_v
    # q shape: B * num_heads, num_blocks, block_size ** 2, dim_head_qk
    # k shape: B * num_heads, num_blocks, win_size ** 2, dim_head_qk
    # v shape: B * num_heads, num_blocks, win_size ** 2, dim_head_v
    # attn shape: B * num_heads, num_blocks, block_size ** 2, win_size ** 2
    module.total_ops += B * num_heads * num_blocks * (block_size**2) * (
        dim_head_qk * win_size**2)  # q @ k.transpose(-1, -2)
    module.total_ops += B * num_heads * num_blocks * (block_size**2) * (
        win_size**2)  # scale
    module.total_ops += B * num_heads * num_blocks * (block_size**2) * (
        2 * win_size**2)  # softmax
    module.total_ops += B * num_heads * num_blocks * (block_size**2) * (
        win_size**2) * dim_head_v  # attn @ v
    module.total_ops += sum_submodule_ops(module)


def layernorm(module, input, output):
    input = input[0]
    batch_flops = np.prod(input.shape)
    if module.elementwise_affine:
        batch_flops *= 2
    module.total_ops += int(batch_flops)


def swinblock(module, input, output):
    if module.gamma_1 is not None:
        module.total_ops += output.numel()
    if module.gamma_2 is not None:
        module.total_ops += output.numel()
    module.total_ops += output.numel() * 2
    module.total_ops += sum_submodule_ops(module)


def swin_window_attention(module, input, output):
    num_heads = module.num_heads
    B, N, C = input[0].shape
    attn_dim = module.qkv.out_features / 3 / num_heads
    # q,k,v shape: B, num_heads, N, attn_dim
    module.total_ops += B * num_heads * N * attn_dim  # scale
    module.total_ops += B * num_heads * N * (attn_dim * N
                                             )  # q @ k.transpose(-2, -1)
    module.total_ops += B * num_heads * N * N  # attn + self._get_rel_pos_bias()
    module.total_ops += B * num_heads * N * (N * attn_dim)  # (attn @ v)
    module.total_ops += sum_submodule_ops(module)


def softmax(module, input, output):
    x = input[0]
    dim = x.shape[module.dim]
    module.total_ops += x.numel() / dim * (2 * dim)


def msdeform_attn_softmax(module, input, output):
    N, Len_q, C = input[0].shape
    n_heads = module.n_heads
    n_levels = module.n_levels
    n_points = module.n_points
    dim = C / n_heads
    # attention
    module.total_ops += N * Len_q * n_heads * n_levels * (n_points * dim)
    # attention_weights shape: N, Len_q, n_heads, n_levels * n_points
    # attention_weights softmax
    module.total_ops += N * Len_q * n_heads * (2 * n_levels * n_points)
    module.total_ops += sum_submodule_ops(module)


def dcnv3(module, input, output):
    x = input[0][0]
    if not module.layer_scale:
        module.total_ops += 2 * x.numel()
    else:
        module.total_ops += 4 * x.numel()
    module.total_ops += sum_submodule_ops(module)


def pvt_v2_atten(module, input, output):
    B, N, C = input[0].shape
    num_heads = module.num_heads
    dim = C // num_heads
    # q shape: B, num_heads, N,  C // num_heads
    # k v shape:  B, num_heads, N, C // num_heads
    module.total_ops += B * num_heads * N * (dim * N
                                             )  # q @ k.transpose(-2, -1)
    module.total_ops += B * num_heads * N * N  # scale
    module.total_ops += B * num_heads * N * (2 * N)  # softmax
    module.total_ops += B * num_heads * N * (N * dim)  # attn @ v
    module.total_ops += sum_submodule_ops(module)


def pvt_v2_block(module, input, output):
    x = input[0]
    if module.gamma_1 is not None:
        module.total_ops += x.numel()
    if module.gamma_2 is not None:
        module.total_ops += x.numel()
    module.total_ops += 2 * x.numel()
    module.total_ops += sum_submodule_ops(module)


def pvt_atten(module, input, output):
    B, N, C = input[0].shape

    if module.sr_ratio > 1:
        N = N / (module.sr_ratio ** 2)

    num_heads = module.num_heads
    dim = C // num_heads
    # q shape: B, num_heads, N,  C // num_heads
    # k v shape:  B, num_heads, N, C // num_heads
    module.total_ops += B * num_heads * N * (dim * N
                                             )  # q @ k.transpose(-2, -1)
    module.total_ops += B * num_heads * N * N  # scale
    module.total_ops += B * num_heads * N * (2 * N)  # softmax
    module.total_ops += B * num_heads * N * (N * dim)  # attn @ v
    module.total_ops += sum_submodule_ops(module)


def pvt_block(module, input, output):
    x = input[0]
    if module.gamma_1 is not None:
        module.total_ops += x.numel()
    if module.gamma_2 is not None:
        module.total_ops += x.numel()
    module.total_ops += 2 * x.numel()
    module.total_ops += sum_submodule_ops(module)


def convnext_block(module, input, output):
    x = input[0]
    if module.gamma is not None:
        module.total_ops += x.numel()
    module.total_ops += x.numel()
    module.total_ops += sum_submodule_ops(module)


custom_modules_hooks = {
    timm.models.layers.halo_attn.PosEmbedRel: timm_halo_attn_posemb,
    models.blocks.halonet.HaloAttn: timm_halo_attn_haloatnn,
    models.blocks.halonet.HaloBlockV2: haloblock,
    torch.nn.LayerNorm: layernorm,
    timm.models.layers.LayerNorm2d: layernorm,
    models.blocks.swin.SwinBlock: swinblock,
    timm.models.swin_transformer.WindowAttention: swin_window_attention,
    torch.nn.Softmax: softmax,
    models.blocks.dcn_v3.MSDeformAttnGrid_softmax: msdeform_attn_softmax,
    models.blocks.dcn_v3.DCNv3Block: dcnv3,
    models.blocks.pvt_v2.Attention: pvt_v2_atten,
    models.blocks.pvt_v2.PvtV2Block: pvt_v2_block,
    models.blocks.pvt.Attention: pvt_atten,
    models.blocks.pvt.PvtBlock: pvt_block,
    models.blocks.convnext.ConvNeXtBlock: convnext_block,
}


def main(args):
    model = create_model(args.model_name, pretrained=False,
                         num_classes=1000).eval().cuda()

    macs, params = profile(model, (torch.randn(1, 3, 224, 224).cuda(), ),
                           custom_ops=custom_modules_hooks,
                           verbose=False)

    macs = str(round(macs / 10.**9, 2))
    params = str(round(params / 10**6, 2))
    print(f"{args.model_name}: MACs {macs}G, Params {params}M")


if __name__ == '__main__':
    main(parse_args())

# salloc -p VC --gres=gpu:1 --quotatype=spot
# srun -p VC --gres=gpu:1 --quotatype=spot python calculate_mac_param.py --model_name unified_pvt_micro
