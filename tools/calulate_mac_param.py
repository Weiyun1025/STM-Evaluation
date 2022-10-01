import argparse

import numpy as np
import torch
from ptflops import get_model_complexity_info
import timm
from timm.models import create_model
import models
import models.swin


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    return parser.parse_args()


def timm_halo_attn_posemb(module, input, output):
    module.__flops__ += input[0].numel(
    ) / module.dim_head * module.width_rel.numel()
    module.__flops__ += input[0].numel(
    ) / module.dim_head * module.height_rel.numel()
    module.__flops__ += output.numel()


def timm_halo_attn_haloatnn(module, input, output):
    B, C, H, W = input[0].shape
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
    module.__flops__ += B * num_heads * num_blocks * (block_size**2) * (
        dim_head_qk * win_size**2)  # q @ k.transpose(-1, -2)
    module.__flops__ += B * num_heads * num_blocks * (block_size**2) * (
        win_size**2)  # scale
    module.__flops__ += B * num_heads * num_blocks * (block_size**2) * (
        2 * win_size**2)  # softmax
    module.__flops__ += B * num_heads * num_blocks * (block_size**2) * (
        win_size**2) * dim_head_v  # attn @ v
    for m in module.modules():
        if hasattr(m, '__flops__'):
            module.__flops__ += m.__flops__


def layernorm(module, input, output):
    input = input[0]
    batch_flops = np.prod(input.shape)
    if module.elementwise_affine:
        batch_flops *= 2
    module.__flops__ += int(batch_flops)


def swinblock(module, input, output):
    if module.gamma_1 is not None:
        module.__flops__ += output.numel()
    if module.gamma_2 is not None:
        module.__flops__ += output.numel()
    module.__flops__ += output.numel() * 2
    for m in module.modules():
        if hasattr(m, '__flops__'):
            module.__flops__ += m.__flops__


def swin_window_attention(module, input, output):
    num_heads = module.num_heads
    B, N, C = input[0].shape
    attn_dim = module.qkv.out_features / 3
    # q,k,v shape: B, num_heads, N, attn_dim
    module.__flops__ += B * num_heads * N * attn_dim  # scale
    module.__flops__ += B * num_heads * N * (attn_dim * N
                                             )  # q @ k.transpose(-2, -1)
    module.__flops__ += B * num_heads * N * N  # attn + self._get_rel_pos_bias()
    if len(input) > 1 and input[1] is not None:
        module.__flops__ += B * num_heads * N * N
    module.__flops__ += B * num_heads * N * (N * attn_dim)  # (attn @ v)
    for m in module.modules():
        if hasattr(m, '__flops__'):
            module.__flops__ += m.__flops__


def softmax(module, input, output):
    x = input[0]
    dim = x.shape[module.dim]
    module.__flops__ += x.numel() / dim * (2 * dim)


def msdeform_attn_softmax(module, input, output):
    N, Len_q, C = input[0].shape
    n_heads = module.n_heads
    n_levels = module.n_levels
    n_points = module.n_points
    dim = C / n_heads
    # attention
    module.__flops__ += N * Len_q * n_heads * n_levels * (n_points * dim)
    # attention_weights shape: N, Len_q, n_heads, n_levels * n_points
    # attention_weights softmax
    module.__flops__ += N * Len_q * n_heads * (2 * n_levels * n_points)
    for m in module.modules():
        if hasattr(m, '__flops__'):
            module.__flops__ += m.__flops__


def dcnv3(module, input, output):
    x = input[0][0]
    if not module.layer_scale:
        module.__flops__ += 2 * x.numel()
    else:
        module.__flops__ += 4 * x.numel()
    for m in module.modules():
        if hasattr(m, '__flops__'):
            module.__flops__ += m.__flops__


def pvt_v2_atten(module, input, output):
    B, N, C = input[0].shape
    num_heads = module.num_heads
    dim = C // num_heads
    # q shape: B, num_heads, N,  C // num_heads
    # k v shape:  B, num_heads, N, C // num_heads
    module.__flops__ += B * num_heads * N * (dim * N
                                             )  # q @ k.transpose(-2, -1)
    module.__flops__ += B * num_heads * N * N  # scale
    module.__flops__ += B * num_heads * N * (2 * N)  # softmax
    module.__flops__ += B * num_heads * N * (N * dim)  # attn @ v
    for m in module.modules():
        if hasattr(m, '__flops__'):
            module.__flops__ += m.__flops__


def pvt_v2_block(module, input, output):
    x = input[0]
    if module.gamma_1 is not None:
        module.__flops__ += x.numel()
    if module.gamma_2 is not None:
        module.__flops__ += x.numel()
    module.__flops__ += 2 * x.numel()
    for m in module.modules():
        if hasattr(m, '__flops__'):
            module.__flops__ += m.__flops__


def convnext_block(module, input, output):
    x = input[0]
    if module.gamma is not None:
        module.__flops__ += x.numel()
    module.__flops__ += x.numel()
    for m in module.modules():
        if hasattr(m, '__flops__'):
            module.__flops__ += m.__flops__


custom_modules_hooks = {
    timm.models.layers.halo_attn.PosEmbedRel: timm_halo_attn_posemb,
    timm.models.layers.halo_attn.HaloAttn: timm_halo_attn_haloatnn,
    torch.nn.LayerNorm: layernorm,
    timm.models.layers.LayerNorm2d: layernorm,
    models.blocks.swin.SwinBlock: swinblock,
    timm.models.swin_transformer.WindowAttention: swin_window_attention,
    torch.nn.Softmax: softmax,
    models.blocks.dcn_v3.MSDeformAttnGrid_softmax: msdeform_attn_softmax,
    models.blocks.dcn_v3.DCNv3Block: dcnv3,
    models.blocks.pvt_v2.Attention: pvt_v2_atten,
    models.blocks.pvt_v2.PvtV2Block: pvt_v2_block,
    models.blocks.convnext.ConvNeXtBlock: convnext_block,
}


def main(args):
    model = create_model(args.model_name, pretrained=False,
                         num_classes=1000).eval().cuda()

    flops, params = get_model_complexity_info(
        model, (3, 224, 224),
        as_strings=False,
        print_per_layer_stat=False,
        custom_modules_hooks=custom_modules_hooks,
        verbose=True)

    flops = str(round(flops / 10.**9, 2))
    params = str(round(params / 10**6, 2))
    print("model_name: MACs {}G, Params {}M".format(flops, params))


if __name__ == '__main__':
    main(parse_args())

# srun -p VC -N 1 --gres=gpu:1 --ntasks=1 \
#      --cpus-per-task=10 --quotatype=spot \
#      python tools/calulate_flops_param.py
