from timm.models import register_model
from .meta_arch import MetaArch, PatchEmbed, PatchMerging
from .blocks.convnext import ConvNeXtBlock, ConvNeXtV2Block, ConvNeXtV3Block
from .blocks.swin import SwinBlock
from .blocks.dcn_v3 import DCNv3Block
from .blocks.pvt import PvtBlock
from .blocks.pvt_v2 import PvtV2Block
from .blocks import halonet_github, halonet_timm
from .blocks import halonet_opt_v1, halo_opt_abs_pe


@ register_model
def pe_convnext_tiny(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[3, 3, 9, 3],
                     dims=[96, 192, 384, 768],
                     stem_type=PatchEmbed,
                     stem_kwargs=dict(patch_size=4),
                     block_type=ConvNeXtBlock,
                     downsample_type=PatchEmbed,
                     downsample_kwargs=dict(patch_size=2),
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@ register_model
def pm_convnext_tiny(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[3, 3, 9, 3],
                     dims=[96, 192, 384, 768],
                     stem_type=PatchEmbed,
                     stem_kwargs=dict(patch_size=4),
                     block_type=ConvNeXtBlock,
                     downsample_type=PatchMerging,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@ register_model
def pe_convnext_v2_tiny(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[3, 3, 9, 3],
                     dims=[96, 192, 384, 768],
                     stem_type=PatchEmbed,
                     stem_kwargs=dict(patch_size=4),
                     block_type=ConvNeXtV2Block,
                     downsample_type=PatchEmbed,
                     downsample_kwargs=dict(patch_size=2),
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@ register_model
def conv_convnext_v2_micro(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[3, 3, 12, 3],
                     dims=[32, 64, 128, 256],
                     block_type=ConvNeXtV2Block,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@ register_model
def conv_convnext_v2_tiny(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[3, 3, 9, 3],
                     dims=[96, 192, 384, 768],
                     block_type=ConvNeXtV2Block,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@ register_model
def conv_convnext_v2_small(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[3, 3, 27, 3],
                     dims=[96, 192, 384, 768],
                     block_type=ConvNeXtV2Block,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@ register_model
def conv_convnext_v2_base(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[3, 3, 27, 3],
                     dims=[128, 256, 512, 1024],
                     block_type=ConvNeXtV2Block,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@ register_model
def conv_convnext_v3_tiny(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[2, 2, 9, 2],
                     dims=[96, 192, 384, 768],
                     block_type=ConvNeXtV3Block,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@ register_model
def conv_convnext_v3_small(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[2, 2, 24, 2],
                     dims=[96, 192, 384, 768],
                     block_type=ConvNeXtV3Block,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model

# *********************************************************
# Swin


@register_model
def pe_swin_tiny(pretrained=False, **kwargs):
    dims = [96 * 2 ** i for i in range(4)]
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]
    window_size = 7

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     stem_type=PatchEmbed,
                     stem_kwargs=dict(patch_size=4),
                     block_type=SwinBlock,
                     block_kwargs=dict(num_heads=num_heads, window_size=window_size),
                     downsample_type=PatchEmbed,
                     downsample_kwargs=dict(patch_size=2),
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def pm_swin_tiny(pretrained=False, **kwargs):
    dims = [96 * 2 ** i for i in range(4)]
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]
    window_size = 7

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     stem_type=PatchEmbed,
                     stem_kwargs=dict(patch_size=4),
                     block_type=SwinBlock,
                     block_kwargs=dict(num_heads=num_heads, window_size=window_size),
                     downsample_type=PatchMerging,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model

# ******************* Swin with Conv Stem and Conv transition layer ******************


@register_model
def conv_swin_micro(pretrained=False, **kwargs):
    dims = [32 * 2 ** i for i in range(4)]
    depths = [2, 2, 9, 2]
    num_heads = [1, 2, 4, 8]
    window_size = 7

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=SwinBlock,
                     block_kwargs=dict(num_heads=num_heads, window_size=window_size),
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def conv_swin_tiny(pretrained=False, **kwargs):
    dims = [96 * 2 ** i for i in range(4)]
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]
    window_size = 7

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=SwinBlock,
                     block_kwargs=dict(num_heads=num_heads, window_size=window_size),
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def conv_swin_small(pretrained=False, **kwargs):
    dims = [96 * 2 ** i for i in range(4)]
    depths = [2, 2, 18, 2]
    num_heads = [3, 6, 12, 24]
    window_size = 7

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=SwinBlock,
                     block_kwargs=dict(num_heads=num_heads, window_size=window_size),
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def conv_swin_base(pretrained=False, **kwargs):
    dims = [128 * 2 ** i for i in range(4)]
    depths = [2, 2, 18, 2]
    num_heads = [4, 8, 16, 32]
    window_size = 7

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=SwinBlock,
                     block_kwargs=dict(num_heads=num_heads, window_size=window_size),
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model
# ******************************************************************************

# ******************************************************************************
# DCN V3

# drop path rate should be set to 0.05


@register_model
def dcn_v3_micro(pretrained=False, **kwargs):
    dims = [32 * 2 ** i for i in range(4)]
    depths = [2, 2, 9, 2]
    num_heads = [2, 4, 8, 16]

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     num_heads=num_heads,
                     block_type=DCNv3Block,
                     block_kwargs=dict(num_heads=num_heads, deform_points=9, kernel_size=3),
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


# drop path rate should be set to 0.1
@register_model
def dcn_v3_tiny(pretrained=False, **kwargs):
    dims = [64 * 2 ** i for i in range(4)]
    depths = [4, 4, 18, 4]
    num_heads = [4, 8, 16, 32]

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     num_heads=num_heads,
                     block_type=DCNv3Block,
                     block_kwargs=dict(num_heads=num_heads, deform_points=9, kernel_size=3),
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


# drop path set to 0.3
@register_model
def dcn_v3_small(pretrained=False, **kwargs):
    dims = [80 * 2 ** i for i in range(4)]
    depths = [4, 4, 21, 4]
    num_heads = [5, 10, 20, 40]

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     num_heads=num_heads,
                     block_type=DCNv3Block,
                     block_kwargs=dict(num_heads=num_heads, deform_points=9, kernel_size=3),
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model

# drop path rate should be set to 0.5


@register_model
def dcn_v3_base(pretrained=False, **kwargs):
    dims = [112 * 2 ** i for i in range(4)]
    depths = [4, 4, 21, 4]
    num_heads = [7, 14, 28, 56]

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     num_heads=num_heads,
                     block_type=DCNv3Block,
                     block_kwargs=dict(num_heads=num_heads, deform_points=9, kernel_size=3),
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model

# ******************************************************************
# PVT v2 with conv stem and conv transition


@register_model
def conv_pvt_micro(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[2, 2, 3, 2],
                     dims=[32, 64, 160, 256],
                     block_type=PvtBlock,
                     block_kwargs=dict(num_heads=[1, 2, 5, 8],
                                       mlp_ratios=[8, 8, 4, 4],
                                       qkv_bias=True,
                                       sr_ratios=[8, 4, 2, 1],),
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model

# B2 config


@register_model
def conv_pvt_tiny(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[3, 4, 9, 3],
                     dims=[64, 128, 320, 512],
                     block_type=PvtBlock,
                     block_kwargs=dict(num_heads=[1, 2, 5, 8],
                                       mlp_ratios=[8, 8, 4, 4],
                                       qkv_bias=True,
                                       sr_ratios=[8, 4, 2, 1],),
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model

# b3 config


@register_model
def conv_pvt_small(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[3, 4, 21, 3],
                     dims=[64, 128, 320, 512],
                     block_type=PvtBlock,
                     block_kwargs=dict(num_heads=[1, 2, 5, 8],
                                       mlp_ratios=[8, 8, 4, 4],
                                       qkv_bias=True,
                                       sr_ratios=[8, 4, 2, 1],),
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def conv_pvt_base(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[3, 8, 45, 3],
                     dims=[64, 128, 320, 512],
                     block_type=PvtBlock,
                     block_kwargs=dict(num_heads=[1, 2, 5, 8],
                                       mlp_ratios=[4, 4, 4, 4],
                                       qkv_bias=True,
                                       sr_ratios=[8, 4, 2, 1],),
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


# ******************************************************************
# PVT v2 with conv stem and conv transition
@register_model
def conv_pvt_v2_b0(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[2, 2, 2, 2],
                     dims=[32, 64, 160, 256],
                     block_type=PvtV2Block,
                     block_kwargs=dict(num_heads=[1, 2, 5, 8],
                                       mlp_ratios=[8, 8, 4, 4],
                                       qkv_bias=True,
                                       sr_ratios=[8, 4, 2, 1],),
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model

# b1 config


@register_model
def conv_pvt_v2_micro(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[2, 2, 3, 2],
                     dims=[32, 64, 160, 256],
                     block_type=PvtV2Block,
                     block_kwargs=dict(num_heads=[1, 2, 5, 8],
                                       mlp_ratios=[8, 8, 4, 4],
                                       qkv_bias=True,
                                       sr_ratios=[8, 4, 2, 1],),
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model

# B2 config


@register_model
def conv_pvt_v2_tiny(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[3, 3, 9, 3],
                     dims=[64, 128, 320, 512],
                     block_type=PvtV2Block,
                     block_kwargs=dict(num_heads=[1, 2, 5, 8],
                                       mlp_ratios=[8, 8, 4, 4],
                                       qkv_bias=True,
                                       sr_ratios=[8, 4, 2, 1],),
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model

# b3 config


@register_model
def conv_pvt_v2_small(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[3, 3, 21, 3],
                     dims=[64, 128, 320, 512],
                     block_type=PvtV2Block,
                     block_kwargs=dict(num_heads=[1, 2, 5, 8],
                                       mlp_ratios=[8, 8, 4, 4],
                                       qkv_bias=True,
                                       sr_ratios=[8, 4, 2, 1],),
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model

# b5 config


@register_model
def conv_pvt_v2_base(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[3, 6, 45, 3],
                     dims=[64, 128, 320, 512],
                     block_type=PvtV2Block,
                     block_kwargs=dict(num_heads=[1, 2, 5, 8],
                                       mlp_ratios=[4, 4, 4, 4],
                                       qkv_bias=True,
                                       sr_ratios=[8, 4, 2, 1],),
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


# ******************************************************************
# HaloNet with swin block design and conv stem & transition
@register_model
def conv_halo_v2_mask_tiny(pretrained=False, **kwargs):
    dims = [96 * 2 ** i for i in range(4)]
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]
    block_size = 7
    halo_size = 3

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=halonet_timm_with_mask.HaloBlockV2,
                     block_kwargs=dict(num_heads=num_heads,
                                       block_size=block_size,
                                       halo_size=halo_size,
                                       out_proj=False),
                     #  downsample_type=nn.Identity,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def conv_halo_v2_mask_out_tiny(pretrained=False, **kwargs):
    dims = [96 * 2 ** i for i in range(4)]
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]
    block_size = 7
    halo_size = 3

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=halonet_timm_with_mask.HaloBlockV2,
                     block_kwargs=dict(num_heads=num_heads,
                                       block_size=block_size,
                                       halo_size=halo_size,
                                       out_proj=True),
                     #  downsample_type=nn.Identity,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def conv_halo_v2_mask_out_base(pretrained=False, **kwargs):
    dims = [128 * 2 ** i for i in range(4)]
    depths = [2, 2, 18, 2]
    num_heads = [4, 8, 16, 32]
    block_size = 7
    halo_size = 3

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=halonet_timm_with_mask.HaloBlockV2,
                     block_kwargs=dict(num_heads=num_heads,
                                       block_size=block_size,
                                       halo_size=halo_size,
                                       out_proj=True),
                     #  downsample_type=nn.Identity,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def conv_halo_v2_timm_micro(pretrained=False, **kwargs):
    dims = [32 * 2 ** i for i in range(4)]
    depths = [2, 2, 9, 2]
    num_heads = [1, 2, 4, 8]
    block_size = 7
    halo_size = 3

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=halonet_timm.HaloBlockV2,
                     block_kwargs=dict(num_heads=num_heads,
                                       block_size=block_size,
                                       halo_size=halo_size),
                     #  downsample_type=nn.Identity,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def conv_halo_v2_timm_tiny(pretrained=False, **kwargs):
    dims = [96 * 2 ** i for i in range(4)]
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]
    block_size = 7
    halo_size = 3

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=halonet_timm.HaloBlockV2,
                     block_kwargs=dict(num_heads=num_heads,
                                       block_size=block_size,
                                       halo_size=halo_size),
                     #  downsample_type=nn.Identity,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def conv_halo_v2_timm_small(pretrained=False, **kwargs):
    dims = [96 * 2 ** i for i in range(4)]
    depths = [2, 2, 18, 2]
    num_heads = [3, 6, 12, 24]
    block_size = 7
    halo_size = 3

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=halonet_timm.HaloBlockV2,
                     block_kwargs=dict(num_heads=num_heads,
                                       block_size=block_size,
                                       halo_size=halo_size),
                     #  downsample_type=nn.Identity,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def conv_halo_v2_timm_base(pretrained=False, **kwargs):
    dims = [128 * 2 ** i for i in range(4)]
    depths = [2, 2, 18, 2]
    num_heads = [4, 8, 16, 32]
    block_size = 7
    halo_size = 3

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=halonet_timm.HaloBlockV2,
                     block_kwargs=dict(num_heads=num_heads,
                                       block_size=block_size,
                                       halo_size=halo_size),
                     #  downsample_type=nn.Identity,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def conv_halo_v2_github_micro(pretrained=False, **kwargs):
    dims = [32 * 2 ** i for i in range(4)]
    depths = [2, 2, 9, 2]
    num_heads = [1, 2, 4, 8]
    block_size = 7
    halo_size = 3

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=halonet_github.HaloBlockV2,
                     block_kwargs=dict(num_heads=num_heads,
                                       block_size=block_size,
                                       halo_size=halo_size),
                     #  downsample_type=nn.Identity,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def conv_halo_v2_github_tiny(pretrained=False, **kwargs):
    dims = [96 * 2 ** i for i in range(4)]
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]
    block_size = 7
    halo_size = 3

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=halonet_github.HaloBlockV2,
                     block_kwargs=dict(num_heads=num_heads,
                                       block_size=block_size,
                                       halo_size=halo_size),
                     #  downsample_type=nn.Identity,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def conv_halo_v2_github_small(pretrained=False, **kwargs):
    dims = [96 * 2 ** i for i in range(4)]
    depths = [2, 2, 18, 2]
    num_heads = [3, 6, 12, 24]
    block_size = 7
    halo_size = 3

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=halonet_github.HaloBlockV2,
                     block_kwargs=dict(num_heads=num_heads,
                                       block_size=block_size,
                                       halo_size=halo_size),
                     #  downsample_type=nn.Identity,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def conv_halo_v2_github_base(pretrained=False, **kwargs):
    dims = [128 * 2 ** i for i in range(4)]
    depths = [2, 2, 18, 2]
    num_heads = [4, 8, 16, 32]
    block_size = 7
    halo_size = 3

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=halonet_github.HaloBlockV2,
                     block_kwargs=dict(num_heads=num_heads,
                                       block_size=block_size,
                                       halo_size=halo_size),
                     #  downsample_type=nn.Identity,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model

# ******************************************************************
# HaloNet with swin block design and switch halo_size and conv stem & transition


@register_model
def conv_halo_v2_micro(pretrained=False, **kwargs):
    dims = [32 * 2 ** i for i in range(4)]
    depths = [2, 2, 9, 2]
    num_heads = [1, 2, 4, 8]
    block_size = 7
    halo_size = 3

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=halonet.HaloBlockV2,
                     block_kwargs=dict(num_heads=num_heads,
                                       block_size=block_size,
                                       halo_size=halo_size),
                     #  downsample_type=nn.Identity,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def conv_halo_v2_tiny(pretrained=False, **kwargs):
    dims = [96 * 2 ** i for i in range(4)]
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]
    block_size = 7
    halo_size = 3

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=halonet.HaloBlockV2,
                     block_kwargs=dict(num_heads=num_heads,
                                       block_size=block_size,
                                       halo_size=halo_size),
                     #  downsample_type=nn.Identity,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def conv_halo_v2_small(pretrained=False, **kwargs):
    dims = [96 * 2 ** i for i in range(4)]
    depths = [2, 2, 18, 2]
    num_heads = [3, 6, 12, 24]
    block_size = 7
    halo_size = 3

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=halonet.HaloBlockV2,
                     block_kwargs=dict(num_heads=num_heads,
                                       block_size=block_size,
                                       halo_size=halo_size),
                     #  downsample_type=nn.Identity,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def conv_halo_v2_base(pretrained=False, **kwargs):
    dims = [128 * 2 ** i for i in range(4)]
    depths = [2, 2, 18, 2]
    num_heads = [4, 8, 16, 32]
    block_size = 7
    halo_size = 3

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=halonet.HaloBlockV2,
                     block_kwargs=dict(num_heads=num_heads,
                                       block_size=block_size,
                                       halo_size=halo_size),
                     #  downsample_type=nn.Identity,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def conv_halo_opt_v1_base(pretrained=False, **kwargs):
    dims = [128 * 2 ** i for i in range(4)]
    depths = [2, 2, 18, 2]
    num_heads = [4, 8, 16, 32]
    block_size = 7
    halo_size = 3

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=halonet_opt_v1.HaloBlockV2,
                     block_kwargs=dict(num_heads=num_heads,
                                       block_size=block_size,
                                       halo_size=halo_size),
                     #  downsample_type=nn.Identity,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model

@register_model
def conv_halo_opt_v1_tiny(pretrained=False, **kwargs):
    dims = [96 * 2 ** i for i in range(4)]
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]
    block_size = 7
    halo_size = 3

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=halonet_opt_v1.HaloBlockV2,
                     block_kwargs=dict(num_heads=num_heads,
                                       block_size=block_size,
                                       halo_size=halo_size),
                     #  downsample_type=nn.Identity,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model

@register_model
def conv_halo_opt_abs_pe_tiny(pretrained=False, **kwargs):
    dims = [96 * 2 ** i for i in range(4)]
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]
    block_size = 7
    halo_size = 3

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=halo_opt_abs_pe.HaloBlockV2,
                     block_kwargs=dict(num_heads=num_heads,
                                       block_size=block_size,
                                       halo_size=halo_size),
                     #  downsample_type=nn.Identity,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model
