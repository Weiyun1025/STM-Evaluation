from timm.models import register_model
from .meta_arch import MetaArch, PatchEmbed, PatchMerging
from .blocks.convnext import ConvNeXtBlock, ConvNeXtV2Block
from .blocks.swin import SwinBlock
from .blocks.pvt_v2 import PvtV2Block
from .blocks.halonet import HaloBlock, HaloBlockV2


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
def conv_convnext_tiny(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[3, 3, 9, 3],
                     dims=[96, 192, 384, 768],
                     block_type=ConvNeXtBlock,
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


@register_model
def conv_halo_tiny(pretrained=False, **kwargs):
    dims = [96 * 2 ** i for i in range(4)]
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]
    block_size = 7
    halo_size = 3

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=HaloBlock,
                     block_kwargs=dict(num_heads=num_heads, block_size=block_size, halo_size=halo_size),
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
                     block_type=HaloBlockV2,
                     block_kwargs=dict(num_heads=num_heads, block_size=block_size, halo_size=halo_size),
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model