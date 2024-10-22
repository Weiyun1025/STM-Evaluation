from timm.models import register_model
from ..meta_arch import MetaArch
from ..blocks.swin import (
    SwinBlock, SwinBlockNoSwitch,
    SwinStem, SwinDownsampleLayer,
)

window_size_dict = {
    192: 12,
    224: 7,
    384: 12,
}


@register_model
def official_swin_tiny(pretrained=False, **kwargs):
    dims = [96 * 2 ** i for i in range(4)]
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]
    window_size = 7

    model = MetaArch(depths=depths,
                     dims=dims,
                     stem_type=SwinStem,
                     stem_kwargs=dict(patch_size=4),
                     block_type=SwinBlock,
                     block_kwargs=dict(num_heads=num_heads, window_size=window_size),
                     downsample_type=SwinDownsampleLayer,
                     extra_transform=False,
                     norm_every_stage=False,
                     norm_after_avg=False,
                     drop_path_rate=0.2,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def unified_swin_micro(pretrained=False, **kwargs):
    dims = [32 * 2 ** i for i in range(4)]
    depths = [2, 2, 9, 2]
    num_heads = [1, 2, 4, 8]

    img_size = kwargs.get('img_size', 224)
    window_size = window_size_dict[img_size]

    model = MetaArch(depths=depths,
                     dims=dims,
                     block_type=SwinBlock,
                     block_kwargs=dict(num_heads=num_heads, window_size=window_size),
                     drop_path_rate=0,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def unified_swin_tiny(pretrained=False, **kwargs):
    dims = [96 * 2 ** i for i in range(4)]
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]

    img_size = 224
    window_size = window_size_dict[img_size]

    model = MetaArch(depths=depths,
                     dims=dims,
                     block_type=SwinBlock,
                     block_kwargs=dict(num_heads=num_heads, window_size=window_size),
                     drop_path_rate=0.2,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def unified_swin_small(pretrained=False, **kwargs):
    dims = [96 * 2 ** i for i in range(4)]
    depths = [2, 2, 18, 2]
    num_heads = [3, 6, 12, 24]

    img_size = kwargs.get('img_size', 224)
    window_size = window_size_dict[img_size]

    model = MetaArch(depths=depths,
                     dims=dims,
                     block_type=SwinBlock,
                     block_kwargs=dict(num_heads=num_heads, window_size=window_size),
                     drop_path_rate=0.3,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def unified_swin_no_switch_small(pretrained=False, **kwargs):
    dims = [96 * 2 ** i for i in range(4)]
    depths = [2, 2, 18, 2]
    num_heads = [3, 6, 12, 24]

    img_size = kwargs.get('img_size', 224)
    window_size = window_size_dict[img_size]

    model = MetaArch(depths=depths,
                     dims=dims,
                     block_type=SwinBlockNoSwitch,
                     block_kwargs=dict(num_heads=num_heads, window_size=window_size),
                     drop_path_rate=0.3,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def unified_swin_base(pretrained=False, **kwargs):
    dims = [128 * 2 ** i for i in range(4)]
    depths = [2, 2, 18, 2]
    num_heads = [4, 8, 16, 32]

    img_size = kwargs.get('img_size', 224)
    window_size = window_size_dict[img_size]

    model = MetaArch(depths=depths,
                     dims=dims,
                     block_type=SwinBlock,
                     block_kwargs=dict(num_heads=num_heads, window_size=window_size),
                     drop_path_rate=0.5,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def unified_swin_large(pretrained=False, **kwargs):
    dims = [192 * 2 ** i for i in range(4)]
    depths = [2, 2, 18, 2]
    num_heads = [6, 12, 24, 48]
    window_size = 7

    img_size = kwargs.get('img_size', 224)
    window_size = window_size_dict[img_size]

    model = MetaArch(depths=depths,
                     dims=dims,
                     block_type=SwinBlock,
                     block_kwargs=dict(num_heads=num_heads, window_size=window_size),
                     drop_path_rate=0.2,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model
