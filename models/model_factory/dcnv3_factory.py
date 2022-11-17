from timm.models import register_model
from ..meta_arch import MetaArch
from ..blocks.dcn_v3 import DCNv3Block, DCNv3SingleResBlock


@register_model
def unified_dcn_v3_micro(pretrained=False, **kwargs):
    dims = [32 * 2 ** i for i in range(4)]
    depths = [2, 2, 9, 2]
    num_heads = [2, 4, 8, 16]

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     num_heads=num_heads,
                     block_type=DCNv3Block,
                     block_kwargs=dict(num_heads=num_heads, deform_points=9, kernel_size=3),
                     drop_path_rate=0,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def unified_dcn_v3_tiny(pretrained=False, **kwargs):
    dims = [64 * 2 ** i for i in range(4)]
    depths = [4, 4, 18, 4]
    num_heads = [4, 8, 16, 32]

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     num_heads=num_heads,
                     block_type=DCNv3Block,
                     block_kwargs=dict(num_heads=num_heads, deform_points=9, kernel_size=3),
                     drop_path_rate=0.1,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def unified_dcn_v3_small(pretrained=False, **kwargs):
    dims = [80 * 2 ** i for i in range(4)]
    depths = [4, 4, 21, 4]
    num_heads = [5, 10, 20, 40]

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     num_heads=num_heads,
                     block_type=DCNv3Block,
                     block_kwargs=dict(num_heads=num_heads, deform_points=9, kernel_size=3),
                     drop_path_rate=0.3,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def unified_dcn_v3_base(pretrained=False, **kwargs):
    dims = [112 * 2 ** i for i in range(4)]
    depths = [4, 4, 21, 4]
    num_heads = [7, 14, 28, 56]

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     num_heads=num_heads,
                     block_type=DCNv3Block,
                     block_kwargs=dict(num_heads=num_heads, deform_points=9, kernel_size=3),
                     drop_path_rate=0.5,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def unified_dcn_v3_b_small(pretrained=False, **kwargs):
    dims = [80 * 2 ** i for i in range(4)]
    depths = [4, 4, 21, 4]
    num_heads = [5, 10, 20, 40]

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     num_heads=num_heads,
                     block_type=DCNv3Block,
                     block_kwargs=dict(num_heads=num_heads, deform_points=9, kernel_size=3),
                     norm_after_avg=True,
                     drop_path_rate=0.3,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def unified_dcn_v3_c_small(pretrained=False, **kwargs):
    dims = [80 * 2 ** i for i in range(4)]
    depths = [4, 4, 21, 4]
    num_heads = [5, 10, 20, 40]

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     num_heads=num_heads,
                     block_type=DCNv3Block,
                     block_kwargs=dict(num_heads=num_heads, deform_points=9, kernel_size=3),
                     norm_every_stage=False,
                     norm_after_avg=True,
                     drop_path_rate=0.3,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def unified_dcn_v3_d_small(pretrained=False, **kwargs):
    dims = [80 * 2 ** i for i in range(4)]
    depths = [4, 4, 21, 4]
    num_heads = [5, 10, 20, 40]

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     num_heads=num_heads,
                     block_type=DCNv3SingleResBlock,
                     block_kwargs=dict(num_heads=num_heads, deform_points=9, kernel_size=3),
                     drop_path_rate=0.3,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def unified_dcn_v3_e_small(pretrained=False, **kwargs):
    dims = [80 * 2 ** i for i in range(4)]
    depths = [4, 4, 21, 4]
    num_heads = [5, 10, 20, 40]

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     num_heads=num_heads,
                     block_type=DCNv3SingleResBlock,
                     block_kwargs=dict(num_heads=num_heads, deform_points=9, kernel_size=3),
                     norm_every_stage=False,
                     norm_after_avg=True,
                     drop_path_rate=0.3,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model
