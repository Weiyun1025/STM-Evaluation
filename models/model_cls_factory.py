from timm.models import register_model
from .meta_arch import MetaArch
from .blocks.swin import SwinBlock
from .cls import MultiLayerClassBlock, ClassBlockV2, ClassBlockV3, GAPBlock

swin_cfgs = {
    'tiny': {
        'dims': [96 * 2 ** i for i in range(4)],
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
        'window_size': 7,
        'drop_path_rate': 0.2,
    },
    'small': {
        'dims': [96 * 2 ** i for i in range(4)],
        'depths': [2, 2, 18, 2],
        'num_heads': [3, 6, 12, 24],
        'window_size': 7,
        'drop_path_rate': 0.3,
    },
}
ablate_scale = 'tiny'


@register_model
def conv_swin_tiny(pretrained=False, **kwargs):
    dims = swin_cfgs[ablate_scale]['dims']
    depths = swin_cfgs[ablate_scale]['depths']
    num_heads = swin_cfgs[ablate_scale]['num_heads']
    window_size = swin_cfgs[ablate_scale]['window_size']
    drop_path_rate = swin_cfgs[ablate_scale]['drop_path_rate']

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=SwinBlock,
                     block_kwargs=dict(num_heads=num_heads, window_size=window_size),
                     active_stages=(3,),
                     cls_type=GAPBlock,
                     drop_path_rate=drop_path_rate,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def a1_swin_tiny(pretrained=False, **kwargs):
    dims = swin_cfgs[ablate_scale]['dims']
    depths = swin_cfgs[ablate_scale]['depths']
    num_heads = swin_cfgs[ablate_scale]['num_heads']
    window_size = swin_cfgs[ablate_scale]['window_size']
    drop_path_rate = swin_cfgs[ablate_scale]['drop_path_rate']

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=SwinBlock,
                     block_kwargs=dict(num_heads=num_heads, window_size=window_size),
                     active_stages=(3,),
                     cls_type=MultiLayerClassBlock,
                     cls_kwargs=dict(layer=1, query_len=1, mlp_ratio=1.),
                     drop_path_rate=drop_path_rate,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def a2_swin_tiny(pretrained=False, **kwargs):
    dims = swin_cfgs[ablate_scale]['dims']
    depths = swin_cfgs[ablate_scale]['depths']
    num_heads = swin_cfgs[ablate_scale]['num_heads']
    window_size = swin_cfgs[ablate_scale]['window_size']
    drop_path_rate = swin_cfgs[ablate_scale]['drop_path_rate']

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=SwinBlock,
                     block_kwargs=dict(num_heads=num_heads, window_size=window_size),
                     active_stages=(3,),
                     cls_type=MultiLayerClassBlock,
                     cls_kwargs=dict(layer=2, query_len=1, mlp_ratio=4.),
                     drop_path_rate=drop_path_rate,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def b1_swin_tiny(pretrained=False, **kwargs):
    dims = swin_cfgs[ablate_scale]['dims']
    depths = swin_cfgs[ablate_scale]['depths']
    num_heads = swin_cfgs[ablate_scale]['num_heads']
    window_size = swin_cfgs[ablate_scale]['window_size']
    drop_path_rate = swin_cfgs[ablate_scale]['drop_path_rate']

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=SwinBlock,
                     block_kwargs=dict(num_heads=num_heads, window_size=window_size),
                     active_stages=(3,),
                     cls_type=ClassBlockV2,
                     cls_kwargs=dict(query_len=5, mlp_ratio=1.5),
                     drop_path_rate=drop_path_rate,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def b2_swin_tiny(pretrained=False, **kwargs):
    dims = swin_cfgs[ablate_scale]['dims']
    depths = swin_cfgs[ablate_scale]['depths']
    num_heads = swin_cfgs[ablate_scale]['num_heads']
    window_size = swin_cfgs[ablate_scale]['window_size']
    drop_path_rate = swin_cfgs[ablate_scale]['drop_path_rate']

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=SwinBlock,
                     block_kwargs=dict(num_heads=num_heads, window_size=window_size),
                     active_stages=(3,),
                     cls_type=ClassBlockV3,
                     cls_kwargs=dict(query_len=5, mlp_ratio=1.),
                     drop_path_rate=drop_path_rate,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def b3_swin_tiny(pretrained=False, **kwargs):
    dims = swin_cfgs[ablate_scale]['dims']
    depths = swin_cfgs[ablate_scale]['depths']
    num_heads = swin_cfgs[ablate_scale]['num_heads']
    window_size = swin_cfgs[ablate_scale]['window_size']
    drop_path_rate = swin_cfgs[ablate_scale]['drop_path_rate']

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=SwinBlock,
                     block_kwargs=dict(num_heads=num_heads, window_size=window_size),
                     active_stages=(3,),
                     cls_type=ClassBlockV3,
                     cls_kwargs=dict(query_len=5, mlp_ratio=1.5),
                     drop_path_rate=drop_path_rate,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def c1_swin_tiny(pretrained=False, **kwargs):
    dims = swin_cfgs[ablate_scale]['dims']
    depths = swin_cfgs[ablate_scale]['depths']
    num_heads = swin_cfgs[ablate_scale]['num_heads']
    window_size = swin_cfgs[ablate_scale]['window_size']
    drop_path_rate = swin_cfgs[ablate_scale]['drop_path_rate']

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=SwinBlock,
                     block_kwargs=dict(num_heads=num_heads, window_size=window_size),
                     active_stages=(0, 1, 2, 3),
                     cls_type=ClassBlockV2,
                     cls_kwargs=dict(query_len=5, mlp_ratio=1.),
                     drop_path_rate=drop_path_rate,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def c2_swin_tiny(pretrained=False, **kwargs):
    dims = swin_cfgs[ablate_scale]['dims']
    depths = swin_cfgs[ablate_scale]['depths']
    num_heads = swin_cfgs[ablate_scale]['num_heads']
    window_size = swin_cfgs[ablate_scale]['window_size']
    drop_path_rate = swin_cfgs[ablate_scale]['drop_path_rate']

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=SwinBlock,
                     block_kwargs=dict(num_heads=num_heads, window_size=window_size),
                     active_stages=(0, 1, 2, 3),
                     cls_type=ClassBlockV3,
                     cls_kwargs=dict(query_len=5, mlp_ratio=1.),
                     drop_path_rate=drop_path_rate,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model
