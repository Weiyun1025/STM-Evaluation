from timm.models import register_model
from .meta_arch import MetaArch
from .blocks.swin import SwinBlock
from .cls import MultiLayerClassBlock, ClassBlockV2


@register_model
def a1_swin_tiny(pretrained=False, **kwargs):
    dims = [96 * 2 ** i for i in range(4)]
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]
    window_size = 7

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=SwinBlock,
                     block_kwargs=dict(num_heads=num_heads, window_size=window_size),
                     active_stages=(3,),
                     cls_type=MultiLayerClassBlock,
                     cls_kwargs=dict(layer=1, query_len=1),
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def a2_swin_tiny(pretrained=False, **kwargs):
    dims = [96 * 2 ** i for i in range(4)]
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]
    window_size = 7

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=SwinBlock,
                     block_kwargs=dict(num_heads=num_heads, window_size=window_size),
                     active_stages=(3,),
                     cls_type=MultiLayerClassBlock,
                     cls_kwargs=dict(layer=2, query_len=1),
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def b1_swin_tiny(pretrained=False, **kwargs):
    dims = [96 * 2 ** i for i in range(4)]
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]
    window_size = 7

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=SwinBlock,
                     block_kwargs=dict(num_heads=num_heads, window_size=window_size),
                     active_stages=(3,),
                     cls_type=ClassBlockV2,
                     cls_kwargs=dict(query_len=5),
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def b2_swin_tiny(pretrained=False, **kwargs):
    dims = [96 * 2 ** i for i in range(4)]
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]
    window_size = 7

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=SwinBlock,
                     block_kwargs=dict(num_heads=num_heads, window_size=window_size),
                     active_stages=(3,),
                     end_attn=True,
                     cls_type=ClassBlockV2,
                     cls_kwargs=dict(query_len=5),
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def c1_swin_tiny(pretrained=False, **kwargs):
    dims = [96 * 2 ** i for i in range(4)]
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]
    window_size = 7

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=SwinBlock,
                     block_kwargs=dict(num_heads=num_heads, window_size=window_size),
                     active_stages=(0, 1, 2, 3),
                     cls_type=ClassBlockV2,
                     cls_kwargs=dict(query_len=5),
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def c2_swin_tiny(pretrained=False, **kwargs):
    dims = [96 * 2 ** i for i in range(4)]
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]
    window_size = 7

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=SwinBlock,
                     block_kwargs=dict(num_heads=num_heads, window_size=window_size),
                     active_stages=(0, 1, 2, 3),
                     end_attn=True,
                     cls_type=ClassBlockV2,
                     cls_kwargs=dict(query_len=5),
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model
