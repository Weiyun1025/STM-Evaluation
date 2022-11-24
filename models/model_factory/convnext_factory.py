from timm.models import register_model
from ..meta_arch import MetaArch
from ..blocks.convnext import (
    ConvNeXtBlock, ConvNeXtV2Block, ConvNeXtV3Block, ConvNeXtV3SingleResBlock,
    ConvNeXtStem, ConvNeXtDownsampleLayer,
)


@ register_model
def official_convnext_tiny(pretrained=False, **kwargs):
    model = MetaArch(img_size=224,
                     depths=[3, 3, 9, 3],
                     dims=[96, 192, 384, 768],
                     stem_type=ConvNeXtStem,
                     block_type=ConvNeXtBlock,
                     downsample_type=ConvNeXtDownsampleLayer,
                     extra_transform=False,
                     norm_every_stage=False,
                     norm_after_avg=True,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@ register_model
def unified_convnext_v2_micro(pretrained=False, **kwargs):
    model = MetaArch(img_size=kwargs.get('img_size', 224),
                     depths=[3, 3, 12, 3],
                     dims=[32, 64, 128, 256],
                     block_type=ConvNeXtV2Block,
                     drop_path_rate=0,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@ register_model
def unified_convnext_v3_micro(pretrained=False, **kwargs):
    model = MetaArch(img_size=kwargs.get('img_size', 224),
                     depths=[3, 3, 8, 3],
                     dims=[32, 64, 128, 256],
                     block_type=ConvNeXtV3Block,
                     drop_path_rate=0,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@ register_model
def unified_convnext_v2_tiny(pretrained=False, **kwargs):
    model = MetaArch(img_size=kwargs.get('img_size', 224),
                     depths=[3, 3, 9, 3],
                     dims=[96, 192, 384, 768],
                     block_type=ConvNeXtV2Block,
                     drop_path_rate=0.1,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@ register_model
def unified_convnext_v3_tiny(pretrained=False, **kwargs):
    model = MetaArch(img_size=kwargs.get('img_size', 224),
                     depths=[2, 2, 9, 2],
                     dims=[96, 192, 384, 768],
                     block_type=ConvNeXtV3Block,
                     drop_path_rate=0.1,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@ register_model
def unified_convnext_v1_small(pretrained=False, **kwargs):
    model = MetaArch(img_size=kwargs.get('img_size', 224),
                     depths=[3, 3, 27, 3],
                     dims=[96, 192, 384, 768],
                     block_type=ConvNeXtBlock,
                     drop_path_rate=0.4,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@ register_model
def unified_convnext_v2_small(pretrained=False, **kwargs):
    model = MetaArch(img_size=kwargs.get('img_size', 224),
                     depths=[3, 3, 27, 3],
                     dims=[96, 192, 384, 768],
                     block_type=ConvNeXtV2Block,
                     drop_path_rate=0.4,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@ register_model
def unified_convnext_v2_large_kernel_small(pretrained=False, **kwargs):
    model = MetaArch(img_size=kwargs.get('img_size', 224),
                     depths=[3, 3, 27, 3],
                     dims=[96, 192, 384, 768],
                     block_type=ConvNeXtV2Block,
                     block_kwargs=dict(kernel_size=13),
                     drop_path_rate=0.4,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@ register_model
def unified_convnext_v3_small(pretrained=False, **kwargs):
    model = MetaArch(img_size=kwargs.get('img_size', 224),
                     depths=[2, 2, 24, 2],
                     dims=[96, 192, 384, 768],
                     block_type=ConvNeXtV3Block,
                     drop_path_rate=0.4,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@ register_model
def unified_convnext_v3_large_kernel_small(pretrained=False, **kwargs):
    model = MetaArch(img_size=kwargs.get('img_size', 224),
                     depths=[2, 2, 24, 2],
                     dims=[96, 192, 384, 768],
                     block_type=ConvNeXtV3Block,
                     block_kwargs=dict(kernel_size=13),
                     drop_path_rate=0.4,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@ register_model
def unified_convnext_v2_base(pretrained=False, **kwargs):
    model = MetaArch(img_size=kwargs.get('img_size', 224),
                     depths=[3, 3, 27, 3],
                     dims=[128, 256, 512, 1024],
                     block_type=ConvNeXtV2Block,
                     drop_path_rate=0.5,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@ register_model
def unified_convnext_v3_base(pretrained=False, **kwargs):
    model = MetaArch(img_size=kwargs.get('img_size', 224),
                     depths=[2, 2, 24, 2],
                     dims=[128, 256, 512, 1024],
                     block_type=ConvNeXtV3Block,
                     drop_path_rate=0.5,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@ register_model
def unified_convnext_v3_large(pretrained=False, **kwargs):
    model = MetaArch(img_size=kwargs.get('img_size', 224),
                     depths=[2, 2, 24, 2],
                     dims=[192, 384, 768, 1536],
                     block_type=ConvNeXtV3Block,
                     drop_path_rate=0.1,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model
