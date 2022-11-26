from timm.models import register_model
from ..meta_arch import MetaArch
from ..blocks.mobilenext import MobileNeXtV2Block


@ register_model
def unified_mobilenext_v2_small(pretrained=False, **kwargs):
    model = MetaArch(depths=[3, 3, 21, 3],
                     dims=[96, 192, 384, 768],
                     block_type=MobileNeXtV2Block,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model
