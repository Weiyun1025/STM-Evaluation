from timm.models import register_model
from timm.models.helpers import build_model_with_cfg
from timm.models.byobnet import ByoBlockCfg, ByoModelCfg, ByobNet


def halo_cfg(b, h, rv, rb, l3, df=None):
    return ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='self_attn', d=3, c=64 * rb, s=1, gs=0, br=rv / rb),
            ByoBlockCfg(type='self_attn', d=3, c=128 * rb, s=2, gs=0, br=rv / rb),
            ByoBlockCfg(type='self_attn', d=l3, c=256 * rb, s=2, gs=0, br=rv / rb),
            ByoBlockCfg(type='self_attn', d=3, c=512 * rb, s=2, gs=0, br=rv / rb),
        ),
        stem_chs=64,
        stem_type='7x7',
        stem_pool='maxpool',

        num_features=df,
        self_attn_layer='halo',
        self_attn_kwargs=dict(block_size=b, halo_size=h),
    )


# TODO: add layer scale
def _create_byoanet(variant, b, h, rv, rb, l3, df=None, pretrained=False,
                    layerscale_opt=None, layerscale_init_values=None, **kwargs):
    return build_model_with_cfg(
        ByobNet, variant, pretrained,
        model_cfg=halo_cfg(b, h, rv, rb, l3, df),
        feature_cfg=dict(flatten_sequential=True),
        **kwargs)


@register_model
def halonet_h0(pretrained=False, **kwargs):
    """ HaloNet-H0. Halo attention in all stages as per the paper.
    NOTE: This runs very slowly!
    """
    return _create_byoanet('halonet_h0', b=8, h=3, rv=1.0, rb=0.5, l3=7,
                           pretrained=pretrained, **kwargs)


@register_model
def halonet_h1(pretrained=False, **kwargs):
    """ HaloNet-H1. Halo attention in all stages as per the paper.
    NOTE: This runs very slowly!
    """
    return _create_byoanet('halonet_h0', b=8, h=3, rv=1.0, rb=1.0, l3=10,
                           pretrained=pretrained, **kwargs)


@register_model
def halonet_h2(pretrained=False, **kwargs):
    """ HaloNet-H2. Halo attention in all stages as per the paper.
    NOTE: This runs very slowly!
    """
    return _create_byoanet('halonet_h0', b=8, h=3, rv=1.0, rb=1.25, l3=11,
                           pretrained=pretrained, **kwargs)


@register_model
def halonet_h3(pretrained=False, **kwargs):
    """ HaloNet-H3. Halo attention in all stages as per the paper.
    NOTE: This runs very slowly!
    """
    return _create_byoanet('halonet_h3', b=10, h=3, rv=1.0, rb=1.5, l3=12, df=1024,
                           pretrained=pretrained, **kwargs)


@register_model
def halonet_h4(pretrained=False, **kwargs):
    """ HaloNet-H4. Halo attention in all stages as per the paper.
    NOTE: This runs very slowly!
    """
    return _create_byoanet('halonet_h4', b=12, h=2, rv=1.0, rb=3.0, l3=12, df=1280,
                           pretrained=pretrained, **kwargs)


@register_model
def halonet_h5(pretrained=False, **kwargs):
    """ HaloNet-H5. Halo attention in all stages as per the paper.
    NOTE: This runs very slowly!
    """
    return _create_byoanet('halonet_h5', b=14, h=2, rv=2.5, rb=2.0, l3=23, df=1536,
                           pretrained=pretrained, **kwargs)


@register_model
def halonet_h6(pretrained=False, **kwargs):
    """ HaloNet-H6. Halo attention in all stages as per the paper.
    NOTE: This runs very slowly!
    """
    return _create_byoanet('halonet_h6', b=8, h=4, rv=3.0, rb=2.75, l3=24, df=1536,
                           pretrained=pretrained, **kwargs)


@register_model
def halonet_h7(pretrained=False, **kwargs):
    """ HaloNet-H7. Halo attention in all stages as per the paper.
    NOTE: This runs very slowly!
    """
    return _create_byoanet('halonet_h7', b=10, h=3, rv=4.0, rb=3.5, l3=26, df=2048,
                           pretrained=pretrained, **kwargs)
