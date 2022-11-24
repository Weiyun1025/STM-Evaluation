from timm.models import register_model
from ..meta_arch import MetaArch
from ..blocks.halonet import HaloBlockV2, HaloSingleResBlock

block_size_dict = {
    192: 12,
    224: 7,
    384: 12,
}

halo_size_dict = {
    192: 3,
    224: 3,
    384: 3,
}


@register_model
def unified_halonet_micro(pretrained=False, **kwargs):
    dims = [32 * 2 ** i for i in range(4)]
    depths = [2, 2, 9, 2]
    num_heads = [1, 2, 4, 8]

    img_size = kwargs.get('img_size', 224)
    block_size = block_size_dict[img_size]
    halo_size = halo_size_dict[img_size]

    model = MetaArch(img_size=img_size,
                     depths=depths,
                     dims=dims,
                     block_type=HaloBlockV2,
                     block_kwargs=dict(num_heads=num_heads,
                                       block_size=block_size,
                                       halo_size=halo_size,
                                       pos_embed_type='query_related'),
                     drop_path_rate=0,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def unified_halonet_tiny(pretrained=False, **kwargs):
    dims = [96 * 2 ** i for i in range(4)]
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]

    img_size = kwargs.get('img_size', 224)
    block_size = block_size_dict[img_size]
    halo_size = halo_size_dict[img_size]

    model = MetaArch(img_size=img_size,
                     depths=depths,
                     dims=dims,
                     block_type=HaloBlockV2,
                     block_kwargs=dict(num_heads=num_heads,
                                       block_size=block_size,
                                       halo_size=halo_size,
                                       pos_embed_type='query_related'),
                     drop_path_rate=0.2,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def unified_halonet_small(pretrained=False, **kwargs):
    dims = [96 * 2 ** i for i in range(4)]
    depths = [2, 2, 18, 2]
    num_heads = [3, 6, 12, 24]

    img_size = kwargs.get('img_size', 224)
    block_size = block_size_dict[img_size]
    halo_size = halo_size_dict[img_size]

    model = MetaArch(img_size=img_size,
                     depths=depths,
                     dims=dims,
                     block_type=HaloBlockV2,
                     block_kwargs=dict(num_heads=num_heads,
                                       block_size=block_size,
                                       halo_size=halo_size,
                                       pos_embed_type='query_related'),
                     drop_path_rate=0.3,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def unified_halonet_base(pretrained=False, **kwargs):
    dims = [128 * 2 ** i for i in range(4)]
    depths = [2, 2, 18, 2]
    num_heads = [4, 8, 16, 32]

    img_size = kwargs.get('img_size', 224)
    block_size = block_size_dict[img_size]
    halo_size = halo_size_dict[img_size]

    model = MetaArch(img_size=img_size,
                     depths=depths,
                     dims=dims,
                     block_type=HaloBlockV2,
                     block_kwargs=dict(num_heads=num_heads,
                                       block_size=block_size,
                                       halo_size=halo_size,
                                       pos_embed_type='query_related'),
                     drop_path_rate=0.5,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


@register_model
def unified_halo_v2_large(pretrained=False, **kwargs):
    dims = [192 * 2 ** i for i in range(4)]
    depths = [2, 2, 18, 2]
    num_heads = [6, 12, 24, 48]

    img_size = kwargs.get('img_size', 224)
    block_size = block_size_dict[img_size]
    halo_size = halo_size_dict[img_size]

    model = MetaArch(img_size=img_size,
                     depths=depths,
                     dims=dims,
                     block_type=HaloBlockV2,
                     block_kwargs=dict(num_heads=num_heads,
                                       block_size=block_size,
                                       halo_size=halo_size,
                                       pos_embed_type='query_related'),
                     drop_path_rate=0.2,
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model
