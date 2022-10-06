import torch
from torch import nn
from timm.models import register_model, create_model
from timm.models.layers import DropPath, LayerNorm2d, Mlp
from models.meta_arch import MetaArch
from models.blocks import halonet_github, halonet_timm


class HaloBlock(nn.Module):
    def __init__(self, dim, drop_path, layer_scale_init_value,
                 block_size, halo_size, stage, num_heads, halo_type,
                 mlp_ratio=4., drop=0., head_dim=None, act_layer=nn.GELU, norm_layer=LayerNorm2d,
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        if halo_type == 'timm':
            self.attn = halonet_timm.HaloAttention(dim=dim,
                                                   dim_head=head_dim,
                                                   num_heads=num_heads[stage],
                                                   block_size=block_size,
                                                   halo_size=halo_size)

        else:
            self.attn = halonet_github.HaloAttention(dim=dim,
                                                     dim_head=head_dim,
                                                     heads=num_heads[stage],
                                                     block_size=block_size,
                                                     halo_size=halo_size)

        self.norm1 = norm_layer(dim)
        self.gamma_1 = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1, 1)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer,
                       drop=drop)

        self.gamma_2 = nn.Parameter(layer_scale_init_value * torch.ones((1, 1, 1, dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        # shape: B, C, H, W
        shortcut = x
        x = self.attn(self.norm1(x))

        if self.gamma_1 is not None:
            x = self.gamma_1 * x

        x = shortcut + self.drop_path(x)

        # FFN
        shortcut = x
        x = self.mlp(self.norm2(x).permute(0, 2, 3, 1))

        if self.gamma_2 is not None:
            x = self.gamma_2 * x

        x = x.permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(x)

        return x


@register_model
def halonet_tiny(pretrained=False, halo_type='github', **kwargs):
    dims = [96 * 2 ** i for i in range(4)]
    depths = [2, 2, 18, 2]
    num_heads = [3, 6, 12, 24]
    block_size = 7
    halo_size = 3

    model = MetaArch(img_size=224,
                     depths=depths,
                     dims=dims,
                     block_type=HaloBlock,
                     block_kwargs=dict(num_heads=num_heads,
                                       block_size=block_size,
                                       halo_size=halo_size,
                                       halo_type=halo_type),
                     **kwargs)

    if pretrained:
        raise NotImplementedError()

    return model


def test(halo_type, x_input, y_gold):
    model = create_model('halonet_tiny', halo_type=halo_type)
    criterion = nn.CrossEntropyLoss()

    y_pred = model(x_input)
    loss = criterion(y_pred, y_gold)
    loss.backward()


if __name__ == '__main__':
    x = torch.randn(16, 3, 224, 224)
    y = torch.nn.functional.softmax(torch.randn(16, 1000), dim=-1)
    test('github', x, y)
    test('timm', x, y)
