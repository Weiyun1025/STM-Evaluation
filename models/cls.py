import torch
from torch import nn
from timm.models.layers import Mlp, DropPath


class ClassBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 mlp_ratio=4.,
                 layer_scale_init_value=1e-6,
                 drop_path=0.):
        super().__init__()

        self.norm_1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim,
                                          num_heads=num_heads,
                                          batch_first=True)
        self.gamma_1 = nn.Parameter(layer_scale_init_value * torch.ones((1, 1, dim))) if layer_scale_init_value > 0 else None

        self.norm_2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       out_features=dim)
        self.gamma_2 = nn.Parameter(layer_scale_init_value * torch.ones((1, 1, dim))) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, q, x):
        shortcut = q
        x = self.norm_1(x)
        q = self.attn(query=q, key=x, value=x)[0]
        if self.gamma_1 is not None:
            q = self.gamma_1 * q
        q = shortcut + self.drop_path(q)

        shortcut = q
        q = self.mlp(self.norm_2(q))
        if self.gamma_2 is not None:
            q = self.gamma_2 * q
        q = shortcut + self.drop_path(q)

        # bsz, query_len, num_classes
        return q


class MultiLayerClassBlock(nn.Module):
    def __init__(self,
                 dim,
                 layer=1,
                 query_len=5,
                 num_heads=8,
                 num_classes=1000,
                 mlp_ratio=4.,
                 layer_scale_init_value=1e-6,
                 drop_path=0.):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, query_len, dim))

        self.blks = nn.ModuleList()
        for _ in range(layer):
            self.blks.append(ClassBlock(dim=dim,
                                        num_heads=num_heads,
                                        mlp_ratio=mlp_ratio,
                                        layer_scale_init_value=layer_scale_init_value,
                                        drop_path=drop_path))

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(in_features=dim, out_features=num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = x.flatten(2).permute(0, 2, 1).contiguous()
        q = self.q.repeat(B, 1, 1)

        for blk in self.blks:
            q = blk(q, x)

        # bsz, query_len, num_classes
        return self.head(self.norm(q))


class ClassBlockV2(nn.Module):
    def __init__(self, dim, query_len=5, num_heads=8, num_classes=1000, mlp_ratio=1.5):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, query_len, dim))

        self.norm_1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim,
                                          num_heads=num_heads,
                                          batch_first=True)

        self.norm_2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = x.flatten(2).permute(0, 2, 1).contiguous()
        x = self.norm_1(x)
        q = self.q.repeat(B, 1, 1)

        x = self.attn(query=q, key=x, value=x)[0]
        x = self.mlp(self.norm_2(x))

        # bsz, query_len, num_classes
        return x


class ClassBlockV3(nn.Module):
    def __init__(self, dim, query_len=5, num_heads=8, num_classes=1000, mlp_ratio=1.5):
        super().__init__()
        self.conv_head = nn.Sequential(
            nn.Conv2d(dim, int(dim * mlp_ratio), 1, 1, 0, bias=False),
            nn.BatchNorm2d(int(dim * mlp_ratio)),
            nn.GELU(),
        )

        dim = int(dim * mlp_ratio)
        self.q = nn.Parameter(torch.randn(1, query_len, dim))
        self.attn = nn.MultiheadAttention(embed_dim=dim,
                                          num_heads=num_heads,
                                          batch_first=True)

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.conv_head(x)

        B = x.shape[0]
        x = x.flatten(2).permute(0, 2, 1).contiguous()
        q = self.q.repeat(B, 1, 1)
        x = self.attn(query=q, key=x, value=x)[0]

        x = self.head(self.norm(x))
        return x


class GAPBlock(nn.Module):
    def __init__(self, dim, num_classes=1000, mlp_ratio=1.5):
        super().__init__()
        self.conv_head = nn.Sequential(
            nn.Conv2d(dim, int(dim * mlp_ratio), 1, 1, 0, bias=False),
            nn.BatchNorm2d(int(dim * mlp_ratio)),
            nn.GELU(),
        )

        dim = int(dim * mlp_ratio)
        self.avg_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            # nn.LayerNorm(dim),
        )

        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.conv_head(x)
        x = self.avg_head(x)
        x = self.head(x)

        # bsz, 1, num_classes
        return x.unsqueeze(1)


class GAPBlockV2(nn.Module):
    def __init__(self, dim, num_classes=1000, mlp_ratio=1.5):
        super().__init__()
        self.avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.LayerNorm(dim)
        )

        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=num_classes)

    def forward(self, x):
        x = self.avg(x)
        x = self.mlp(x)

        # bsz, 1, num_classes
        return x.unsqueeze(1)
