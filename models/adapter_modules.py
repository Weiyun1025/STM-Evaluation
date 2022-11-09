from functools import partial

import torch
from torch import nn
import torch.utils.checkpoint as cp
from timm.models.layers import DropPath
from .meta_arch import LayerNorm2d


class SpatialPriorModule(nn.Module):
    def __init__(self, dims, vit_dim=384, ratio=0.5, norm_layer=LayerNorm2d, act_layer=nn.GELU):
        super().__init__()

        stem = nn.Sequential(
            nn.Conv2d(3, int(dims[0] * ratio),
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            norm_layer(int(dims[0] * ratio)),
            act_layer(),
            nn.Conv2d(int(dims[0] * ratio), dims[0],
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            norm_layer(dims[0])
        )

        self.reduction = nn.ModuleList([stem])
        for i in range(1, len(dims)):
            self.reduction.append(nn.Sequential(
                nn.Conv2d(dims[i-1], dims[i],
                          kernel_size=(3, 3),
                          stride=(2, 2),
                          padding=(1, 1),
                          bias=False),
                norm_layer(dims[i]),
            ))

        self.mlp = nn.Linear(dims[i], vit_dim)
        self.cls = nn.Parameter(torch.randn(1, 1, vit_dim))

    def forward(self, x):
        for downsample in self.reduction:
            x = downsample(x)
        _, _, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1).contiguous()

        return torch.cat([self.cls.repeat(x.shape[0], 1, 1), self.mlp(x)], dim=1), H, W


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1,
        #                         bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        # B, L, C = x.shape
        # x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        # x = self.dwconv(x, H, W)
        # x = x.reshape(B, C, L).permute(0, 2, 1).contiguous()

        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Extractor(nn.Module):
    def __init__(self,
                 dim,
                 vit_dim=384,
                 num_heads=6,
                 with_cffn=True,
                 cffn_ratio=0.25,
                 drop=0.,
                 drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(vit_dim)
        self.feat_norm = norm_layer(dim)
        self.attn = nn.MultiheadAttention(embed_dim=vit_dim, num_heads=num_heads, kdim=dim, vdim=dim, batch_first=True)

        self.with_cp = with_cp
        self.with_cffn = with_cffn
        if with_cffn:
            self.ffn = ConvFFN(in_features=vit_dim, hidden_features=int(vit_dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(vit_dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, feat):

        def _inner_forward(query, feat):
            B, C, *_ = feat.shape
            feat = feat.reshape(B, C, -1).permute(0, 2, 1).contiguous()

            q = self.query_norm(query)
            kv = self.feat_norm(feat)

            attn = self.attn(query=q, key=kv, value=kv)[0]
            query = query + attn

            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query)))

            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class Injector(nn.Module):
    def __init__(self,
                 dim,
                 vit_dim=384,
                 num_heads=6,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 init_values=0.,
                 with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(vit_dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, kdim=vit_dim, vdim=vit_dim, batch_first=True)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, feat):

        def _inner_forward(query, feat):
            B, C, H, W = query.shape
            query = query.reshape(B, C, H * W).permute(0, 2, 1).contiguous()

            q = self.query_norm(query)
            kv = self.feat_norm(feat)

            attn = self.attn(query=q, key=kv, value=kv)[0]
            query = query + self.gamma * attn
            return query.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query
