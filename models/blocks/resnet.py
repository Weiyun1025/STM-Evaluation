from torch import nn
from timm.models.layers import create_classifier


class ResNetStem(nn.Module):
    def __init__(self, in_channels, embed_dim, norm_layer, act_layer, **kwargs):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=7, stride=2, padding=3, bias=False),
            norm_layer(embed_dim),
            act_layer(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        return self.stem(x)


class ResNetDownsampleLayer(nn.Module):
    def forward(self, x):
        return x


class ResNetHead(nn.Module):
    def __init__(self, in_features, num_classes, **kwargs):
        super().__init__()
        self.global_pool, self.fc = create_classifier(in_features, num_classes,
                                                      pool_type='avg')

    def forward(self, x):
        return self.fc(self.global_pool(x))
