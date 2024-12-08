import torch.nn as nn
import math


class ScaleDiscriminator(nn.Module):
    def __init__(self, scale, conv_kernel_sizes, conv_strides, groupings, feature_channels):
        super(ScaleDiscriminator, self).__init__()
        self.scale = scale

        self.normalize = nn.utils.spectral_norm if self.scale == 1 else nn.utils.weight_norm

        if self.scale == 1:
            self.pool = nn.Identity()
        else:
            num_pools = int(math.log2(self.scale))
            self.pool = nn.Sequential(*[nn.AvgPool1d(4, 2, 2) for _ in range(num_pools)])
        
        self.layers = self.build_layers(conv_kernel_sizes, conv_strides, groupings, feature_channels)

    def build_layers(self, kernel_sizes, strides, groups, channels):
        layers = []
        input_channels = [1] + channels

        for idx in range(len(kernel_sizes)):
            conv = self.normalize(
                nn.Conv1d(
                    in_channels=input_channels[idx],
                    out_channels=input_channels[idx + 1],
                    kernel_size=kernel_sizes[idx],
                    stride=strides[idx],
                    groups=groups[idx],
                    padding=(kernel_sizes[idx] - 1) // 2
                )
            )
            layers.append(nn.Sequential(conv, nn.LeakyReLU()))
        
        final_conv = self.normalize(
            nn.Conv1d(
                in_channels=input_channels[-1],
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )
        layers.append(final_conv)
        return nn.ModuleList(layers)

    def forward(self, x):
        feature_maps = []
        x = self.pool(x)
        for layer in self.layers:
            x = layer(x)
            feature_maps.append(x)
        return x, feature_maps[:-1]



class MultiScaleDiscriminator(nn.Module):
    def __init__(self, scales, kernel_sizes, strides, groups, channels):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(scale, kernel_sizes, strides, groups, channels)
            for scale in scales
        ])

    def forward(self, x):
        outputs = []
        feature_lists = []
        for discriminator in self.discriminators:
            out, features = discriminator(x)
            outputs.append(out)
            feature_lists.append(features)
        return outputs, feature_lists
