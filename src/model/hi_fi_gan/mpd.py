# import torch.nn as nn
# import torch.nn.functional as F


# class PeriodDiscriminator(nn.Module):
#     def __init__(self, period, kernel_size, stride, channels):
#         super().__init__()
#         self.period = period
#         in_channels = [1] + channels

#         conv_layers = []
#         for idx in range(len(channels)):
#             conv = nn.utils.weight_norm(
#                 nn.Conv2d(
#                     in_channels=in_channels[idx],
#                     out_channels=in_channels[idx + 1],
#                     kernel_size=(kernel_size, 1),
#                     stride=(stride, 1),
#                     padding=((kernel_size - 1) // 2, 0)
#                 )
#             )
#             conv_layers.append(nn.Sequential(conv, nn.LeakyReLU()))

#         conv_layers.append(
#             nn.Sequential(
#                 nn.utils.weight_norm(
#                     nn.Conv2d(
#                         in_channels=in_channels[-1],
#                         out_channels=1024,
#                         kernel_size=(5, 1),
#                         padding='same'
#                     )
#                 ),
#                 nn.LeakyReLU()
#             )
#         )

#         conv_layers.append(
#             nn.utils.weight_norm(
#                 nn.Conv2d(
#                     in_channels=1024,
#                     out_channels=1,
#                     kernel_size=(3, 1),
#                     padding='same'
#                 )
#             )
#         )
#         self.layers = nn.ModuleList(conv_layers)

#     def forward(self, x):
#         features_from_layers = []
#         if x.size(-1) % self.period != 0:
#             pad_size = self.period - x.size(-1) % self.period
#             x = F.pad(x, (0, pad_size), mode='reflect')
#         x = x.view(x.size(0), 1, x.size(-1) // self.period, self.period)
#         for layer in self.layers:
#             x = layer(x)
#             features_from_layers.append(x)
#         return x.flatten(-2, -1), features_from_layers[:-1]


# class MultiPeriodDiscriminator(nn.Module):
#     def __init__(self, periods, kernel_size, stride, channels):
#         super().__init__()
#         self.discriminators = nn.ModuleList([
#             PeriodDiscriminator(per, kernel_size, stride, channels) for per in periods
#         ])

#     def forward(self, x):
#         disc_outputs = []
#         disc_features = []
#         for disc in self.discriminators:
#             output, features_list = disc(x)
#             disc_outputs.append(output)
#             disc_features.append(features_list)
#         return disc_outputs, disc_features
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.utils.weight_norm(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
            ),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.block(x)

class PeriodicDiscriminator(nn.Module):
    def __init__(self, periods, kernel_size, stride, channel_sequence):
        super().__init__()
        self.period = periods

        self.channels = [1] + channel_sequence
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv_layers = nn.ModuleList()
        for idx in range(len(self.channels) - 1):
            self.conv_layers.append(
                Conv2dBlock(
                    in_channels=self.channels[idx],
                    out_channels=self.channels[idx + 1],
                    kernel_size=(self.kernel_size, 1),
                    stride=(self.stride, 1),
                    padding=((self.kernel_size - 1) // 2, 0)
                )
            )

        self.extra_conv = nn.Sequential(
            Conv2dBlock(
                in_channels=self.channels[-1],
                out_channels=1024,
                kernel_size=(5, 1),
                stride=(1, 1),
                padding=(2, 0)
            ),
            nn.utils.weight_norm(
                nn.Conv2d(
                    in_channels=1024,
                    out_channels=1,
                    kernel_size=(3, 1),
                    stride=(1, 1),
                    padding=(1, 0)
                )
            )
        )

    def forward(self, x):
        feature_maps = []

        if x.size(-1) % self.period != 0:
            pad_amount = self.period - (x.size(-1) % self.period)
            x = F.pad(x, (0, pad_amount), mode='reflect')

        b, c, t = x.size()
        x = x.view(b, c, t // self.period, self.period)

        for layer in self.conv_layers:
            x = layer(x)
            feature_maps.append(x)
        
        x = self.extra_conv(x)
        feature_maps.append(x)

        x = x.view(x.size(0), -1)
        return x, feature_maps[:-1]

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods, kernel_size, stride, channels):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            PeriodicDiscriminator(period, kernel_size, stride, channels)
            for period in periods
        ])

    def forward(self, x):
        outputs = []
        feature_maps_list = []

        for discriminator in self.discriminators:
            out, feats = discriminator(x)
            outputs.append(out)
            feature_maps_list.append(feats)

        return outputs, feature_maps_list
