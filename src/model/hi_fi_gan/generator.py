import torch.nn as nn

from src.model.hi_fi_gan.mrf import MultiReceptiveField


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, upsample_kernel, mrf_kernels, mrf_dilations):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU()
        self.norm = nn.utils.weight_norm(
                nn.ConvTranspose1d(
                    in_channels,
                    in_channels // 2,
                    kernel_size=upsample_kernel,
                    stride=upsample_kernel // 2,
                    padding=(upsample_kernel - upsample_kernel // 2) // 2
                )
            )
        self.mrf = MultiReceptiveField(
                channels=in_channels // 2,
                kernel_sizes=mrf_kernels,
                dilation_configs=mrf_dilations
            )

    def forward(self, x):
        x = self.leaky_relu(x)
        x = self.norm(x)
        x = self.mrf(x)

        return x


class Generator(nn.Module):
    def __init__(self, input_channels, hidden_channels, upsample_kernels, mrf_kernels, mrf_dilations):
        super().__init__()
        self.initial_conv = nn.utils.weight_norm(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=hidden_channels,
                kernel_size=7,
                padding='same'
            )
        )
        self.upsample_blocks = nn.ModuleList([
            UpsampleBlock(
                in_channels=hidden_channels // (2 ** idx),
                upsample_kernel=upsample_kernels[idx],
                mrf_kernels=mrf_kernels,
                mrf_dilations=mrf_dilations
            )
            for idx in range(len(upsample_kernels))
        ])
        final_channels = hidden_channels // (2 ** len(upsample_kernels))
        self.final_conv = nn.Sequential(
            nn.LeakyReLU(),
            nn.utils.weight_norm(
                nn.Conv1d(
                    in_channels=final_channels,
                    out_channels=1,
                    kernel_size=7,
                    padding='same'
                )
            ),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial_conv(x)
        for block in self.upsample_blocks:
            x = block(x)
        x = self.final_conv(x)
        return x
