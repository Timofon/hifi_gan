import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self,
                 channels,
                 kernel_size,
                 dilation_patterns):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.LeakyReLU(),
                    nn.utils.weight_norm(
                        nn.Conv1d(
                            channels, channels,
                            kernel_size=kernel_size,
                            dilation=dilation,
                            padding='same'
                        )
                    )
                )
                for dilation in pattern
            ])
            for pattern in dilation_patterns
        ])

    def forward(self, x):
        output = x
        for layers in self.conv_layers:
            residual = output
            for conv in layers:
                output = conv(output)
            output += residual
        return output
        

class MultiReceptiveField(nn.Module):
    def __init__(self,
                 channels,
                 kernel_sizes,
                 dilation_configs):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels, k_size, dilation_cfg)
            for k_size, dilation_cfg in zip(kernel_sizes, dilation_configs)
        ])


    def forward(self, x):
        outputs = self.res_blocks[0](x)
        for res_block in self.res_blocks[1:]:
            outputs += res_block(x)
        return outputs
