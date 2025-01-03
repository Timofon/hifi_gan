import torch.nn as nn
from typing import Dict
from src.model.hi_fi_gan.generator import Generator
from src.model.hi_fi_gan.mpd import MultiPeriodDiscriminator
from src.model.hi_fi_gan.msd import MultiScaleDiscriminator


class HiFiGAN(nn.Module):
    def __init__(self,
                 generator_config: Dict,
                 mpd_config: Dict,
                 msd_config: Dict):
        super().__init__()

        self.generator = Generator(**generator_config)
        self.mpd = MultiPeriodDiscriminator(**mpd_config)
        self.msd = MultiScaleDiscriminator(**msd_config)

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
