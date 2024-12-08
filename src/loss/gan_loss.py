from torch import nn

from src.loss.adv_loss import DiscriminatorAdversarialLoss, GeneratorAdversarialLoss
from src.loss.fm_loss import FeatureMatchingLoss
from src.loss.mel_loss import MelSpectrogramLoss

class HiFiGANLoss(nn.Module):
    def __init__(self, feature_map_mult=2, mel_spec_mult=45):
        super().__init__()
        self.discriminator_loss = DiscriminatorAdversarialLoss()
        self.adversarial_loss = GeneratorAdversarialLoss()
        self.feature_matching_loss = FeatureMatchingLoss(feature_map_mult)
        self.mel_spectrogram_loss = MelSpectrogramLoss(mel_spec_mult)
