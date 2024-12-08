from torch import nn

class MelSpectrogramLoss(nn.Module):
    def __init__(self, multiplier):
        super().__init__()
        
        self.mult = multiplier
        self.loss = nn.L1Loss()

    def forward(self, ground_truth_spectrogram, prediction_spectrogram):
        return self.mult * self.loss(ground_truth_spectrogram, prediction_spectrogram)
