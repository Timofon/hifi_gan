import torch
from torch import nn

class DiscriminatorAdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ground_truths, predictions):
        total_loss = 0.0
        for ground_truth, prediction in zip(ground_truths, predictions):
            loss = torch.mean((ground_truth - 1) ** 2) + torch.mean(prediction ** 2)
            total_loss += loss
        return total_loss

class GeneratorAdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_outputs):
        total_loss = sum(torch.mean((pred_output - 1) ** 2) for pred_output in pred_outputs)
        return total_loss
