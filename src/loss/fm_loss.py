from torch import nn


class FeatureMatchingLoss(nn.Module):
    def __init__(self, multiplier):
        super().__init__()
        
        self.mult = multiplier
        self.loss = nn.L1Loss()

    def forward(self, ground_truth_feats_list, predictions_feats_list):
        loss = 0
        
        for ground_truth_discs, prediction_features_discs in zip(ground_truth_feats_list, predictions_feats_list):
            for ground_truth_features, prediction_features in zip(ground_truth_discs, prediction_features_discs):
                loss += self.loss(ground_truth_features, prediction_features)
        
        return self.mult * loss
