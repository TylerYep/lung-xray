import torch
import torch.nn as nn
from .dice import DiceLoss


class NewMixedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCELoss()
        self.bce2 = nn.BCELoss()

    def forward(self, output, target):
        mask, result = output
        mask = mask.squeeze()
        binary_target = (torch.sum(target, (1, 2, 3)) > 0).float()
        loss = self.dice(mask, target) \
            + 3 * self.bce(mask, target.squeeze()) \
            + 0.1 * self.bce2(result.squeeze(), binary_target)
        return loss.mean()
