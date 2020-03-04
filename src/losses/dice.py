import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        return self.dice_loss(output, target)

    @staticmethod
    def dice_loss(output, target, eps=1):
        batch_size = output.shape[0]
        dice_target = target.reshape(batch_size, -1)
        dice_output = output.reshape(batch_size, -1)
        intersection = torch.sum(dice_output * dice_target, dim=1)
        union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1)
        loss = ((2 * intersection + eps) / (union + eps))
        return -torch.log(loss).mean()
