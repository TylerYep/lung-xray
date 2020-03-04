import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        return self.focal_loss(output, target)

    @staticmethod
    def focal_loss(output, target, eps=1):
        batch_size = output.shape[0]
        dice_target = target.reshape(batch_size, -1)
        dice_output = output.reshape(batch_size, -1)
       
        return loss
