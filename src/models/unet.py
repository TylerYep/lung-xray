import torch
import torch.nn as nn
from torch.nn import functional as F
import segmentation_models_pytorch as smp

class UNet(nn.Module):
    """ Neural network """
    def __init__(self):
        super().__init__()
        self.unet = smp.Unet(encoder_name="resnet18",
                             encoder_weights='imagenet',
                             activation='sigmoid',
                             classes=1)

    def forward(self, x):
        """ Forward pass for your feedback prediction network. """
        out = torch.cat([x] * 3, dim=1)
        out = self.unet(out)
        return out

    def forward_with_activations(self, x):
        pass
