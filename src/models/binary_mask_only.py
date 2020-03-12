import os
import torch
import torch.nn as nn

from .unet import UNet
from .binary import Binary

class BinaryMaskOnly(nn.Module):
    """ Neural network """
    def __init__(self, binary_checkpoint=None, mask_checkpoint=None):
        super().__init__()
        self.bin_model = Binary()
        self.mask_only_model = UNet()
        if binary_checkpoint:
            binary_path = os.path.join('checkpoints', binary_checkpoint, 'model_best.pth.tar')
            bin_model_weights = torch.load(binary_path)
            self.bin_model.load_state_dict(bin_model_weights['state_dict'])
        if mask_checkpoint:
            mask_only_path = os.path.join('checkpoints', mask_checkpoint, 'model_best.pth.tar')
            mask_only_model_weights = torch.load(mask_only_path)
            self.mask_only_model.load_state_dict(mask_only_model_weights['state_dict'])

    def forward(self, x):
        """ Forward pass for your feedback prediction network. """
        is_binary = self.bin_model(x)
        result = self.mask_only_model(x)
        indices = torch.nonzero(is_binary < 0.5)
        result[indices] = torch.zeros(x.shape[1], x.shape[2])
        return result

    def forward_with_activations(self, x):
        pass


if __name__ == '__main__':
    BinaryMaskOnly()(torch.randn((10, 1, 256, 256)))