import os
import torch
import torch.nn as nn

from .unet import UNet
from .binary import Binary

class BinaryMaskOnly(nn.Module):
    """ Neural network """
    def __init__(self, binary_checkpoint, mask_checkpoint):
        super().__init__()
        binary_path = os.path.join('checkpoints', binary_checkpoint, 'model_best.pth.tar')
        mask_only_path = os.path.join('checkpoints', mask_checkpoint, 'model_best.pth.tar')
        bin_model_weights = torch.load(binary_path)#, map_location=torch.device('cpu'))
        mask_only_model_weights = torch.load(mask_only_path)#, map_location=torch.device('cpu'))
        self.bin_model = Binary()
        self.mask_only_model = UNet()
        self.bin_model.load_state_dict(bin_model_weights['state_dict'])
        self.mask_only_model.load_state_dict(mask_only_model_weights['state_dict'])

    def forward(self, x):
        """ Forward pass for your feedback prediction network. """
        out = torch.cat([x] * 3, dim=1)
        is_binary = self.bin_model(out)
        if is_binary > 0.5:
            return torch.zeros(out.shape[1], out.shape[2])
        return self.mask_only_model(out)

    def forward_with_activations(self, x):
        pass
