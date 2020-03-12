import torch
import torch.nn as nn

class BinaryMaskOnly(nn.Module):
    """ Neural network """
    def __init__(self, binary_path, mask_only_path):
        super().__init__()
        self.bin_model = torch.load(binary_path, map_location=torch.device('cpu'))
        self.mask_only_model = torch.load(mask_only_path, map_location=torch.device('cpu'))

    def forward(self, x):
        """ Forward pass for your feedback prediction network. """
        out = torch.cat([x] * 3, dim=1)
        is_binary = self.bin_model(out)
        if is_binary > 0.5:
            return torch.zeros(out.shape[1], out.shape[2])
        else:
            return self.mask_only_model(out)

        return out

    def forward_with_activations(self, x):
        pass
