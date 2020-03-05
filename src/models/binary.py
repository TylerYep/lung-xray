import torch
import torch.nn as nn
import torchvision.models as models

class Binary(nn.Module):
    """ Neural network """
    def __init__(self):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.lin = nn.Linear(1000, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """ Forward pass for your feedback prediction network. """
        out = torch.cat([x] * 3, dim=1)
        out = self.resnet18(out)
        out = self.sig(self.lin(out))
        return out

    def forward_with_activations(self, x):
        pass
