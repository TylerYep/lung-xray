import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet as EffNet


class Binary(nn.Module):
    """ Neural network """
    def __init__(self, b=0):
        super().__init__()
        model_str = f'efficientnet-b{b}'
        self.efficient_net = EffNet.from_pretrained(model_str, num_classes=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """ Forward pass for your feedback prediction network. """
        out = torch.cat([x] * 3, dim=1)
        out = self.efficient_net(out)
        out = self.sig(out)
        return out

    def forward_with_activations(self, x):
        pass
