import torch
import torch.nn as nn
from torch.nn import functional as F

class BasicCNN(nn.Module):
    """ Neural network """
    def __init__(self):
        super().__init__()
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(32, 1, 3, 1, padding=1)
        # self.dropout1 = nn.Dropout2d(0.25)
        # self.dropout2 = nn.Dropout2d(0.5)

    def forward(self, x):
        """ Forward pass for your feedback prediction network. """
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = torch.sigmoid(out)
        out = out.squeeze(dim=1)
        return out

    def forward_with_activations(self, x):
        activations = []
        out = self.conv1(x)
        activations.append(out)
        out = F.relu(out)
        out = self.conv2(out)
        activations.append(out)
        out = F.relu(out)
        out = self.conv3(out)
        activations.append(out)
        out = torch.sigmoid(out)
        out = out.squeeze(dim=1)
        return out, activations
