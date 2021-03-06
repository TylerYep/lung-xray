import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class MaskRCNN(nn.Module):
    """ Neural network """
    def __init__(self, num_classes=2, hidden_size=256):
        super().__init__()
        self.model_ft = maskrcnn_resnet50_fpn(pretrained=True)
        in_features = self.model_ft.roi_heads.box_predictor.cls_score.in_features
        self.model_ft.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = self.model_ft.roi_heads.mask_predictor.conv5_mask.in_channels
        self.model_ft.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_size, num_classes)
        for param in self.model_ft.parameters():
            param.requires_grad = True

    def forward(self, x):
        """ Forward pass for your feedback prediction network. """
        out = self.model_ft(x)
        return out

    def forward_with_activations(self, x):
        pass
