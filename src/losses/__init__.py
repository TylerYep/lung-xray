''' Imports all Loss functions. '''
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dice import DiceLoss
from .focal import FocalLoss
from .mixed import MixedLoss
from .new_mixed import NewMixedLoss


def get_loss_initializer(loss_fn):
    ''' Retrieves class initializer from its string name. '''
    if loss_fn == 'nn.CrossEntropyLoss()':
        return nn.CrossEntropyLoss()
    if loss_fn == 'F.nll_loss':
        return F.nll_loss
    return getattr(sys.modules[__name__], loss_fn)
