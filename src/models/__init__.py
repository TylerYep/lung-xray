import sys
from .rnn import BasicRNN
from .cnn import BasicCNN
from .unet import UNet
from .binary import Binary
from .binary_mask_only import BinaryMaskOnly
# from .efficient_net import EfficientNet

def get_model_initializer(model_name):
    ''' Retrieves class initializer from its string name. '''
    return getattr(sys.modules[__name__], model_name)
