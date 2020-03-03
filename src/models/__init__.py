from .rnn import BasicRNN
from .cnn import BasicCNN
from .unet import UNet

MODEL_DICT = {
    'cnn':BasicCNN
    , 'unet':UNet
}
# from .efficient_net import EfficientNet
