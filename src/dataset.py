import sys
import glob
import csv
import pydicom
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from src.metric_tracker import Mode
if 'google.colab' in sys.modules:
    DATA_PATH = '/content'
else:
    DATA_PATH = 'data'

INPUT_SHAPE = (1, 28, 28)


def load_train_data(args):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    train_set = LungDataset(f"{DATA_PATH}/train-rle.csv")
    val_set = LungDataset()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.test_batch_size)
    return train_loader, val_loader, class_names, {}


def load_test_data(args):
    test_set = LungDataset()
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size)
    return test_loader


# def get_transforms():
#     return transforms.Compose([transforms.ToTensor(),
#                                transforms.Normalize((0.1307,), (0.3081,))])
    # norm = transforms.Compose([transforms.Grayscale(num_output_channels=3),
    #                            transforms.Resize(224),
    #                            transforms.ToTensor(),
    #                            transforms.Normalize((0.1307,), (0.3081,))])


# This function works
def run_length_decode(rle, height=1024, width=1024, fill_value=1):
    ''' https://www.kaggle.com/rishabhiitbhu/unet-with-resnet34-encoder-pytorch '''
    component = np.zeros((height, width), np.float32)
    rle = np.array([int(s) for s in rle.strip().split(' ')])
    if rle[0] == -1: return component
    component = component.reshape(-1)
    rle = rle.reshape(-1, 2)
    start = 0
    for index, length in rle:
        start = start+index
        end = start+length
        component[start: end] = fill_value
        start = end
    component = component.reshape(width, height).T
    return (component >= 1).astype('float32')

# TODO haven't tested
def run_length_encode(component):
    component = component.T.flatten()
    start = np.where(component[1:] > component[:-1])[0]+1
    end = np.where(component[:-1] > component[1:])[0]+1
    length = end-start
    rle = []
    for i in range(len(length)):
        if i == 0:
            rle.extend([start[0], length[0]])
        else:
            rle.extend([start[i]-end[i-1], length[i]])
    rle = ' '.join([str(r) for r in rle])
    return rle

def row_to_data(x, y):
    """
    Takes a row from train-rle.csv (parsed as x, y)
    and converts it into the image and the mask
    """
    return (
        pydicom.read_file(f'{DATA_PATH}/train_images/{x}.dcm').pixel_array/255
        , run_length_decode(y)
        )

def read_csv(fname):
    """
    Reads a csv file as a list of lists
    """
    with open(fname, newline='') as csvfile:
        return list(csv.reader(csvfile))

class LungDataset(Dataset):
    ''' Dataset for training a model on a dataset. '''
    def __init__(self, n=None, mode=Mode.TRAIN, lazy=True, mask_only=False):
        super().__init__()
        self.lazy = lazy
        self.mode = mode
        if mode == Mode.TRAIN:
            self.data = read_csv(f'{DATA_PATH}/train-rle.csv')[1:]
            if n is not None:
                self.data = self.data[:n]
            if mask_only:
                self.data = [x for x in self.data if x[1] != "-1"]
            if not self.lazy:
                self.xy = [row_to_data(x, y) for x, y in self.data]

        elif mode == Mode.TEST:
            self.data = sorted(glob.glob(f'{DATA_PATH}/test_images/*.dcm'))[:100]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode == Mode.TRAIN:
            if not self.lazy: 
                return self.xy[index]
            return row_to_data(*self.data[index])
