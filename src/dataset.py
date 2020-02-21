import sys
import glob
import pydicom
import numpy as np
import pandas as pd
import torch
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


def run_length_decode(rle, height=1024, width=1024, fill_value=1):
    ''' https://www.kaggle.com/rishabhiitbhu/unet-with-resnet34-encoder-pytorch '''
    component = np.zeros((height, width), np.float32)
    component = component.reshape(-1)
    rle = np.array([int(s) for s in rle.strip().split(' ')])
    print(rle.shape, rle)
    rle = rle.reshape(-1, 2)
    start = 0
    for index, length in rle:
        start = start+index
        end = start+length
        component[start: end] = fill_value
        start = end
    component = component.reshape(width, height).T
    return component


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


class LungDataset(Dataset):
    ''' Dataset for training a model on a dataset. '''
    def __init__(self, data_path, mode=Mode.TRAIN):
        super().__init__()
        self.mode = mode
        if mode == Mode.TRAIN:
            train_files = sorted(glob.glob(f'{DATA_PATH}/train_images/*.dcm'))[:100]
            self.data = pd.read_csv(data_path, index_col='ImageId')

        elif mode == Mode.TEST:
            self.data = sorted(glob.glob(f'{DATA_PATH}/test_images/*.dcm'))[:100]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode == Mode.TRAIN:
            filename = f'{DATA_PATH}/train_images/{self.data.index[index]}.dcm'
            image = pydicom.read_file(filename).pixel_array
            annotations = self.data['EncodedPixels'][index].split()
            final_mask = np.zeros([1024, 1024])
            if annotations[0] != -1:
                for rle in annotations:
                    final_mask += run_length_decode(rle)
            final_mask = (final_mask >= 1).astype('float32') # for overlap cases
            return image, final_mask

        filename = f'{DATA_PATH}/test_images/{self.data[index]}.dcm'
        image = pydicom.read_file(filename).pixel_array
        return image
