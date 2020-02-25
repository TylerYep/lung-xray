import sys
import glob
import csv
import pydicom
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
if 'google.colab' in sys.modules:
    DATA_PATH = '/content'
else:
    DATA_PATH = 'data'

INPUT_SHAPE = (1, 1024, 1024)
CLASS_LABELS = []


def get_collate_fn(device):
    return lambda x: map(lambda b: b.to(device), default_collate(x))


def load_train_data(args, device):
    collate_fn = get_collate_fn(device)
    train_set = LungDataset('train', n=100, mask_only=True)
    val_set = LungDataset('val')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=collate_fn)
    return train_loader, val_loader, {}


def load_test_data(args, device):
    collate_fn = get_collate_fn(device)
    test_set = LungDataset('test')
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, collate_fn=collate_fn)
    return test_loader


# This function works
def rle2mask(rle, height=1024, width=1024, fill_value=1):
    ''' https://www.kaggle.com/rishabhiitbhu/unet-with-resnet34-encoder-pytorch '''
    component = np.zeros((height, width), np.float32)
    rle = np.array([int(s) for s in rle.strip().split(' ')])
    if rle[0] == -1:
        return component

    component = component.reshape(-1)
    rle = rle.reshape(-1, 2)
    start = 0
    for index, length in rle:
        start = start+index
        end = start+length
        component[start: end] = fill_value
        start = end
    component = component.reshape(width, height).T
    return (component >= 1).astype(np.float32)


# TODO haven't tested
def mask2rle(component):
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


def row_to_data(id_, rle):
    """
    Takes a row from train-rle.csv (parsed as id, rle)
    and converts it into the image and the mask
    """
    filename = f'{DATA_PATH}/train_images/{id_}.dcm'
    img = pydicom.read_file(filename).pixel_array / 255
    mask = rle2mask(rle)
    return img[None, :, :].astype("float32"), mask


def read_csv(filename, has_masks=True):
    """
    Reads a csv file as a list of lists
    """
    with open(filename, newline='') as csvfile:
        return list(csv.reader(csvfile))[1:]


class LungDataset(Dataset):
    ''' Dataset for training a model on a dataset. '''
    def __init__(self, mode, n=None, lazy=True, mask_only=False):
        super().__init__()
        self.lazy = lazy
        self.mode = mode
        if mode == 'train' or mode == 'val':
            self.data = read_csv(f'{DATA_PATH}/train-rle.csv')
            if n is not None:
                self.data = self.data[:n]
            if mask_only:
                self.data = [(id_, rle) for (id_, rle) in self.data if rle != "-1"]
            if not self.lazy:
                self.xy = [row_to_data(id_, rle) for id_, rle in self.data]

        elif mode == 'test':
            self.data = sorted(glob.glob(f'{DATA_PATH}/test_images/*.dcm'))
            if n is not None:
                self.data = self.data[:n]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode in ('train', 'val'):
            if not self.lazy:
                return self.xy[index]
            return row_to_data(*self.data[index])
        else:
            raise NotImplementedError


if __name__ == '__main__':
    # z = read_csv(f'{DATA_PATH}/train-rle.csv')
    # print(x)
    z = LungDataset('train')
    print(z[0][0].shape, z[0][1].shape)
