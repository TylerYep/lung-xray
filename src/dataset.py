import sys
import glob
import pandas as pd
import torch
import pydicom
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
            masks = self.data['EncodedPixels'][index]
            return image, masks

        filename = f'{DATA_PATH}/test_images/{self.data[index]}.dcm'
        image = pydicom.read_file(filename).pixel_array
        return image
