import sys
import glob
import csv
import pydicom
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from PIL import Image

if 'google.colab' in sys.modules:
    DATA_PATH = '/content'
else:
    DATA_PATH = 'data'

INPUT_SHAPE = (1, 256, 256)
CLASS_LABELS = []


def get_collate_fn(device):
    return lambda x: map(lambda b: b.to(device), default_collate(x))


def load_train_data(args, device):
    collate_fn = get_collate_fn(device)
    train_set = LungDataset('train', n=args.n, img_dim=args.img_dim)
    val_set = LungDataset('val', n=args.n, img_dim=args.img_dim)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_set,
                            batch_size=args.batch_size,
                            collate_fn=collate_fn)
    return train_loader, val_loader, len(train_set), len(val_set), {}


def load_test_data(args):
    test_set = LungDataset('test')
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size)
    return test_loader, len(test_set)


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


def mask2rle(img, width=1024, height=1024):
    rle = []
    last_color = 0
    current_pixel = 0
    run_start, run_length = -1, 0

    for x in range(width):
        for y in range(height):
            current_color = img[x][y]
            if current_color != last_color:
                if current_color == 1:
                    run_start = current_pixel
                    run_length = 1
                else:
                    rle.append(str(run_start))
                    rle.append(str(run_length))
                    run_start = -1
                    run_length = 0
                    current_pixel = 0
            elif run_start > -1:
                run_length += 1
            last_color = current_color
            current_pixel += 1
    return " " + " ".join(rle)


def row_to_data(id_, rle):
    """
    Takes a row from train-rle.csv (parsed as id, rle)
    and converts it into the image and the mask
    """
    filename = f'{DATA_PATH}/train_images/{id_}.dcm'
    img = pydicom.read_file(filename).pixel_array / 255
    mask = rle2mask(rle)
    return img[:, :, None].astype("float32"), mask


def image_transform(img_dim):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize((img_dim, img_dim)),
        transforms.ToTensor(),
    ])


def mask_transform(mask_dim):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize((mask_dim, mask_dim), Image.NEAREST),
        transforms.ToTensor(),
    ])


class LungDataset(Dataset):
    ''' Dataset for training a model on a dataset. '''
    def __init__(self, mode, n=None, lazy=True, mask_only=False
                , img_transform=image_transform, mask_transform=mask_transform
                , img_dim=128):
        super().__init__()
        self.lazy = lazy
        self.mode = mode
        self.img_transform = img_transform(img_dim)
        self.mask_transform = mask_transform(img_dim)
        self.fname = f'{DATA_PATH}/{mode}.csv'
        assert bool(mask_transform) == bool(img_transform)

        if mode == 'test':
            self.data = sorted(glob.glob(f'{DATA_PATH}/test_images/*.dcm'))
            if n is not None:
                self.data = self.data[:n]

        else: # train or val
            with open(self.fname, newline='') as csvfile:
                self.data = list(csv.reader(csvfile))[1:]
                if n is not None:
                    self.data = self.data[:n]
                if mask_only:
                    self.data = [(id_, rle) for (id_, rle) in self.data if rle != "-1"]
                if not self.lazy:
                    self.data = [row_to_data(id_, rle) for id_, rle in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode in ('train', 'val'):
            if self.lazy:
                img, mask = row_to_data(*self.data[index])
            else:
                img, mask = self.data[index]

            if self.img_transform:
                img = self.img_transform(img)
                mask = self.mask_transform(mask)

            return img, mask
        else:
            filename = self.data[index]
            dicom_data = pydicom.read_file(filename)
            image_id = dicom_data.SOPInstanceUID
            img = dicom_data.pixel_array / 255
            img = img[:, :, None].astype("float32")
            if self.img_transform:
                img = self.img_transform(img)
            return img, image_id


if __name__ == '__main__':
    train_set = LungDataset('train', mask_only=True)
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    print(len(train_set))
    print(train_loader.batch_size * len(train_loader))
