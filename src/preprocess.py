import sys
import pandas as pd
import pydicom
import csv
import random
from dataset import read_csv

if 'google.colab' in sys.modules:
    DATA_PATH = '/content'
else:
    DATA_PATH = 'data'


def train_test_split(array_list, train_frac=0.8, val_frac=0.1, test_frac=0.1):
    """Split data into three subsets (train, validation, and test).

    @param: array_list
            list of np.arrays/torch.Tensors
            we will split each entry accordingly
    @param train_frac: float [default: 0.8]
                       must be within (0.0, 1.0)
    @param val_frac: float [default: 0.8]
                     must be within [0.0, 1.0)
    @param train_frac: float [default: 0.8]
                       must be within (0.0, 1.0)
    """
    assert (train_frac + val_frac + test_frac) == 1.0
    train_list, val_list, test_list = [], [], []

    for arr in array_list:
        size = len(arr)
        start, mid = int(train_frac * size), int((train_frac + val_frac) * size)
        train_list.append(arr[:start])
        if val_frac > 0.0:
            val_list.append(arr[start:mid])
        test_list.append(arr[mid:])

    if val_frac > 0.0:
        return train_list, val_list, test_list

    return train_list, test_list


def preprocess():
    data = pd.read_csv(f'{DATA_PATH}/train-rle.csv')
    data.columns = ['image_id', 'encoded_pixels']
    new_data = pd.read_csv

    for id_ in data['image_id']:
        filename = f'{DATA_PATH}/train_images/{id_}.dcm'
        dicom_data = pydicom.read_file(filename)
        data['image_id'] = id_
        data['patient_id'] = dicom_data.PatientID
        data['patient_age'] = int(dicom_data.PatientAge)
        data['patient_sex'] = dicom_data.PatientSex
        data['pixel_spacing'] = dicom_data.PixelSpacing
    #     encoded_pixels_list = rles_df[rles_df['ImageId'] == dicom_data.SOPInstanceUID]['EncodedPixels'].values
    #     pneumothorax = any([encoded_pixels != ' -1' for encoded_pixels in encoded_pixels_list])
    #     data['encoded_pixels_list'] = encoded_pixels_list
    #     data['has_pneumothorax'] = pneumothorax
    #     data['rle_size'] = len(encoded_pixels_list)

    data = pd.join(data, new_data)
    print(new_data)

    return data

if __name__ == "__main__":
    train_val_split = [0.9, 0.1]
    data = read_csv('data/train-rle.csv')
    random.shuffle(data)
    train = data[:int(train_val_split[0]*len(data))]
    val = data[int(train_val_split[0]*len(data)):]
    print(len(train), len(val))
    with open('data/train.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["ImageId", "EncodedPixels"])
        writer.writerows(train)
    with open('data/val.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["ImageId", "EncodedPixels"])
        writer.writerows(val)
    # train_inds = []
