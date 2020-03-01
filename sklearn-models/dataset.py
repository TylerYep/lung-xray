import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pydicom
import numpy as np
from tqdm import tqdm

x_columns = [
    "Age"
    , "Sex"
    , "Modality"
    , "BodyPartExamined"
    , "ViewPosition"
]

def pydicom_features(id_):
    filename = f'../data/train_images/{id_}.dcm'
    f = pydicom.read_file(filename)
    return [
        int(f.PatientAge)
        , 1 if f.PatientSex == 'M' else 0
        , 1 if f.Modality == 'CR' else 0
        # , f.BodyPartExamined
        , 1 if f.ViewPosition == 'AP' else 0
    ]


def load_data():
    with open("../data/train-rle.csv", newline='') as csvfile:
        data = list(csv.reader(csvfile))[1:]
    features = [pydicom_features(x) + [int(y != '-1')] for x, y in tqdm(data)]
    data = np.array([features]).squeeze()
    X = data[:, :-1]
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    print(X_train)
    # TODO figure out if we need to use label encoder for categorical data
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    load_data()