import pandas as pd
import csv
from sklearn.model_selection import train_test_split
import pydicom
import numpy as np

x_columns = [
    "Age"
    , "Sex"
    , "Modality"
    , "BodyPartExamined"
    , "ViewPosition"
]

def pydicom_feats(id_):
    filename = f'data/train_images/{id_}.dcm'
    f = pydicom.read_file(filename)
    return [
        int(f.PatientAge)
        , f.PatientSex
        , f.Modality
        , f.BodyPartExamined
        , f.ViewPosition
    ]
    

def load_data():
    with open("data/train-rle.csv", newline='') as csvfile:
        data = list(csv.reader(csvfile))[1:]
    data = np.array([[pydicom_feats(x) + [int(y != '-1')] for x, y in data]]).squeeze()
    X = data[:, :-1]
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    # TODO figure out if we need to use label encoder for categorical data
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    load_data()