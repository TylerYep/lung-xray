from src.dataset import LungDataset, DATA_PATH
import matplotlib.pyplot as plt

def explore():
    dataset = LungDataset(f"{DATA_PATH}/train-rle.csv")
    print(dataset[1])
    plot_pixel_array(dataset[1][0])


def show_dcm_info(dataset):
    print("Filename.........:", file_path)
    print("Storage type.....:", dataset.SOPClassUID)

    pat_name = dataset.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name......:", display_name)
    print("Patient id..........:", dataset.PatientID)
    print("Patient's Age.......:", dataset.PatientAge)
    print("Patient's Sex.......:", dataset.PatientSex)
    print("Modality............:", dataset.Modality)
    print("Body Part Examined..:", dataset.BodyPartExamined)
    print("View Position.......:", dataset.ViewPosition)

    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
            rows=rows, cols=cols, size=len(dataset.PixelData)))
        if 'PixelSpacing' in dataset:
            print("Pixel spacing....:", dataset.PixelSpacing)


def plot_pixel_array(arr, figsize=(10,10)):
    """arr should be a numpy array"""
    plt.figure(figsize=figsize)
    plt.imshow(arr, cmap=plt.cm.bone)
    plt.show()


if __name__ == '__main__':
    explore()