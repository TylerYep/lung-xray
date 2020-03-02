import matplotlib.pyplot as plt
from src.viz import plot_pixel_array, plot_with_mask

from src.dataset import LungDataset, DATA_PATH

def explore():
    dataset = LungDataset(f"{DATA_PATH}/train-rle.csv")
    print(dataset[1])
    plot_pixel_array(dataset[1][0])
    plot_pixel_array(dataset[1][1])


# def show_dcm_info(dataset):
#     print("Filename.........:", file_path)
#     print("Storage type.....:", dataset.SOPClassUID)

#     pat_name = dataset.PatientName
#     display_name = pat_name.family_name + ", " + pat_name.given_name
#     print("Patient's name......:", display_name)
#     print("Patient id..........:", dataset.PatientID)
#     print("Patient's Age.......:", dataset.PatientAge)
#     print("Patient's Sex.......:", dataset.PatientSex)
#     print("Modality............:", dataset.Modality)
#     print("Body Part Examined..:", dataset.BodyPartExamined)
#     print("View Position.......:", dataset.ViewPosition)

#     if 'PixelData' in dataset:
#         rows = int(dataset.Rows)
#         cols = int(dataset.Columns)
#         print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
#             rows=rows, cols=cols, size=len(dataset.PixelData)))
#         if 'PixelSpacing' in dataset:
#             print("Pixel spacing....:", dataset.PixelSpacing)




if __name__ == '__main__':
    explore()
