import torchvision.models as models
import matplotlib.pyplot as plt

from src import util
from src.args import init_pipeline
from src.dataset import load_train_data, CLASS_LABELS
from src.models import UNet as Model

from src.visualizations import *


def plot_sbs(image, output, target):
    _, axs = plt.subplots(3)
    axs[0].imshow(image)
    axs[1].imshow(output)
    axs[2].imshow(target)
    plt.show()


def plot_pixel_array(arr, figsize=(10, 10)):
    """arr should be a numpy array"""
    arr = arr.squeeze()
    plt.figure(figsize=figsize)
    plt.imshow(arr, cmap=plt.cm.bone)
    plt.show()


def plot_with_mask(im, mask, figsize=(10, 10)):
    im = im.squeeze()
    mask = mask.squeeze()
    plt.figure(figsize=figsize)
    plt.imshow(im + mask, cmap=plt.cm.bone)
    if mask.sum() == 0:
        plt.title('no mask')
    plt.show()


def _main(args, device, checkpoint):
    train_loader, _, _, _, init_params = load_train_data(args, device)
    model = Model(*init_params).to(device)
    util.load_state_dict(checkpoint, model)
    visualize(model, train_loader)


def main():
    args, device, checkpoint = init_pipeline()
    train_loader, _, _, _, init_params = load_train_data(args, device)
    model = Model(*init_params).to(device)
    util.load_state_dict(checkpoint, model)
    visualize(model, train_loader)


def visualize(model, loader, run_name='', metrics=None):
    data, target = next(iter(loader))
    view_input(data, target, run_name)
    data, target = next(iter(loader))
    compare_prediction(model, data, target, run_name)


def visualize_trained(model, loader, run_name='', metrics=None):
    data, target = next(iter(loader))
    make_fooling_image(model, data[5], target[5], CLASS_LABELS, target[9], run_name)
    data, target = next(iter(loader))
    show_saliency_maps(model, data, target, CLASS_LABELS, run_name)
    data, target = next(iter(loader))
    create_class_visualization(model, data, CLASS_LABELS, target[1], run_name)


if __name__ == '__main__':
    main()
