import torchvision.models as models

from src import util
from src.args import init_pipeline
from src.dataset import load_train_data, CLASS_LABELS
from src.models import BasicCNN as Model

from src.visualizations import *


def main():
    args, device, checkpoint = init_pipeline()
    train_loader, val_loader, _, _, init_params = load_train_data(args, device)
    model = Model(*init_params).to(device)
    # util.load_state_dict(checkpoint, model)

    visualize(model, val_loader)


def visualize(model, loader, run_name='', metrics=None):
    data, target = next(iter(loader))
    view_input(data, target, run_name)
    data, target = next(iter(loader))
    compute_activations(model, data, target, run_name)


def visualize_trained(model, loader, run_name='', metrics=None):
    data, target = next(iter(loader))
    make_fooling_image(model, data[5], target[5], CLASS_LABELS, target[9], run_name)
    data, target = next(iter(loader))
    show_saliency_maps(model, data, target, CLASS_LABELS, run_name)
    data, target = next(iter(loader))
    create_class_visualization(model, data, CLASS_LABELS, target[1], run_name)


if __name__ == '__main__':
    main()
