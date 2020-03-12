import os
import torch

from src.models import Binary
from src.dataset import load_train_data
from src.args import init_pipeline
from src.losses import DiceLoss


def evaluate_model(model, loader, device, criterion=['dice']):
    ## TODO this is function I haven't finished yet
    model.eval()
    # dice, correct = 0, 0
    image_ids = []
    encoded_pixels = []
    totals = [0 for _ in range(len(criterion))]
    n = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            output = model(data)
            for i, c in enumerate(criterion):
                totals[i] += c(output, target)
                n += 1
    return [t/n for t in totals]


def evaluate():
    args, device, checkpoint = init_pipeline()
    criterion = DiceLoss()
    _, val_loader, _, _ = load_train_data(args, device)

    model = Binary()
    binary_checkpoint = 'AG'
    binary_path = os.path.join('checkpoints', binary_checkpoint, 'model_best.pth.tar')
    bin_model_weights = torch.load(binary_path)
    model.load_state_dict(bin_model_weights['state_dict'])
    evaluate_model(model, val_loader, device, criterion)


if __name__ == '__main__':
    evaluate()