import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary

from src import util
from src.args import init_pipeline
from src.dataset import load_test_data, INPUT_SHAPE
from src.models import UNet as Model
from src.losses import DiceLoss

if 'google.colab' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


def test_model(test_loader, model, criterion):
    model.eval()
    dice, correct = 0, 0
    with torch.no_grad():
        with tqdm(desc='Test', total=len(test_loader), ncols=120) as pbar:
            for data, target in test_loader:
                output = (model(data) > 0.5).float()
                dice += criterion(output, target).item()
                pbar.update()

    dice /= len(test_loader.dataset)

    print(f'\nTest set: Average Dice: {dice:.4f},',)


def main():
    args, device, checkpoint = init_pipeline()
    criterion = DiceLoss()
    test_loader = load_test_data(args, device)
    init_params = checkpoint.get('model_init', {})
    model = Model(*init_params).to(device)
    util.load_state_dict(checkpoint, model)
    torchsummary.summary(model, INPUT_SHAPE)

    test_model(test_loader, model, criterion)


if __name__ == '__main__':
    main()
