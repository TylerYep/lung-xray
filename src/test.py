import sys
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary

from src import util
from src.args import init_pipeline
from src.dataset import load_test_data, INPUT_SHAPE, mask2rle
from src.models import UNet as Model
from src.losses import DiceLoss

if 'google.colab' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


def test_model(test_loader, model, criterion, device):
    model.eval()
    # dice, correct = 0, 0
    image_ids = []
    encoded_pixels = []
    df = pd.read_csv('data/sample_submission.csv')
    with torch.no_grad():
        with tqdm(desc='Test', total=len(test_loader), ncols=120) as pbar:
            for data, image_id in test_loader:
                data = data.to(device)
                image_ids.append(image_id)
                output = model(data)
                output = (output > 0.5).float()
                output = output.detach().cpu().numpy().squeeze()
                no_pneumothorax = output.sum() == 0
                for pred in output:
                    if pred.shape != (1024, 1024):
                        pred = cv2.resize(pred, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
                    if no_pneumothorax:
                        encoded_pixels.append('-1')
                    else:
                        encoded_pixels.append(mask2rle(pred))

                # dice += criterion(output, target).item()
                pbar.update()

    # dice /= len(test_loader.dataset)
    # print(f'\nTest set: Average Dice: {dice:.4f},')
    df['ImageId'] = image_ids
    df['EncodedPixels'] = encoded_pixels
    df.to_csv('submission.csv', columns=['ImageId', 'EncodedPixels'], index=False)


def test():
    args, device, checkpoint = init_pipeline()
    criterion = DiceLoss()
    test_loader, len_test = load_test_data(args)
    init_params = checkpoint.get('model_init', {})
    model = Model(*init_params).to(device)
    util.load_state_dict(checkpoint, model)
    torchsummary.summary(model, INPUT_SHAPE)

    test_model(test_loader, model, criterion, device)
