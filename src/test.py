import sys
from collections import defaultdict
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary

from src import util
from src.args import init_pipeline
from src.dataset import load_test_data, INPUT_SHAPE, mask2rle
from src.models import get_model_initializer
from src.losses import DiceLoss
from src.metrics import dice

if 'google.colab' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

def model_on_loader(model, loader, device, criterion=[dice]):
    ## TODO this is function I haven't finished yet
    model.eval()
    # dice, correct = 0, 0
    image_ids = []
    encoded_pixels = []
    totals = [0 for _ in range(len(criterion)]
    n = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            output = model(data)
            for i, c in enumerate(criterion):
                total[i] += c(output, target)
                n += 1
    return [t/n for t in totals]            


def test_model(test_loader, model, criterion, device):
    model.eval()
    # dice, correct = 0, 0
    encoded_pixels = defaultdict(lambda: '-1')
    df = pd.read_csv('data/sample_submission.csv')
    df = df.drop_duplicates('ImageId', keep='last').reset_index(drop=True)
    with torch.no_grad():
        with tqdm(desc='Test', total=len(test_loader), ncols=120) as pbar:
            for data, image_id in test_loader:
                data = data.to(device)
                output = model(data)
                output = (output > 0.5).float()
                output = output.detach().cpu().numpy().squeeze()

                for img_id, pred in zip(image_id, output):
                    no_pneumothorax = pred.sum() == 0
                    if no_pneumothorax:
                        encoded_pixels[img_id] = '-1'
                    else:
                        if pred.shape != (1024, 1024):
                            pred = cv2.resize(pred, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
                        encoded_pixels[img_id] = mask2rle(pred)

                # dice += criterion(output, target).item()
                pbar.update()

    # dice /= len(test_loader.dataset)
    # print(f'\nTest set: Average Dice: {dice:.4f},')
    new_df = pd.DataFrame(encoded_pixels.items(), columns=['ImageId', 'EncodedPixels'])
    print(new_df)
    new_df.to_csv('submission.csv', columns=['ImageId', 'EncodedPixels'], index=False)


def test():
    args, device, checkpoint = init_pipeline()
    criterion = DiceLoss()
    test_loader, len_test = load_test_data(args)
    init_params = checkpoint.get('model_init', {})
    model = get_model_initializer(args.model)(*init_params).to(device)
    util.load_state_dict(checkpoint, model)
    torchsummary.summary(model, INPUT_SHAPE)
    test_model(test_loader, model, criterion, device)
