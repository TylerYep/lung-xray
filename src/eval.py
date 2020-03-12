import os
import torch
from tqdm import tqdm

from src.models import Binary
from src.dataset import load_train_data
from src.args import init_pipeline
from src.losses import DiceLoss


def evaluate_model(model, loader, device, criteria=[DiceLoss()]):
    ## TODO this is function I haven't finished yet
    model.eval()
    totals = [0 for _ in range(len(criteria))]
    with torch.no_grad():
        with tqdm(desc='Test', total=len(loader), ncols=120) as pbar:
            for data, target in loader:
                data = data.to(device)
                output = model(data)
                for i, criterion in enumerate(criteria):
                    totals[i] += criterion(output, target)
            pbar.update()

    return [t / len(loader) for t in totals]


def evaluate():
    args, device, checkpoint = init_pipeline()
    _, val_loader, _, _, _ = load_train_data(args, device)

    model = Binary()
    # binary_checkpoint = 'AG'
    # binary_path = os.path.join('checkpoints', binary_checkpoint, 'model_best.pth.tar')
    # bin_model_weights = torch.load(binary_path)
    # model.load_state_dict(bin_model_weights['state_dict'])
    result = evaluate_model(model, val_loader, device)
    print(result)


if __name__ == '__main__':
    evaluate()