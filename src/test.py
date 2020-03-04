import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary

from src import util
from src.args import init_pipeline
from src.dataset import load_test_data, INPUT_SHAPE
from src.models import BasicCNN as Model

if 'google.colab' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


def test_model(test_loader, model, criterion):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        with tqdm(desc='Test', total=len(test_loader), ncols=120) as pbar:
            for data, target in test_loader:
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                pbar.update()

    test_loss /= len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f},',
          f'Accuracy: {correct}/{len(test_loader.dataset)}',
          f'({100. * correct / len(test_loader.dataset):.2f}%)\n')


    # preds = torch.sigmoid(model(batch.to(device)))
    # preds = preds.detach().cpu().numpy().squeeze()
    # for probability in preds:
    #     if probability.shape != (1024, 1024):
    #         probability = cv2.resize(probability, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
    #     predict, num_predict = post_process(probability, best_threshold, min_size)
    #     if num_predict == 0:
    #         encoded_pixels.append('-1')
    #     else:
    #         r = run_length_encode(predict)
    #         encoded_pixels.append(r)
    # df['EncodedPixels'] = encoded_pixels
    # df.to_csv('submission.csv', columns=['ImageId', 'EncodedPixels'], index=False)


def main():
    args, device, checkpoint = init_pipeline()
    criterion = nn.BCELoss()
    test_loader = load_test_data(args, device)
    init_params = checkpoint.get('model_init', {})
    model = Model(*init_params).to(device)
    util.load_state_dict(checkpoint, model)
    torchsummary.summary(model, INPUT_SHAPE)

    test_model(test_loader, model, criterion)


if __name__ == '__main__':
    main()
