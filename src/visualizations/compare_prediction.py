import numpy as np
import matplotlib.pyplot as plt
import torch
from .viz_utils import save_figure

def compare_prediction(model, data, target):
    print("Comparing predictions to actual")
    model.eval()
    pred = model(data).detach().numpy() > 0.3
    NUM_EXAMPLES = 8
    data, pred, target = data.squeeze(), pred.squeeze(), target.squeeze()
    _, axs = plt.subplots(NUM_EXAMPLES, 3, figsize=(10, 20))
    for i in range(NUM_EXAMPLES):
        axs[i, 0].imshow(data[i], cmap=plt.cm.bone)
        axs[i, 1].imshow(pred[i], cmap=plt.cm.bone)
        axs[i, 2].imshow(target[i], cmap=plt.cm.bone)
    plt.show()
    plt.axis('off')

    # save_figure(run_name, 'activation_layers.png')
