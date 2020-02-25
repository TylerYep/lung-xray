import numpy as np
import matplotlib.pyplot as plt
import torch
from .viz_utils import save_figure

def compute_activations(model, data, target, run_name):
    model.eval()
    _, activations = model.forward_with_activations(data)
    NUM_EXAMPLES = 4
    NUM_SUBPLOTS = NUM_EXAMPLES * len(activations)
    _, axs = plt.subplots(NUM_SUBPLOTS // NUM_EXAMPLES, NUM_EXAMPLES)
    for i in range(NUM_EXAMPLES):
        for j, activ in enumerate(activations):
            activation = torch.abs(activ).mean(dim=1)[i]
            activation = activation.detach().cpu().numpy()
            activation /= activation.max()
            activation = plt.get_cmap('inferno')(activation)
            activation = np.delete(activation, 3, 2)  # deletes 4th channel created by cmap

            ax = axs[j, i]
            ax.imshow(activation)
            ax.axis('off')

    save_figure(run_name, 'activation_layers.png')
