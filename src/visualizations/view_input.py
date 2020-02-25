import matplotlib.pyplot as plt
from .viz_utils import rearrange, save_figure


def view_input(data, target, run_name):
    ''' Data is of shape (B, C, H, W) '''
    NUM_EXAMPLES = 6
    NUM_ROWS = 2
    _, axs = plt.subplots(NUM_ROWS, NUM_EXAMPLES // NUM_ROWS + 1)
    data, target = data.cpu(), target.cpu()
    for i, ax in enumerate(axs.flat):
        img = rearrange(data[i])
        mask = rearrange(target[i])
        ax.imshow(img + mask, cmap=plt.cm.bone)
        if mask.sum() == 0:
            plt.title('no mask')
        ax.axis('off')

    save_figure(run_name, 'input_data.png')
