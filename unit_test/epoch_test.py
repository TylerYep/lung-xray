''' epoch_test.py '''
import cv2
from torch.utils.data import DataLoader

from src.train import train
from src.dataset import LungDataset, mask2rle, rle2mask

class TestCNN:

    @staticmethod
    def test_rle_conversion():
        NUM_EXAMPLES = 5
        train_set = LungDataset('train', n=NUM_EXAMPLES, img_dim=1024)
        train_loader = DataLoader(train_set, batch_size=NUM_EXAMPLES, shuffle=False)
        for _, mask in train_loader:
            for i in range(mask.shape[0]):
                out = mask[i].squeeze()
                rle = train_set.data[i][1]
                assert out.shape == (1024, 1024)
                assert mask2rle(out) == rle
                assert (rle2mask(rle) == out.numpy()).all()


    @staticmethod
    def test_epoch_resume():
        val_loss_start = train(['--epoch=2', '--name=TEST', '--n=2'])
        val_loss_end = train(['--epoch=2', '--checkpoint=TEST', '--n=2'])

        val_loss_test = train(['--epoch=4', '--n=2'])

        assert val_loss_test == val_loss_end
