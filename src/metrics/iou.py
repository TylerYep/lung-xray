from .metric import Metric

SMOOTH = 1e-6


class IoU(Metric):
    def __init__(self):
        super().__init__()
        self.epoch_acc = 0.0
        self.running_acc = 0.0

    def reset(self):
        self.running_acc = 0.0

    def update(self, val_dict):
        output, target = val_dict['output'], val_dict['target']
        output = output > 0.5
        accuracy = (((output & (target).bool()).float().sum((1, 2)) + SMOOTH)/((output | target.bool()).float().sum((1, 2)) + SMOOTH)).sum().item()
        self.epoch_acc += accuracy
        self.running_acc += accuracy
        return accuracy

    def get_batch_result(self, log_interval):
        return self.running_acc / log_interval

    def get_epoch_result(self, num_examples):
        return self.epoch_acc / num_examples
