from typing import Any, Dict
from enum import Enum, unique
import torch
from torch.utils.tensorboard import SummaryWriter

from src.metrics import get_metric
from src.dataset import CLASS_LABELS

@unique
class Mode(Enum):
    TRAIN = 'Train'
    VAL = 'Val'
    TEST = 'Test'


# Adding metrics here will automatically search the metrics/ folder for an implementation.
METRIC_NAMES = ['Loss', 'IoU', 'Dice']


class MetricTracker:
    def __init__(self,
                 run_name,
                 train_len,
                 val_len,
                 log_interval,
                 epoch=0,
                 num_batches=0,
                 metric_data=None,
                 best_metric=None):
        assert METRIC_NAMES
        self.writer = SummaryWriter(run_name)
        self.epoch = epoch
        self.train_len = train_len
        self.val_len = val_len
        self.log_interval = log_interval
        self.num_batches = num_batches
        self.metric_names = METRIC_NAMES
        self.primary_metric = self.metric_names[0]
        self.metric_data = metric_data if metric_data else \
                           {name: get_metric(name)() for name in self.metric_names}
        self.best_metric = best_metric if best_metric else \
                           self.metric_data[self.primary_metric].init_val

    def json_repr(self) -> Dict[str, Any]:
        return {'epoch': self.epoch,
                'metric_data': self.metric_data,
                'num_batches': self.num_batches,
                'best_metric': self.best_metric}

    def __getattr__(self, name):
        return self.metric_data[name]

    def add_network(self, model, loader):
        data, _ = next(iter(loader))
        self.writer.add_graph(model, data)

    def update_best_metric(self, val_loss) -> bool:
        is_best = val_loss < self.best_metric
        self.best_metric = min(val_loss, self.best_metric)
        return is_best

    def write(self, title: str, val: float, step_num: int):
        self.writer.add_scalar(title, val, step_num)

    def next_epoch(self):
        self.epoch += 1

    def set_num_batches(self, num_batches: int):
        self.num_batches = num_batches

    def reset_all(self):
        for metric in self.metric_data:
            self.metric_data[metric].reset()

    def reset_hard(self):
        self.metric_data = {name: get_metric(name)() for name in self.metric_names}

    def update_all(self, val_dict):
        ret_dict = {}
        for metric, metric_obj in self.metric_data.items():
            ret_dict[metric] = metric_obj.update(val_dict)
        return ret_dict

    def write_all(self, num_steps, mode):
        for metric, metric_obj in self.metric_data.items():
            batch_result = metric_obj.get_batch_result(self.log_interval)
            self.write(f'{mode}_Batch_{metric}', batch_result, num_steps)

    def add_images(self, val_dict, num_steps):
        data, output, target = val_dict['data'], val_dict['output'], val_dict['target']
        for j in range(output.shape[0]):
            if j > 5:
                break
            out = output.detach()[j]
            out = out > 0.3
            targ = target.detach()[j]
            self.writer.add_image(f'{j}/{self.epoch}/pred', out, num_steps)
            self.writer.add_image(f'{j}/{self.epoch}/target', targ, num_steps)

    def batch_update(self, i, data, loss, output, target, mode):
        names = ('data', 'loss', 'output', 'target')
        variables = (data, loss, output, target)
        val_dict = dict(zip(names, variables))

        tqdm_dict = self.update_all(val_dict)
        num_steps = (self.epoch - 1) * self.num_batches + i
        if mode == Mode.TRAIN and i % self.log_interval == 0:
            if i > 0:
                self.write_all(num_steps, mode)
            self.reset_all()
        elif mode == Mode.VAL:
            if i == 0:
                self.add_images(val_dict, num_steps)
        return {}

    def get_epoch_results(self, mode) -> float:
        result_str = ''
        for metric, metric_obj in self.metric_data.items():
            num_examples = self.train_len if mode == Mode.TRAIN else self.val_len
            epoch_result = metric_obj.get_epoch_result(num_examples)
            result_str += f'{metric_obj.formatted(epoch_result)} '
            self.write(f'{mode}_Epoch_{metric}', epoch_result, self.epoch)

        print(f'{mode} {result_str}')
        ret_val = self.metric_data[self.primary_metric].get_epoch_result(num_examples)
        self.reset_hard()
        return ret_val
