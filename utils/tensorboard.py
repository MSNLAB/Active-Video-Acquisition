from argparse import Namespace
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter


class TensorboardWriter:

    def __init__(self, log_dir=None):
        if log_dir is None:
            log_dir = f"runs/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.log_dir = log_dir

    @property
    def writer(self):
        if not hasattr(self, "_writer"):
            self._writer = SummaryWriter(self.log_dir)
        return self._writer

    def save_hyper_params(self, hyper_params):
        if isinstance(hyper_params, dict):
            hyper_params = hyper_params.items()
        elif isinstance(hyper_params, Namespace):
            hyper_params = vars(hyper_params).items()
        else:
            raise ValueError("Input object must be a dict or a namespace.")

        hyper_params = [f"{key}: {value}" for key, value in hyper_params]
        hyper_params = "\n\n".join(hyper_params)

        self.writer.add_text("hyper_params", hyper_params)

    def save_metrics(self, metrics, step=None):
        for tag, value in metrics.items():
            self.writer.add_scalar(tag, value, step)


def get_writer(log_dir=None):
    return TensorboardWriter(log_dir)
