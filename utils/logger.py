from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch


class Logger:
    def __init__(self):
        self.writer = SummaryWriter()
        self.keys = {}

    def __call__(self, key, value):
        if key in self.keys:
            self.keys[key] += 1
        else:
            self.keys[key] = 1
        if type(value) in [int, float]:
            self.writer.add_scalar(key, value, self.keys[key])
        elif type(value) is np.ndarray:
            if len(value.shape) == 3:
                self.writer.add_image(key, value, self.keys[key])
            else:
                self.writer.add_images(key, value, self.keys[key])
        elif type(value) is torch.Tensor:
            if torch.numel(value) == 1:
                self.writer.add_scalar(key, float(value.item()), self.keys[key])
            else:
                raise ValueError("Can't log tensor containing more than one elements.")
        else:
            raise TypeError('Unexpected type. Options: int, float, 3d or 4d numpy arrays.')