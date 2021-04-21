import random
import numpy
import torch

from . import timit
from . import trainer


def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_devices(devices):
    pass


get_dataloaders = timit.get_dataloaders
set_time_limit = timit.set_time_limit
get_trainer = trainer.get_trainer
get_loss = trainer.get_loss
