import functools

from . import dataset
from . import model
from . import training

from .utils import add_module_properties
from .utils import staticproperty


Dataset = dataset.Dataset
BenchmarkingDataset = dataset.BenchmarkingDataset
StaticInfoDataset = dataset.StaticInfoDataset

from_folder = dataset.from_folder


def set_default_backend(backend):
    from .model import set_default_backend as impl1
    from .training import set_default_backend as impl2
    name1 = impl1(backend)
    name2 = impl2(backend)
    return name1, name2

def get_backend_name():
    from .model import get_backend_name as impl1
    from .training import get_backend_name as impl2
    return impl1(), impl2()

@functools.wraps(training.set_seed)
def set_seed(*args, **kwargs):
    return training.set_seed(*args, **kwargs)

@functools.wraps(training.prepare_devices)
def prepare_devices(*args, **kwargs):
    return training.prepare_devices(*args, **kwargs)

@functools.wraps(model.get_model)
def get_model(*args, **kwargs):
    return model.get_model(*args, **kwargs)

@functools.wraps(training.get_dataloaders)
def get_dataloaders(*args, **kwargs):
    return training.get_dataloaders(*args, **kwargs)

@functools.wraps(training.get_loss)
def get_loss(*args, **kwargs):
    return training.get_loss(*args, **kwargs)

@functools.wraps(training.get_trainer)
def get_trainer(*args, **kwargs):
    return training.get_trainer(*args, **kwargs)


def _get_version():
    from . import version
    return version.version

def _get_has_repo():
    from . import version
    return version.has_repo

def _get_repo():
    from . import version
    return version.repo

def _get_commit():
    from . import version
    return version.commit


add_module_properties(__name__, {
    '__version__': staticproperty(staticmethod(_get_version)),
    '__has_repo__': staticproperty(staticmethod(_get_has_repo)),
    '__repo__': staticproperty(staticmethod(_get_repo)),
    '__commit__': staticproperty(staticmethod(_get_commit))
})
