from .. import utils


_backends = utils.BackendsAccessor(__file__, __name__)


class Trainer():
    def __init__(self, dataloaders, gpus=None, save_dir=None, verbose=True):
        raise NotImplementedError()

    def train(self, model, epochs=40, lr=0.0001, reset=False, model_name=None):
        raise NotImplementedError()

    def step(self, input, training=True):
        raise NotImplementedError()

    def save(self, checkpoint):
        raise NotImplementedError()

    def load(self, checkpoint):
        raise NotImplementedError()

    def remember(self):
        raise NotImplementedError()

    def recall(self):
        raise NotImplementedError()


def get_available_backends():
    return list(_backends.available_backends)


def set_default_backend(backend):
    return _backends.get_backend(backend, set_default=True).__name__.rsplit('.')[-1]


def get_backend_name(backend=None):
    return _backends.get_backend(backend).__name__.rsplit('.')[-1]


def set_seed(*args, backend=None, **kwargs):
    return _backends.get_backend(backend).set_seed(*args, **kwargs)


def prepare_devices(devices, backend=None):
    return _backends.get_backend(backend).prepare_devices(devices)


def get_dataloaders(timit_root, batch_size=64, backend=None):
    ''' Prepare dataset.

        Arguments:
            timit_root (os.PathLike) - root folder holding TIMIT dataset
            batch_size (int) - batch size to use when training
            backend (str) - optional, specifies backend to use

        Returns:
            tuple: a tuple of 5 values, in order:

                - encoder object (used to encode phonemes)
                - iterable yielding training examples
                - iterable yielding validation examples
                - iterable yielding testing examples
                - backend-specific data
    '''
    return _backends.get_backend(backend).get_dataloaders(timit_root, batch_size=batch_size)


def get_loss(backend=None):
    return _backends.get_backend(backend).get_loss()


def get_trainer(dataloaders, loss, gpus=None, save_dir=None, verbose=False, backend=None) -> Trainer:
    ''' Return a :py:class:`Trainer` object which implements training functionality.

        Arguments:
            dataloaders - a set of data-related objects obtained by calling :py:func:`nasbench_asr.get_dataloaders`
            epochs (int) - number of epochs to train, default: 40
            lr (float) - learning rate to use, default: 0.0001
            gpus (list[int]) - a list of GPUs to use when training a model, by default use CPU only
            save_dir (str) - an optional directory name where the model will be save, if the directory exists
                when the training begins and ``reset`` is ``False``, the trainer will try to continue training
                from a checkpoint stored in the directory
            reset (bool) - specifies if training script should ignore existing checkpoints in ``save_dir`` at
                the beginning of training
            verbose (bool) - whether training should print to standard output
            backend (str) - optional, specifies backend to use
    '''
    return _backends.get_backend(backend).get_trainer(dataloaders, loss, gpus=gpus, save_dir=save_dir, verbose=verbose)
