from .. import utils


_backends = utils.BackendsAccessor(__file__, __name__)


def get_available_backends():
    return list(_backends.available_backends)


def set_default_backend(backend):
    return _backends.get_backend(backend, set_default=True).__name__.rsplit('.')[-1]


def get_backend_name(backend=None):
    return _backends.get_backend(backend).__name__.rsplit('.')[-1]


def get_model(*args, backend=None, **kwargs):
    return _backends.get_backend(backend).get_model(*args, **kwargs)


def print_model_summary(model):
    return _backends.get_backend(model.backend).print_model_summary(model)
