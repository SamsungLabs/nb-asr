import pathlib
import contextlib

import numpy as np

from nasbench_asr.quiet_tensorflow import tensorflow as tf


def _call_once(func):
    _called = False
    _cache = None
    def impl(*args, **kwargs):
        nonlocal _called, _cache
        if _called:
            return _cache
        _cache = func(*args, **kwargs)
        _called = True
        return _cache
    return impl



@_call_once
def _get_data_norm():
    stats_file = pathlib.Path(__file__).parents[2].joinpath('training', 'timit_train_stats.npz')
    norm_stats = np.load(stats_file)
    mean = norm_stats['moving_mean']
    variance = norm_stats['moving_variance']
    return mean, variance


def get_model(arch_vec, use_rnn, dropout_rate, gpu=None):
    from .model import ASRModel

    with contextlib.ExitStack() as stack:
        if gpu is not None:
            stack.enter_context(tf.device(f'/GPU:{gpu}'))

        model = ASRModel(arch_vec,
            num_classes=48,
            use_rnn=use_rnn,
            dropout_rate=dropout_rate,
            input_shape=[None, 80],
            data_norm=_get_data_norm(),
            epsilon=0.001)

    return model


def print_model_summary(model):
    model._model.summary()
