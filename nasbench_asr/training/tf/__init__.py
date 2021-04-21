import os
import pathlib
import random

import numpy as np
from nasbench_asr.quiet_tensorflow import tensorflow as tf
from attrdict import AttrDict

from . import trainer
from .datasets.audio_featurizer import AudioFeaturizer
from .datasets.audio_sentence_timit import get_timit_audio_sentence_ds
from .datasets.preprocess import preprocess
from .datasets.text_encoder import TextEncoder
from .datasets.cache_shard_shuffle_batch import cache_shard_shuffle_batch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def prepare_devices(devices):
    physical_devices = [gpu for idx, gpu in enumerate(tf.config.list_physical_devices('GPU')) if idx in devices]
    if len(physical_devices) != len(devices):
        raise ValueError('Could not find all devices!')

    try:
        tf.config.experimental.set_visible_devices(physical_devices, 'GPU')
    except RuntimeError:
        return
    for pd in physical_devices:
        tf.config.experimental.set_memory_growth(pd, True)


def get_dataloaders(timit_root, batch_size):
    # hidden arguments
    encoder_class = 'phoneme'
    num_parallel_calls = tf.data.experimental.AUTOTUNE
    deterministic = True

    curriculum_learnings = [[16000, 2], [32000, 2]]

    splits = ['TRAIN', 'VAL', 'TEST']


    # Create common objects

    featurizer = AudioFeaturizer(
        sample_rate=16000,
        feature_type='lmel',
        normalize_full_scale=False,
        window_len_in_sec=0.025,
        step_len_in_sec=0.010,
        num_feature_filters=80,
        mel_weight_mat=None,
        verbose=False
    )

    encoder = TextEncoder(encoder_class=encoder_class)

    # helper function to apply common transformations for different
    # parts of timit
    def get_timit_ds(split_name, max_len):
        ds = get_timit_audio_sentence_ds(timit_root,
            split_name,
            remove_sa=True,
            encoder_class=encoder_class,
            num_parallel_calls=num_parallel_calls,
            deterministic=deterministic,
            max_audio_size=max_len)

        # stats_file = str(pathlib.Path(__file__).parents[1].joinpath(f'timit_train_stats.npz'))
        stats_file = None

        ds = preprocess(ds=ds,
            encoder=encoder,
            featurizer=featurizer,
            norm_stats=stats_file,
            epsilon=0.001,
            num_parallel_calls=num_parallel_calls,
            deterministic=deterministic,
            max_feature_size=0)

        ds = cache_shard_shuffle_batch(ds=ds,
            ds_cache_in_disk=False,
            path_ds_cache=None,
            ds_cache_in_memory=False,
            shard_num_shards=None,
            shard_index=None,
            shuffle=(split_name == 'TRAIN'),
            shuffle_buffer_size=2048,
            num_feature_filters=80,
            pad_strategy='bucket_by_sequence_length',
            batch_size=batch_size,
            padded_shapes=([None, 80], [], [None], []),
            drop_remainder=False,
            bucket_boundaries=[300],
            bucket_batch_sizes=[min(batch_size, 64), min(batch_size, 48)],
            device=None,
            prefetch_buffer_size=1)

        steps = 0
        for _ in ds:
            steps += 1

        #print('!!!!!', split_name, steps, batch_size)

        ds = AttrDict({
            'ds': ds,
            'encoder': encoder,
            'featurizer': featurizer,
            'steps': steps
        })

        return ds

    all_ds = []
    for split in splits:
        curriculum = []
        if split == 'TRAIN':
            for max_len, epochs in curriculum_learnings:
                c_ds = get_timit_ds(split, max_len)
                c_ds.ds = c_ds.ds.repeat(epochs)
                curriculum.append(c_ds)

        ds = get_timit_ds(split, 0)
        ds.ds = ds.ds.repeat()

        if curriculum:
            t = curriculum[0].ds
            for c in curriculum[1:]:
                t = t.concatenate(c.ds)
            t = t.concatenate(ds.ds)
            ds.ds = t

        all_ds.append(ds)

    return (encoder, *all_ds)


get_trainer = trainer.get_trainer
get_loss = trainer.get_loss