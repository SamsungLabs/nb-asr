# pylint: skip-file
import os
import sys

import numpy as np
from nasbench_asr.quiet_tensorflow import tensorflow as tf


def cache_shard_shuffle_batch(*,
                              ds,
                              ds_cache_in_disk=False,
                              path_ds_cache="",
                              ds_cache_in_memory=False,
                              shard_num_shards=None,
                              shard_index=None,
                              shuffle=False,
                              shuffle_buffer_size=1,
                              num_feature_filters=None,
                              pad_strategy="padded_batch",
                              batch_size=2,
                              padded_shapes=([None, None], [], [None], []),
                              drop_remainder=True,
                              bucket_boundaries=[sys.maxsize],
                              bucket_batch_sizes=[2, 1],
                              device=None,
                              prefetch_buffer_size=1):
    """
    Args:
      - ds: yields (feature, feature_size, encoded, encoded_size) tuples where
        - feature has shape [time, channels], and is of type tf.float32
        - feature_size has shape [], and is of type tf.int32, and represents
          the number of time frames
        - encoded has shape [None], and is of type tf.int32, and represents a
          text encoded version of the original sentence; it contains values in
          the range [1, encoder.vocab_size)
        - encoded_size has shape [], and is of type tf.int32, and represents
          the number of tokens in each text encoded version of the original
          sentence
    Returns:
      - ds: yields (features, features_size, encodeds, encodeds_size) tuples
        where
        - features has shape [batch_size, time, channels]
        - features_size has shape [batch_size]
        - encodeds has shape [batch_size, None]
        - encodeds_size has shape [batch_size]
    """

    if ds_cache_in_disk:
        # cache to disk
        ds = ds.cache(path_ds_cache)

    if shard_num_shards is not None and shard_index is not None:
        ds = ds.shard(num_shards=shard_num_shards, index=shard_index)

    if ds_cache_in_memory:
        # cache to memory
        ds = ds.cache()

    if shuffle:
        ds = ds.shuffle(shuffle_buffer_size)

    if pad_strategy == "padded_batch":
        ds = ds.padded_batch(
            batch_size=batch_size,
            padded_shapes=padded_shapes,
            drop_remainder=drop_remainder,
        )
    elif pad_strategy == "bucket_by_sequence_length":

        def element_length_func(feature, feature_size, encoded, encoded_size):
            return feature_size

        transformation_func = tf.data.experimental.bucket_by_sequence_length(
            element_length_func=element_length_func,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            padded_shapes=padded_shapes,
            padding_values=None,
            pad_to_bucket_boundary=False,
            no_padding=False,
            drop_remainder=drop_remainder,
        )
        ds = ds.apply(transformation_func)
    else:
        assert False

    if device is not None:
        ds = ds.apply(
            tf.data.experimental.prefetch_to_device(
                device, buffer_size=prefetch_buffer_size))
    else:
        ds = ds.prefetch(prefetch_buffer_size)

    return ds
