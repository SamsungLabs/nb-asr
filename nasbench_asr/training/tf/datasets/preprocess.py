# pylint: skip-file
import numpy as np
from nasbench_asr.quiet_tensorflow import tensorflow as tf


def preprocess(
    *,
    ds,
    encoder,
    featurizer,
    norm_stats=None,
    epsilon=0.001,
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
    deterministic=True,
    max_feature_size=0
):
    """
    Args:
      - ds: yields (audio, sentence) tuples where
        - audio has shape [None], and is of type tf.float32
        - sentence has shape [], and is of type tf.string
    Returns:
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
      - featurizer
      - encoder
    """
    if norm_stats:
        norm_stats = np.load(norm_stats)
        mean = norm_stats['moving_mean']
        variance = norm_stats['moving_variance']
        norm_stats = True
  
    def preprocess_map_func(audio, sentence):
        feature = featurizer(audio)
        feature_size = tf.shape(feature)[0]
        encoded = encoder.get_encoded_from_sentence(sentence)
        encoded_size = tf.shape(encoded)[0]

        if norm_stats:
            feature = (feature - mean) / tf.math.sqrt(variance + epsilon)

        return feature, feature_size, encoded, encoded_size

    ds = ds.map(preprocess_map_func,
                num_parallel_calls=num_parallel_calls,
                deterministic=deterministic)

    if max_feature_size > 0:
        def filter_fn(feature, feature_size, encoded, encoded_size):
            return feature_size < tf.saturate_cast(max_feature_size, tf.int32)

        ds = ds.filter(filter_fn)

    return ds
