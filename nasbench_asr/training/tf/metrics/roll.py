# pylint: skip-file
from nasbench_asr.quiet_tensorflow import tensorflow as tf


def roll(logits_transposed):
    return tf.roll(logits_transposed, shift=-1, axis=-1)
