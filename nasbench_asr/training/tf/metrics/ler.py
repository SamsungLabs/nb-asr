# pylint: skip-file
from nasbench_asr.quiet_tensorflow import tensorflow as tf


def count_non_zero_indices(x):
    return tf.cast(tf.math.reduce_sum(tf.cast(x != 0, tf.int64), axis=1),
                   dtype=tf.float32)


def get_ler_numerator_denominator(*, encodeds, logits_encodeds):
    """
    encodeds: vector of shape [batch_size, time_1] where each row corresponds
    to a sample, whose zero elements come from padded_batch and whose non-zero
    elements are in the range [1, vocab_size) which correspond to valid indices
    from the tfds.features.text.SubwordTextEncoder

    logits_encodeds: vector of shape [batch_size, time_2] where each row
    corresponds to a sample, whose zero elements come from padded_batch and
    whose non-zero elements are in the range [1, vocab_size) which correspond
    to valid indices from the tfds.features.text.SubwordTextEncoder
    """
    ler_numerator = tf.edit_distance(
        hypothesis=tf.sparse.from_dense(logits_encodeds),
        truth=tf.sparse.from_dense(encodeds),
        normalize=False,
        name="ler_numerator",
    )

    encodeds_len = count_non_zero_indices(encodeds)
    # logits_encodeds_len = count_non_zero_indices(logits_encodeds)

    ler_denominator = tf.identity(encodeds_len, name="ler_denominator")

    return ler_numerator, ler_denominator
