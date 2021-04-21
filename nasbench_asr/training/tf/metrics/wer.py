# pylint: skip-file
from nasbench_asr.quiet_tensorflow import tensorflow as tf


def separate_sentences_into_words_fn(x):
    return tf.strings.split(x, sep=" ")


def count_words_in_separated_sentences(x):
    return tf.cast(
        tf.sparse.reduce_sum(
            tf.ragged.map_flat_values(
                lambda y: tf.cast(y != "", dtype=tf.int64), x).to_sparse(),
            axis=1,
        ),
        dtype=tf.float32,
    )


def get_wer_numerator_denominator(*, sentences, logits_sentences):
    """
    sentences: vector of shape [batch_size] of type tf.string

    logits_sentences: vector of shape [batch_size] of type tf.string
    """
    words = separate_sentences_into_words_fn(sentences)
    logits_words = separate_sentences_into_words_fn(logits_sentences)

    wer_numerator = tf.edit_distance(
        hypothesis=logits_words.to_sparse(),
        truth=words.to_sparse(),
        normalize=False,
        name="wer_numerator",
    )

    words_len = count_words_in_separated_sentences(words)
    # logits_words_len = count_words_in_separated_sentences(logits_words)

    wer_denominator = tf.identity(words_len, name="wer_denominator")

    return wer_numerator, wer_denominator
