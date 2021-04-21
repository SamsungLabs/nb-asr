# pylint: skip-file
import os
import sys

from nasbench_asr.quiet_tensorflow import tensorflow as tf

from .roll import roll


def get_normalized_ctc_loss_without_reduce(*, logits_transposed, logits_size,
                                           encodeds, encodeds_size):
    ctc_loss_without_reduce = tf.nn.ctc_loss(
        labels=encodeds,
        logits=logits_transposed,
        label_length=encodeds_size,
        logit_length=logits_size,
        logits_time_major=True,
        blank_index=0,
    )

    # tf.nn.ctc_loss returns a tensor of shape [batch_size] with negative log
    # probabilities, but each probability may have been computed with an
    # argument with different length (which turn into sums, each with different
    # number of summands in the case of independence). For this reason we
    # divide each negative log probability by the logits_size
    # replacing "logits_size" with "logits_size + 1" to avoid division by zero
    ctc_loss_without_reduce /= tf.cast(logits_size + 1,
                                       ctc_loss_without_reduce.dtype)

    ctc_loss_without_reduce = tf.debugging.check_numerics(
        tensor=ctc_loss_without_reduce,
        message="nan or inf in ctc_loss",
        name="ctc_loss_without_reduce",
    )

    return ctc_loss_without_reduce


def get_normalized_ctc_loss(*, logits_transposed, logits_size, encodeds,
                            encodeds_size):
    ctc_loss_without_reduce = get_normalized_ctc_loss_without_reduce(
        logits_transposed=logits_transposed,
        logits_size=logits_size,
        encodeds=encodeds,
        encodeds_size=encodeds_size,
    )

    # Finally, average across the samples of the batch
    ctc_loss = tf.reduce_mean(ctc_loss_without_reduce)

    return ctc_loss


def get_logits_encodeds(
    *,
    logits_transposed,
    logits_size,
    greedy_decoder,
    beam_width,
):
    # Unlike tf.nn.ctc_loss, the functions
    # tf.nn.ctc_greedy_decoder and tf.nn.ctc_beam_search_decoder don't have
    # a parameter to signal which is the blank_index. In fact, in the
    # tf.nn.ctc_greedy_decoder the documentation mentions that blank index
    # (num_classes - 1)

    # To account for the fact that the text encoder
    # https://www.tensorflow.org/datasets/api_docs/python/tfds/features/text/TextEncoder
    # encodes to the range [1,
    # vocab_size), and we took advantage of that by setting blank_index=0
    # in the get_normalized_ctc_loss, we now roll the logits_transposed
    # with shift=-1, axis=-1, so that the blank_index is moved from the
    # 0-th position to the last
    logits_transposed = roll(logits_transposed)

    if greedy_decoder:
        logits_encodeds, _ = tf.nn.ctc_greedy_decoder(
            inputs=logits_transposed,
            sequence_length=logits_size,
            merge_repeated=True,
        )
    else:
        logits_encodeds, _ = tf.nn.ctc_beam_search_decoder(
            inputs=logits_transposed,
            sequence_length=logits_size,
            beam_width=beam_width,
            top_paths=1,
        )
    logits_encodeds = logits_encodeds[0]

    # Given that the text encoder
    # https://www.tensorflow.org/datasets/api_docs/python/tfds/features/text/TextEncoder
    # encodes to and decodes from the range [1, vocab_size), we shift the
    # output of the ctc decoder which is in the range [0, vocab_size - 1)
    # to the correct range [1, vocab_size) by adding one each index
    logits_encodeds = tf.sparse.SparseTensor(
        indices=logits_encodeds.indices,
        values=logits_encodeds.values + 1,
        dense_shape=logits_encodeds.dense_shape,
    )

    logits_encodeds = tf.sparse.to_dense(logits_encodeds)
    logits_encodeds = tf.cast(logits_encodeds, tf.int32)

    return logits_encodeds
