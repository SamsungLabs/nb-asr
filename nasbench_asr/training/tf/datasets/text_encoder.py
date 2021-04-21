# pylint: skip-file

from nasbench_asr.quiet_tensorflow import tensorflow as tf

from .phoneme_encoder import PhonemeEncoder


def get_utf8_valid_sentence(sentence):
    return sentence.numpy()


def get_corpus_generator(ds):

    for _, sentence in ds:
        yield get_utf8_valid_sentence(sentence)


def get_encoded_from_sentence_fn(encoder):
    def get_encoded_from_sentence_helper(sentence):
        # the following [] are essential!
        encoded = [encoder.encode(get_utf8_valid_sentence(sentence))]

        return encoded

    def get_encoded_from_sentence(sentence):
        # the following [] are essential!
        encoded = tf.py_function(get_encoded_from_sentence_helper, [sentence],
                                 tf.int32)

        return encoded

    return get_encoded_from_sentence


def get_decoded_from_encoded_fn(encoder):
    def get_decoded_from_encoded_helper(encoded):
        # the following [] are essential!
        decoded = [
            get_utf8_valid_sentence(
                tf.constant(encoder.decode(encoded.numpy().tolist())))
        ]

        return decoded

    def get_decoded_from_encoded(encoded):
        # the following [] are essential!
        decoded = tf.py_function(get_decoded_from_encoded_helper, [encoded],
                                 tf.string)

        return decoded

    return get_decoded_from_encoded


class TextEncoder:
    def __init__(
        self,
        encoder_class,
    ):
        if encoder_class != 'phoneme':
            raise ValueError('Unsupported encoder type {!r}'.format(encoder_class))

        self.encoder_class = encoder_class
        self.encoder = PhonemeEncoder()
        self.get_encoded_from_sentence = get_encoded_from_sentence_fn(self.encoder)
        self.get_decoded_from_encoded = get_decoded_from_encoded_fn(self.encoder)
