# pylint: skip-file

from .timit_foldings import *


class PhonemeEncoder:
    def __init__(self):
        source_phonemes, source_encodes, dest_phonemes, dest_encodes, _ = get_phoneme_mapping(source_enc_name='p61', dest_enc_name='p48')
        
        self.source_phonemes = source_phonemes
        self.dest_phonemes = dest_phonemes

        self.phoneme_to_index = {}
        for phoneme, encoding in zip(source_phonemes, dest_encodes):
            self.phoneme_to_index[phoneme] = encoding

        # As in tfds.features.text.SubwordTextEncoder, we assume Size of the
        # vocabulary. Decode produces ints [1, vocab_size). Hence the addition
        # of one to the len of the phonemes
        self.vocab_size = len(dest_phonemes) + 1

    def encode(self, sentence):
        indices = []
        for phoneme in sentence:
            phoneme = phoneme.decode("utf-8")
            assert (phoneme in self.phoneme_to_index), f"{phoneme} not present in the encoder list"
            if phoneme == "q":
                # there is no "q" in p48 nor in p39 and #
                # self.phoneme_to_index["q"] is 0 which is not a valid index
                continue
            indices.append(self.phoneme_to_index[phoneme])

        return indices

    def decode(self, indices):
        if len(indices) != 0:
            assert max(indices) < self.vocab_size
        sentence = ""

        return sentence
