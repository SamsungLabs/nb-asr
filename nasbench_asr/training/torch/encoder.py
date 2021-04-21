import pathlib



def split_with_pad(line, sep, min_len, pad_with=''):
    parts = line.split(sep)
    if len(parts) < min_len:
        missing = (min_len - len(parts))
        parts.extend([pad_with] * missing)
    return parts


class PhonemeEncoder:
    all_encodings = [61, 48, 39]

    def __init__(self, num_classes, remove_folded=True):
        assert num_classes in PhonemeEncoder.all_encodings

        self.num_classes = num_classes
        self.class_idx = PhonemeEncoder.all_encodings.index(self.num_classes)
        self.remove_folded = remove_folded

        num_encodings = len(PhonemeEncoder.all_encodings)

        mapping = pathlib.Path(__file__).parents[1].joinpath('timit_folding.txt').read_text().strip().split('\n')
        mapping = [split_with_pad(line, '\t', num_encodings) for line in mapping]
        self.mappings = {}
        self.to_delete = {}
        for src in range(num_encodings-1):
            self.mappings[src] = {}
            self.to_delete[src] = {}
            for dst in range(src+1, num_encodings):
                self.mappings[src][dst] = { line[src]: line[dst] for line in mapping }
                self.to_delete[src][dst] = set(p for p, v in self.mappings[src][dst].items() if not v)

        self.encodeds = [set(line[idx] for line in mapping if line[idx]) for idx in range(num_encodings)]
        self.encodeds = [sorted(list(encodeds)) for encodeds in self.encodeds]

        self.idx_mappings = {}
        for src in range(num_encodings-1):
            self.idx_mappings[src] = {}
            for dst in range(src+1, num_encodings):
                self.idx_mappings[src][dst] = { 0: 0 }
                for src_idx, src_ph in enumerate(self.encodeds[src]):
                    dst_ph = self.mappings[src][dst][src_ph]
                    dst_idx = self.encodeds[dst].index(dst_ph)+1 if dst_ph else 0
                    self.idx_mappings[src][dst][src_idx+1] = dst_idx

    def get_vocab(self, inc_blank=False, num_classes=None):
        class_idx = PhonemeEncoder.all_encodings.index(num_classes) if num_classes is not None else self.class_idx
        ret = list(self.encodeds[class_idx])
        if inc_blank:
            ret = ['_'] + ret
        return ret

    def _fold(self, phonemes, dst_class_idx=None):
        if dst_class_idx is None:
            dst_class_idx = self.class_idx
        if dst_class_idx == 0:
            return phonemes

        return [self.mappings[0][dst_class_idx][p] for p in phonemes if not self.remove_folded or p not in self.to_delete[0][dst_class_idx]]

    def fold_encoded(self, encodeds, num_classes):
        if num_classes >= self.num_classes:
            return encodeds
        if num_classes not in PhonemeEncoder.all_encodings:
            raise ValueError(num_classes)

        new_class_idx = PhonemeEncoder.all_encodings.index(num_classes)
        for old_idx, new_idx in self.idx_mappings[self.class_idx][new_class_idx].items():
            encodeds[encodeds == old_idx] = new_idx

        return encodeds


    def encode(self, phonemes):
        phonemes_folded = self._fold(phonemes)
        enc = [self.encodeds[self.class_idx].index(p)+1 if p else 0 for p in phonemes_folded] #start from 1, 0 is used for blank
        return enc

    def decode(self, encodeds):
        return [self.encodeds[self.class_idx][idx-1] if idx else '' for idx in encodeds]

