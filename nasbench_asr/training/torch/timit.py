import pathlib

import torch
import torchaudio
import torchvision
import numpy as np

from .encoder import PhonemeEncoder


torchaudio.set_audio_backend('sox_io')


class TimitDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder, encoder, subset='TRAIN', ignore_sa=True, transforms=None):
        root = pathlib.Path(root_folder).expanduser()
        wavs = list(root.rglob(f'{subset}/**/*.RIFF.WAV'))
        wavs = sorted(wavs)
        if ignore_sa:
            wavs = [w for w in wavs if not w.name.startswith('SA')]
        phonemes = [(f.parent / f.stem).with_suffix('.PHN') for f in wavs]

        self.audio = []
        self.audio_len = []
        for wav in wavs:
            tensor, sample_rate = torchaudio.load(str(wav))
            self.audio.append(tensor)
            self.audio_len.append(tensor.shape[1] / sample_rate)

        def load_sentence(f):
            lines = f.read_text().strip().split('\n')
            last = [l.rsplit(' ', maxsplit=1)[-1] for l in lines]
            last = encoder.encode(last)
            return last

        self.root_folder = root_folder
        self.encoder = encoder
        self.sentences = [load_sentence(f) for f in phonemes]
        self.transforms = transforms

        assert len(self.audio) == len(self.sentences)

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        audio = self.audio[idx]
        sentence = self.sentences[idx]
        if self.transforms is not None:
            audio = self.transforms(audio)
        return audio, sentence

    def get_indices_shorter_than(self, time_limit):
        return [i for i, audio_len in enumerate(self.audio_len) if time_limit is None or audio_len < time_limit]


def pad_sequence_bft(sequences, extra=0, padding_value=0.0):
    batch_size = len(sequences)
    leading_dims = sequences[0].shape[:-1]
    max_t = max([s.shape[-1]+extra for s in sequences])

    out_dims = (batch_size, ) + leading_dims + (max_t, )

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.shape[-1]
        out_tensor[i, ..., :length] = tensor

    return out_tensor


def pad_sentences(sequences, padding_value=0.0):
    max_t = max([len(s) for s in sequences])
    sequences = [s+[0]*(max_t-len(s)) for s in sequences]
    return sequences


def get_normalize_fn(part_name, eps=0.001):
    stats = np.load(pathlib.Path(__file__).parents[1].joinpath(f'timit_train_stats.npz'))
    mean = stats['moving_mean'][None,:,None]
    variance = stats['moving_variance'][None,:,None]
    def normalize(audio):
        return (audio - mean) / (variance + eps)
    return normalize


def get_dataloaders(timit_root, batch_size):
    encoder = PhonemeEncoder(48)

    def get_transforms(part_name):
        transforms = torchvision.transforms.Compose([
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, win_length=400, hop_length=160, n_mels=80),
            torch.log,
            get_normalize_fn(part_name)
        ])

        return transforms

    def collate_fn(batch):
        audio = [b[0][0] for b in batch]
        audio_lengths = [a.shape[-1] for a in audio]
        sentence = [b[1] for b in batch]
        sentence_lengths = [len(s) for s in sentence]
        audio = pad_sequence_bft(audio, extra=0, padding_value=0.0)
        sentence = pad_sentences(sentence, padding_value=0.0)
        return (audio, torch.tensor(audio_lengths)), (torch.tensor(sentence, dtype=torch.int32), torch.tensor(sentence_lengths))


    subsets = ['TRAIN', 'VAL', 'TEST']
    datasets = [TimitDataset(timit_root, encoder, subset=s, ignore_sa=True, transforms=get_transforms(s)) for s in subsets]
    train_sampler = torch.utils.data.SubsetRandomSampler(datasets[0].get_indices_shorter_than(None))
    loaders = [torch.utils.data.DataLoader(d, batch_size=batch_size, sampler=train_sampler if not i else None, pin_memory=True, collate_fn=collate_fn) for i, d in enumerate(datasets)]
    return (encoder, *loaders)


def set_time_limit(loader, time_limit):
    db = loader.dataset
    sampler = loader.sampler
    sampler.indices = db.get_indices_shorter_than(time_limit)


if __name__ == '__main__':
    import pprint
    train_load, val_load, test_load = get_dataloaders('TIMIT', 3)
    for (audio, lengths), sentence in train_load:
        print(audio.shape, audio)
        print()
        print(lengths.shape, lengths)
        print()
        print(sentence)
        break
