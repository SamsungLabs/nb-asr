import pathlib
import collections.abc as cabc

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_edit_distance as ed
from ctcdecode import CTCBeamDecoder

from ...model.torch.ops import PadConvRelu
from ...model import print_model_summary
from .timit import set_time_limit


class AvgMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.n = 0

    def update(self, a):
        if not self.n:
            self.avg = a
            self.n = 1
        else:
            self.avg = self.avg * (self.n / (self.n + 1)) + (a / (self.n + 1))
            self.n += 1

    def get(self):
        return self.avg


def get_loss():
    def loss(output, output_len, targets, targets_len):
        output_trans = output.permute(1, 0, 2) # needed by the CTCLoss
        loss = F.ctc_loss(output_trans, targets, output_len, targets_len, reduction='none', zero_infinity=True)
        loss /= output_len
        loss = loss.mean()
        return loss

    return loss


class Trainer():
    def __init__(self, dataloaders, loss, gpus=None, save_dir=None, verbose=True):
        #we don't use config param, it is just to have consistent api
        encoder, train_load, valid_load, test_load = dataloaders

        self.encoder = encoder
        self.train_load = train_load
        self.valid_load = valid_load
        self.test_load = test_load
        self.gpus = gpus
        self.save_dir = pathlib.Path(save_dir) if save_dir else save_dir
        if self.save_dir:
            self.save_dir.mkdir(exist_ok=True)
        self.verbose = verbose

        self.loss = loss
        if self.gpus is not None and (not isinstance(self.gpus, cabc.Sequence) or bool(self.gpus)):
            if not isinstance(gpus, cabc.Sequence):
                self.gpus = [self.gpus]

            self.device = torch.device(f'cuda:{gpus[0]}')
        else:
            self.device = torch.device('cpu')

        self.decoder = CTCBeamDecoder(encoder.get_vocab(inc_blank=True), beam_width=12, log_probs_input=True)

        self.model = None
        self._model = None
        self.lr = None
        self.optimizer = None
        self.scheduler = None
        self._best_weights = None

    def train(self, model, epochs=40, lr=0.0001, reset=False, model_name=None):
        self.model = model
        self._model = model
        self.lr = lr
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-07)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.9)

        if self.verbose:
            print_model_summary(self.model)

        self.model.to(device=self.device)
        if len(self.gpus) > 1:
            self.model = torch.nn.DataParallel(self.model, self.gpus)

        epoch = 0
        best_val = None
        val_scores = []

        this_model_save_dir = None
        if self.save_dir:
            this_model_save_dir = pathlib.Path(self.save_dir)
            if model_name is not None:
                this_model_save_dir = this_model_save_dir / str(model_name)

            this_model_save_dir.mkdir(exist_ok=True)

            latest_ckpt = this_model_save_dir / 'latest.ckpt'
            best_ckpt = this_model_save_dir / 'best.ckpt'

            # TODO: is that enough to restore state? or maybe we should always start from the beginnin
            if best_ckpt.exists():
                if reset:
                    best_ckpt.unlink()
                else:
                    self.load(best_ckpt)
                    self.remember_best()
            if latest_ckpt.exists():
                if reset:
                    latest_ckpt.unlink()
                else:
                    self.load(latest_ckpt)

        loss_tracker = AvgMeter()
        per_tracker = AvgMeter()

        warmup_limits = [1.0, 1.0, 2.0, 2.0]
        warmup = 0

        while epoch < epochs:
            if warmup < len(warmup_limits):
                set_time_limit(self.train_load, warmup_limits[warmup])
            else:
                set_time_limit(self.train_load, None)

            loss_tracker.reset()
            self.model.train()
            with tqdm.tqdm(self.train_load) as pbar:
                for train_input in pbar:
                    pbar.set_description(f'Avg. loss: {loss_tracker.get():.4f}')
                    loss, *_ = self.step(train_input, training=True)
                    loss_tracker.update(loss.item())

            if self.verbose:
                print(f'{"Warmup e" if warmup < len(warmup_limits) else "E"}poch {warmup+1 if warmup < len(warmup_limits) else epoch+1}: average loss: {loss_tracker.get():.4f}')

            if warmup < len(warmup_limits):
                warmup += 1
            else:
                loss_tracker.reset()
                per_tracker.reset()
                self.model.eval()
                for val_input in self.valid_load:
                    loss, logits, logits_len = self.step(val_input, training=False)
                    per = self.decode(logits, logits_len, val_input)
                    loss_tracker.update(loss.item())
                    per_tracker.update(per.item())

                val_loss = loss_tracker.get()
                val_per = per_tracker.get()
                val_scores.append((val_loss, val_per))

                if self.verbose:
                    print(f'Epoch {epoch+1}: average val loss: {val_loss:.4f}, average val per: {val_per:.4f}')

                is_best = False
                epoch += 1
                if best_val is None or val_per < best_val:
                    is_best = True
                    best_val = val_per

                if is_best:
                    if self.verbose:
                        print(f'    Best model, saving...')
                    self.remember_best()

                if epoch >= 5: # ignore epochs with time limits
                    self.scheduler.step()

                if self.save_dir:
                    self.save(latest_ckpt)
                    if is_best:
                        self.save(best_ckpt)

        if self.verbose:
            print('Performing final test')

        self.recall_best()
        loss_tracker.reset()
        per_tracker.reset()
        self.model.eval()
        for test_input in self.test_load:
            loss, logits, logits_len = self.step(test_input, training=False)
            per = self.decode(logits, logits_len, test_input)
            loss_tracker.update(loss.item())
            per_tracker.update(per.item())

        test_loss = loss_tracker.get()
        test_per = per_tracker.get()

        self.model = None
        self._model = None
        self.lr = None
        self.optimizer = None
        self.scheduler = None
        self._best_weights = None

        return val_scores, test_loss, test_per

    def step(self, inputs, training):
        (audio, audio_len), (targets, targets_len) = inputs
        audio = audio.to(device=self.device)
        audio_len = audio_len.to(device=self.device)
        targets = targets.to(device=self.device)
        targets_len = targets_len.to(device=self.device)

        if training:
            self.optimizer.zero_grad()
        output = self.model(audio)
        output = F.log_softmax(output, dim=2)
        output_len = audio_len // 4
        loss = self.loss(output, output_len, targets, targets_len)
        _regu_loss = loss + 0.01 * sum(torch.norm(l.conv.weight) for l in self._model.modules() if isinstance(l, PadConvRelu))
        if training:
            _regu_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

        return loss.detach(), output.detach(), output_len.detach()

    def decode(self, output, output_len, val_inputs):
        _, (targets, targets_len) = val_inputs
        targets = targets.to(device=self.device, dtype=torch.int)
        targets_len = targets_len.to(device=self.device, dtype=torch.int)

        targets = self.encoder.fold_encoded(targets, 39)

        beams, _, _, beams_len = self.decoder.decode(output, output_len)
        top_beams = beams[:,0].to(device=self.device, dtype=torch.int)
        top_beams_len = beams_len[:,0].to(device=self.device, dtype=torch.int)

        top_beams = self.encoder.fold_encoded(top_beams, 39)

        blank = torch.Tensor([0]).to(device=self.device, dtype=torch.int)
        sep = torch.Tensor([]).to(device=self.device, dtype=torch.int)

        per = ed.compute_wer(top_beams, targets, top_beams_len, targets_len, blank, sep)
        per = per.mean()
        return per

    def save(self, ckpt_name):
        torch.save({
            'model': self._model.state_dict(),
            'optim': self.optimizer.state_dict()
        }, str(ckpt_name))

    def load(self, ckpt_name):
        state = torch.load(str(ckpt_name), map_location=self.device)
        self._model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optim'])

    def remember_best(self):
        self._best_weights = self._model.state_dict()

    def recall_best(self):
        self._model.load_state_dict(self._best_weights)


def get_trainer(*args, **kwargs):
    return Trainer(*args, **kwargs)
