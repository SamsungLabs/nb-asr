import torch.nn
import torch.nn.init

from .. import utils


def get_model(arch_vec, use_rnn, dropout_rate, gpu=None):
    from . import model
    from ... import search_space as ss
    arch_desc = ss.arch_vec_to_names(arch_vec)
    model = model.ASRModel(arch_desc, use_rnn=use_rnn, dropout_rate=dropout_rate)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.LSTM):
            for l in range(m.num_layers):
                wi = getattr(m, f'weight_ih_l{l}')
                wh = getattr(m, f'weight_hh_l{l}')
                bi = getattr(m, f'bias_ih_l{l}')
                bh = getattr(m, f'bias_hh_l{l}')
                torch.nn.init.xavier_uniform_(wi)
                torch.nn.init.xavier_uniform_(wh)
                torch.nn.init.zeros_(bi)
                torch.nn.init.zeros_(bh)

    model.apply(init_weights)
    if gpu is not None:
        model.to(device=f'cuda:{gpu}')

    return model


def print_model_summary(model):
    print(model)
    print('======================')
    def _print(m, level=0):
        for n, child in m.named_children():
            print('  '*level + type(child).__name__, ' ', n, ' ', sum(p.numel() for p in child.parameters()))
            _print(child, level+1)
    _print(model.model)
    print('======================')
    print('Trainable parameters:', utils.make_nice_number(sum(p.numel() for p in model.parameters())))
