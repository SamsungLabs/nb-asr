import pathlib
import argparse

from nasbench_asr import set_default_backend, get_backend_name, set_seed, prepare_devices, get_model, get_dataloaders, get_trainer, get_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=int, nargs=9)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--data', type=str, default='TIMIT')
    parser.add_argument('--rnn', type=bool, default=True)
    parser.add_argument('--exp_folder', type=str, default='results')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--backend', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--reset', action='store_true')
    parser.add_argument('--seed', type=int, default=1235)
    args = parser.parse_args()

    flat_model = tuple(map(str, args.model))
    args.model = [args.model[0:2], args.model[2:5], args.model[5:9]]

    if not args.exp_name:
        args.exp_name = '_'.join(flat_model) + f'_b{args.batch_size}_rnn{int(args.rnn)}'

    set_default_backend(args.backend)
    set_seed(args.seed)
    prepare_devices(args.gpus)

    args.backend = get_backend_name()[0]

    print(f'Using backend: {get_backend_name()}')
    print(f'    Model vec: {args.model}')
    print(f'    Training for {args.epochs} epochs')
    print(f'    Batch size: {args.batch_size}')
    print(f'    Learning rate: {args.lr}')
    print(f'    Dropout: {args.dropout}')
    print(f'    GPUs: {args.gpus}')

    results_folder = pathlib.Path(args.exp_folder) / args.backend

    first_gpu = None
    if args.gpus:
        first_gpu = args.gpus[0]

    dataloaders = get_dataloaders(args.data, batch_size=args.batch_size)
    loss = get_loss()
    model = get_model(args.model, use_rnn=args.rnn, dropout_rate=args.dropout, gpu=first_gpu)
    trainer = get_trainer(dataloaders, loss, gpus=args.gpus, save_dir=results_folder, verbose=True)
    trainer.train(model, epochs=args.epochs, lr=args.lr, reset=args.reset, model_name=args.exp_name)


if __name__ == "__main__":
    main()
