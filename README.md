# NASBench-ASR

Code for the "NAS-Bench-ASR: Reproducible Neural Architecture Search for Speech Recognition" paper published at ICLR'21: https://openreview.net/forum?id=CU0APx9LMaL

Content:
 * [Installing](#Installing)
 * [Dataset format](#Dataset-format)
 * [Using the dataset](#using-the-dataset)
 * [Creating and training models](#Creating-and-training-models)

## Installing

You can type `pip install <nbasr_folder>` to install our tool (optionally with the -e argument for in-place installation) and its dependencies.
We recommend using pipenv (or some other virtual environment management tool) to avoid problems with TF/Pytorch versions.

> **Note:** please let us know if you run into any problems related to missing dependencies


## Dataset format

We split out dataset into multiple pickle files containing information about the models from the search space in different settings.
Each dataset contains two parts: a header and the actual data. Those are serialized using pickle one after another, so when reading a similar sequence
needs to be used:
```python
with open('nb-asr-e40-1234.pickle', 'rb') as f:
    header = pickle.load(f)
    data = pickle.load(f)
```

The header contains usual metainformation about the dataset file, including things like: search space used when generating the dataset, column names, dataset type and version.
The following chunk of data is a python list with raw values - the order of values follows the order of column names in the header.

We distinguish three main types of datasets:
 * training datasets, which contain information about accuracy of models during/after training
 * benchmarking datasets, which contain information about on-device performance of models
 * static information datasets, which constain static information about models, such as number of parameters

Please see the following section to see a brief summary of how to use the dataset conveniently. 

## Using the dataset

All relevant files can be downaloded from the `releases` page in this repo. 

We provide a `Dataset` class as the top-level interface to querying the datasets (although the user is free to read the pickle files on their own).
The `Dataset` class is primarily used to deal with training datasets but it has an option to also piggy-back benchmarking and static datasets for convenience.
If needed the user can also use `BenchmarkingDataset` and `StaticInfoDataset` classes to access only benchmarking and/or static information about models without loading the data realted to training.

Assuming all NB-ASR files are in the same directory and no special behaviour is needed, the user can also use a high-level `nasbench_asr.from_folder` function which automatically searches for files in the given directory and creates a `Dataset` object from them.

The rest of the section presents some typical use cases.

Creating the dataset:
```python
>>> import nasbench_asr as nbasr
>>> d = nbasr.from_folder('~/data/nasbench_asr/', include_static_info=True)
```

Querying all information, returned as dict or list:
```python
>>> d.full_info([[1,0], [1,0,0], [1,0,0,0]], include_static_info=True)
{'val_per': [0.47851497, 0.32516438, 0.27674836, 0.25589427, 0.24639702, 0.23125456, 0.22919573, 0.228598, 0.22308561, 0.21856944, 0.22109318, 0.2183702, 0.21451816, 0.21498306, 0.21458457, 0.21239291, 0.21431892, 0.21418609, 0.21584645, 0.21584645, 0.21578003, 0.21704191, 0.21664342, 0.21843661, 0.2188351, 0.22003055, 0.22109318, 0.22149166, 0.23816165, 0.23643488, 0.22886366, 0.22082752, 0.2207611, 0.22142525, 0.22169091, 0.22056186, 0.22149166, 0.22182374, 0.22142525, 0.22202298], 'test_per': 0.242688849568367, 'arch_vec': [(1, 0), (1, 0, 0), (1, 0, 0, 0)], 'model_hash': '36855332a5778e0df5114305bc3ce238', 'seed': 1235, 'gtx-1080ti-fp32': {'latency': 0.04320073127746582}, 'jetson-nano-fp32': {'latency': 0.5421140193939209}, 'info': {'params': 26338848}}
>>> d.full_info([[1,0], [1,0,0], [1, 0,0,0]], include_static_info=True, return_dict=False)
['36855332a5778e0df5114305bc3ce238', [0.4840938, 0.31912068, 0.27867436, 0.25908217, 0.24433818, 0.23291492, 0.22713688, 0.22979344, 0.22288637, 0.22036262, 0.22056186, 0.21637776, 0.21823737, 0.21637776, 0.21272498, 0.21245933, 0.21318989, 0.21458457, 0.21591286, 0.2169755, 0.21797171, 0.21863586, 0.22036262, 0.22129242, 0.22129242, 0.2216245, 0.23152022, 0.24480309, 0.23450887, 0.22554293, 0.22268713, 0.221226, 0.22175732, 0.2216245, 0.22202298, 0.22182374, 0.22149166, 0.22222222, 0.22242147, 0.22228864], 0.23728343844413757, [(1, 0), (1, 0, 0), (1, 0, 0, 0)], 1236, [0.04320073127746582], [0.5421140193939209], [26338848]]
```

Removing static information:
```python
>>> d.full_info([[1,0], [1,0,0], [1, 0,0,0]], include_static_info=False)
{'val_per': [0.47851497, 0.32516438, 0.27674836, 0.25589427, 0.24639702, 0.23125456, 0.22919573, 0.228598, 0.22308561, 0.21856944, 0.22109318, 0.2183702, 0.21451816, 0.21498306, 0.21458457, 0.21239291, 0.21431892, 0.21418609, 0.21584645, 0.21584645, 0.21578003, 0.21704191, 0.21664342, 0.21843661, 0.2188351, 0.22003055, 0.22109318, 0.22149166, 0.23816165, 0.23643488, 0.22886366, 0.22082752, 0.2207611, 0.22142525, 0.22169091, 0.22056186, 0.22149166, 0.22182374, 0.22142525, 0.22202298], 'test_per': 0.242688849568367, 'arch_vec': [(1, 0), (1, 0, 0), (1, 0, 0, 0)], 'model_hash': '36855332a5778e0df5114305bc3ce238', 'seed': 1235, 'gtx-1080ti-fp32': {'latency': 0.04320073127746582}, 'jetson-nano-fp32': {'latency': 0.5421140193939209}}
```

Asking for a particular device performance only:
```python
>>> d.full_info([[1,0], [1,0,0], [1, 0,0,0]], devices='jetson-nano-fp32')
{'val_per': [0.4840938, 0.31912068, 0.27867436, 0.25908217, 0.24433818, 0.23291492, 0.22713688, 0.22979344, 0.22288637, 0.22036262, 0.22056186, 0.21637776, 0.21823737, 0.21637776, 0.21272498, 0.21245933, 0.21318989, 0.21458457, 0.21591286, 0.2169755, 0.21797171, 0.21863586, 0.22036262, 0.22129242, 0.22129242, 0.2216245, 0.23152022, 0.24480309, 0.23450887, 0.22554293, 0.22268713, 0.221226, 0.22175732, 0.2216245, 0.22202298, 0.22182374, 0.22149166, 0.22222222, 0.22242147, 0.22228864], 'test_per': 0.23728343844413757, 'arch_vec': [(1, 0), (1, 0, 0), (1, 0, 0, 0)], 'model_hash': '36855332a5778e0df5114305bc3ce238', 'seed': 1236, 'jetson-nano-fp32': {'latency': 0.5421140193939209}, 'info': {'params': 26338848}}
```

Do not include any benchmarking results:
```python
>>> d.full_info([[1,0], [1,0,0], [1, 0,0,0]], devices=False, include_static_info=False)
{'val_per': [0.48522282, 0.32031614, 0.28338978, 0.25430033, 0.24128312, 0.23942353, 0.22547652, 0.22733612, 0.22527727, 0.22109318, 0.21670984, 0.21929999, 0.21551438, 0.21458457, 0.21226008, 0.21305706, 0.2137876, 0.21352194, 0.2127914, 0.21491665, 0.21597928, 0.21777247, 0.21996413, 0.2249452, 0.2412167, 0.23484094, 0.23152022, 0.22281995, 0.21890152, 0.21870227, 0.21896791, 0.21896791, 0.21810454, 0.21863586, 0.21923357, 0.21896791, 0.21923357, 0.2198313, 0.21996413, 0.22056186], 'test_per': 0.23395703732967377, 'arch_vec': [(1, 0), (1, 0, 0), (1, 0, 0, 0)], 'model_hash': '36855332a5778e0df5114305bc3ce238', 'seed': 1234}
```

Querying test accuracy:
```python
>>> d.test_acc([[1,0], [1,0,0], [1,0,0,0]])
0.242688849568367
```

Querying validation accuracy:
```python
>>> d.val_acc([[1,0], [1,0,0], [1,0,0,0]])
0.21245933
>>> d.val_acc([[1,0], [1,0,0], [1,0,0,0]], best=False)
0.22056186
>>> d.val_acc([[1,0], [1,0,0], [1,0,0,0]], epoch=8)
0.22547652
>>> d.val_acc([[1,0], [1,0,0], [1,0,0,0]], epoch=8, best=False)
0.22979344
```

Querying latency alone:
```python
>>> d.latency([[1,0], [1,0,0], [1,0,0,0]], devices='gtx-1080ti-fp32')
[[0.04320073127746582]]
>>> d.latency([[1,0], [1,0,0], [1,0,0,0]], devices='gtx-1080ti-fp32', return_dict=True)
{'gtx-1080ti-fp32': {'latency': 0.04320073127746582}}
>>> d.bench_info.latency([[1,0], [1,0,0], [1,0,0,0]], devices='gtx-1080ti-fp32', return_dict=True)
{'gtx-1080ti-fp32': {'latency': 0.04320073127746582}}
```

Asking for missing information will result in an error:
```python
>>> d = nbasr.from_folder('~/data/nasbench_asr/', include_static_info=False)
>>> d.full_info([[1,0], [1,0,0], [1, 0,0,0]], include_static_info=True)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/SERILOCAL/l.dudziak/dev/asr/pytorch-asr/nasbench_asr/dataset.py", line 345, in full_info
    return self._query(model_hash, seed, devices, include_static_info, return_dict)
  File "/home/SERILOCAL/l.dudziak/dev/asr/pytorch-asr/nasbench_asr/dataset.py", line 304, in _query
    raise ValueError('No static information attached')
ValueError: No static information attached
```

Default values will always include data only if available:
```python
>>> d = nbasr.from_folder('~/data/nasbench_asr/', max_epochs=5, devices=False)
>>> d.full_info([[1,0], [1,0,0], [1, 0,0,0]])
{'val_per': [0.4846915, 0.3614266, 0.32323837, 0.31241283, 0.3053065], 'test_per': 0.3227997124195099, 'arch_vec': [(1, 0), (1, 0, 0), (1, 0, 0, 0)], 'model_hash': '36855332a5778e0df5114305bc3ce238', 'seed': 1234}
>>> d = nbasr.from_folder('~/data/nasbench_asr/', max_epochs=5, devices=None)
>>> d.full_info([[1,0], [1,0,0], [1, 0,0,0]])
{'val_per': [0.4846915, 0.3614266, 0.32323837, 0.31241283, 0.3053065], 'test_per': 0.3227997124195099, 'arch_vec': [(1, 0), (1, 0, 0), (1, 0, 0, 0)], 'model_hash': '36855332a5778e0df5114305bc3ce238', 'seed': 1234, 'gtx-1080ti-fp32': {'latency': 0.04320073127746582}, 'jetson-nano-fp32': {'latency': 0.5421140193939209}}
```

`nasbench_asr.dataset.from_folder` silently fails to include requested data if it doesn't exist, if this is undesired please consider using `Dataset` directly.

```python
>>> d = nbasr.from_folder('~/data/nasbench_asr/', max_epochs=5, devices='non-existing-device')
>>> d.full_info([[1,0], [1,0,0], [1, 0,0,0]])
{'val_per': [0.4846915, 0.3614266, 0.32323837, 0.31241283, 0.3053065], 'test_per': 0.3227997124195099, 'arch_vec': [(1, 0), (1, 0, 0), (1, 0, 0, 0)], 'model_hash': '36855332a5778e0df5114305bc3ce238', 'seed': 1234}
>>> d.full_info([[1,0], [1,0,0], [1, 0,0,0]], devices='non-existing-device')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/SERILOCAL/l.dudziak/dev/asr/pytorch-asr/nasbench_asr/dataset.py", line 346, in full_info
    return self._query(model_hash, seed, devices, include_static_info, return_dict)
  File "/home/SERILOCAL/l.dudziak/dev/asr/pytorch-asr/nasbench_asr/dataset.py", line 295, in _query
    raise ValueError('No benchmarking information attached')
ValueError: No benchmarking information attached
```

## Creating and training models
Alongside the dataset, we provide code to create models and training environment to train them in a reproducible way.
We anticipate that this might be especially useful for people working on differentiable NAS.

We support two training backends - Tensorflow or PyTorch. **However, please bear in mind that only the TF backend is meant to reproduce results from the dataset as we were using TF during our experiments.
PyTorch implementation is provided as a courtesy and comes with no guarantees about achievable results**

The following is a short summary of top-level function exposed by the package, most of those functions take an extra argument called `backend` which can be used to explicitly specify which implementation should be called.
If unused, the functions will try to use the default backend - which can be either set explicitly or it can be deduced.
The default backend is deduced based on available packages and prefers to use TF, falling back to PyTorch if TF is unavailable.
 * `set_default_backend(name)` sets the default backend
 * `get_backend_name()` gets the name of the backend in use (unless overwritten by a function-specific argument) 
 * `set_seed(seed)` sets random seed(s) to the specific value
 * `prepare_devices(devices)` prepared the specified GPUs for training (only relevant for TF backend, this e.g. turns on dynamic memory growth)
 * `get_model(arch)` return an implementation of a model with the specified architecture (`arch` should come from the search space, see e.g. `nasbench_asr.search_space.get_all_architectures`)
 * `get_dataloader(timti_root, batch_size)` returns a tuple of 5 values, in order: encoder object (used to encode phonemes), iterable yielding training examples, iterable yielding validation examples, iterable yielding testing examples, backend-specific data
 * `get_loss()` returns a callable objective, the signature is `(logits, logits_size, targets, targets_size)`
 * `get_trainer(dataloaders, gpus, save_dir, verbose)` returns a trainer class which can be used to train models.

 For more information on about to use those functions, please see for example `train.py` which can be used to run trainings of models, using those functions.
 You can also take a look at the `Trainer` abstract class defined in `nasbench_asr/training/__init__.py`.
 Briefly speaking, a sequence of functions like the following should do the trick:

 ```python
    set_default_backend(args.backend)
    set_seed(args.seed)
    prepare_devices(args.gpus)

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
    model = get_model(args.model, use_rnn=args.rnn, dropout_rate=args.dropout, gpu=first_gpu)
    trainer = get_trainer(dataloaders, gpus=args.gpus, save_dir=results_folder, verbose=True)
    trainer.train(model, epochs=args.epochs, lr=args.lr, reset=args.reset, model_name=args.exp_name)
 ```


## Citation

Please consider citing NB-ASR if you find our work useful!
```
@inproceedings{
mehrotra2021nasbenchasr,
title={{\{}NAS{\}}-Bench-{\{}ASR{\}}: Reproducible Neural Architecture Search for Speech Recognition},
author={Abhinav Mehrotra and Alberto Gil C. P. Ramos and Sourav Bhattacharya and {\L}ukasz Dudziak and Ravichander Vipperla and Thomas Chau and Mohamed S Abdelfattah and Samin Ishtiaq and Nicholas Donald Lane},
booktitle={International Conference on Learning Representations (ICLR)},
year={2021}
}
```
