# Tripletformer

This is the source code for the paper ``Tripletformer for Probabilistic Interpolation of Asynchronous Time Series``


# Requirements

python                    3.8.11

Pytorch                   1.9.0

sklearn                   0.0

numpy                     1.19.3

# Training and Evaluation

We provide an example for one dataset ``physionet``. All the datasets can be run in the similar manner using the hyperparameters provided.

```
python train_tripletformer.py --niters 2000 --dataset physionet --norm --shuffle --sample-tp 0.1 --mse-weight 1.0 --imab-dim 64 --cab-dim 256 --decoder-dim 128 --nlayers 1 --sample-type random --num-ref-points 128
```

# Hyperparameters

dataset | sample-type | sample-tp | mse-weight | imab-dim | cab-dim | decoder-dim | nlayers | num-ref-points
---|---|---|---|---|---|---|---|---
physionet | random | 0.1 | 1.0 | 64 | 256 | 128 | 1 | 128
physionet | random | 0.5 | 1.0 | 128 | 126 | 64 | 4 | 32
physionet | random | 0.9 | 5.0 | 256 | 256 | 256 | 3 | 128
mimiciii | random | 0.1 | 10.0 | 256 | 256 | 64 | 3 | 16
mimiciii | random | 0.5 | 0.0 | 128 | 256 | 256 | 1 | 128
mimiciii | random | 0.9 | 1.0 | 64 | 256 | 256 | 4 | 16
physionet2019| random | 0.1 | 0.0 | 128 | 64 | 256 | 3 | 128
physionet2019| random | 0.5 | 5.0 | 128 | 128 | 256 | 4 | 16
physionet2019| random | 0.9 | 1.0 | 128 | 128 | 128 | 1 | 128
PenDigits | random | 0.1 | 1.0 | 128 | 128 | 128 | 3 | 128
PenDigits | random | 0.5 | 10.0 | 128 | 128 | 128 | 4 | 16
PenDigits | random | 0.9 | 0.0 | 64 | 256 | 64 | 2 | 16
PhonemeSpectra | random | 0.1 | 5.0 | 256 | 64 | 256 | 4 | 16
PhonemeSpectra | random | 0.5 | 5.0 | 64 | 256 | 64 | 4 | 16
PhonemeSpectra | random | 0.9 | 5.0 | 128 | 256 | 256 | 2 | 64
physionet | bursts | 0.1 | 5.0 | 128 | 256 | 64 | 3 | 32
physionet | bursts | 0.5 | 10.0 | 128 | 256 | 128 | 3 | 128
physionet | bursts | 0.9 | 10.0 | 128 | 256 | 128 | 2 | 32
mimiciii | bursts | 0.1 | 0.0 | 128 | 256 | 256 | 1 | 32
mimiciii | bursts | 0.5 | 5.0 | 128 | 128 | 256 | 4 | 64
mimiciii | bursts | 0.9 | 10.0 | 256 | 256 | 128 | 4 | 128
physionet2019 | bursts | 0.1 | 0.0 | 128 | 128 | 128 | 2 | 32
physionet2019 | bursts | 0.5 | 0.0 | 64 | 128 | 256 | 3 | 16
physionet2019 | bursts | 0.9 | 1.0 | 128 | 64 | 256 | 4 | 16
PenDigits | bursts | 0.1 | 10.0 | 64 | 256 | 64 | 3 | 64
PenDigits | bursts | 0.5 | 0.0 | 256 | 256 | 128 | 4 | 16
PhonemeSpectra | bursts | 0.1 | 10.0 | 64 | 128 | 256 | 1 | 16
PhonemeSpectra | bursts | 0.5 | 5.0 | 256 | 64 | 256 | 3 | 64
PhonemeSpectra | bursts | 0.9 | 0.0 | 64 | 256 | 128 | 1 | 32

# Creating Synthetic Dataset

You can create synthetic dataset using ``make_ts_dataset_async.py`` in ``data_lib`` folder.
