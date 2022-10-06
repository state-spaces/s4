This repository provides a modular and flexible implementation of general deep sequence models.

```
baselines/   Ported baseline models
functional/  Mathematical utilities
hippo/       Utilities for defining HiPPO operators
nn/          Standalone neural network components (nn.Module)
s4/          Standalone S4 modules
sequence/    Modular sequence model interface
```

# HiPPO

HiPPO is the mathematical framework upon which the papers HiPPO, LSSL, and S4 are built on.
HiPPO operators are defined in [hippo/hippo.py](hippo/hippo.py).
Function reconstruction experiments and visualizations are presented in [hippo/visualizations.py](hippo/visualizations.py).

# S4

Standalone implementations of S4 can be found inside [s4/](s4/) (see the README for usage).

# Modular Sequence Model Interface

This README provides a basic overview of the model source code.
It is recommended to see the [config README](../../configs/model/README.md) for running experiments with these models.


## SequenceModule
The SequenceModule class ([sequence/base.py](sequence/base.py)) is the abstract interface that all sequence models adhere to.
In this codebase, sequence models are defined as a sequence-to-sequence map of shape `(batch size, sequence length, model dimension)` to `(batch size, sequence length, output dimension)`.

The SequenceModule comes with other methods such as `step` which is meant for autoregressive settings, and logic to carry optional hidden states (for stateful models such as RNNs or S4).

To add a new model to this codebase, subclass `SequenceModule` and implement the required methods.

## SequenceModel
The `SequenceModel` class ([sequence/model.py](sequence/model.py)) is the main backbone with configurable options for residual function, normalization placement, etc.

SequenceModel accepts a black box config for a layer. Compatible layers are SequenceModules (i.e. composable sequence transformations) found under `sequence/`.

## Example Layers

### S4

The S4 module is found at [sequence/ss/s4.py](sequence/ss/s4.py).

Standalone versions are in the folder [s4/](s4/).

### LSSL

The LSSL is the predecessor of S4. It is currently not recommended for use, but the model can be found at [sequence/ss/lssl.py](sequence/ss/lssl.py).

It can be run by adding `model/layer=lssl` to the command line, or `model/layer=lssl model.layer.learn=0` for the LSSL-fixed model which does not train $A, B, \Delta$.

### RNNs

This codebase also contains a modular implementation of many RNN cells.
These include HiPPO-RNN cells from the original [HiPPO paper](https://arxiv.org/abs/2008.07669).

Some examples include `model=rnn/hippo-legs` and `model=rnn/hippo-legt` for HiPPO variants from the original [paper](https://arxiv.org/abs/2008.07669), or `model=rnn/gru` for a GRU reimplementation, etc.

An exception is `model=lstm` to use the PyTorch LSTM.

Example command (reproducing the Permuted MNIST number from the HiPPO paper, which was SotA at the time):
```
python train.py pipeline=mnist model=rnn/hippo-legs model.cell_args.hidden_size=512 train.epochs=50 train.batch_size=100 train.lr=0.001
```

# Baselines
Other sequence models are easily incorporated into this repository,
and several other baselines have been ported.

These include CNNs such as [CKConv](https://arxiv.org/abs/2102.02611) and continuous-time/RNN models such as [UnICORNN](https://arxiv.org/abs/2103.05487) and [LipschitzRNN](https://arxiv.org/abs/2006.12070).

Models and datasets can be flexibly interchanged.
Examples:
```
python -m train pipeline=cifar model=ckconv
python -m train pipeline=mnist model=lipschitzrnn
```


