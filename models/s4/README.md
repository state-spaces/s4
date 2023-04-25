## Papers

### S4 (ICLR 2022)

![Structured State Spaces](/assets/s4.png "Properties of Structured State Spaces")
> **Efficiently Modeling Long Sequences with Structured State Spaces**\
> Albert Gu, Karan Goel, Christopher Ré\
> Paper: https://arxiv.org/abs/2111.00396

### S4D (NeurIPS 2022)

![S4D](/assets/s4d.png "S4D: The diagonal variant of S4")
> **On the Parameterization and Initialization of Diagonal State Space Models**\
> Albert Gu, Ankit Gupta, Karan Goel, Christopher Ré\
> Paper: https://arxiv.org/abs/2206.11893

## Models

The core S4 model is a linear sequence-to-sequence transformation.
It can be computed in multiple ways; the primary way for training is through the convolution view, which proceeds in two steps.

First, S4 generates an explicit convolution kernel which is a function of its SSM parameters $(A, B, C)$.
Different variants of S4 use different parameterizations and algorithms to compute this kernel.
The original S4 model has a diagonal plus low-rank (DPLR) $A$, while S4D has a diagonal $A$.
These are computed by the `SSKernelDPLR` and `SSKernelDiag` classes in [[/src/models/sequence/kernels/ssm.py](/src/models/sequence/kernels/ssm.py)], which are modules that produce a convolution kernel.

The S4 kernel can then be used in any vanilla CNN block. It is important to note that *S4 refers only to the core linear model* (e.g. the convolution kernel), not the exact structure of the deep neural network.
The CNN block used in the original S4 paper can be found at
[[/src/models/sequence/modules/s4block.py](/src/models/sequence/modules/s4block.py)], which accepts any type of convolution kernel besides S4.

Beside the convolution mode, S4 has many more properties explained in the papers. Some of these are documented in the next section.

## Experiments

[[experiments.md](experiments.md)] documents reproducible experiments from the above papers.

## Standalone Code

This folder contains standalone implementations of the full S4 DNN layer, where the above classes are consolidated into one file for ease of exporting.
The file [[s4.py](s4.py)] contains the full implementation of S4(D) with almost all available options, which subsumes several variants of S4.

The corresponding [config](/configs/model/layer/s4.yaml) also lists the available options.

### S4

S4 is characterized by the arguments `mode=nplr` (the Normal Plus Low-Rank kernel described in the original S4 paper) and `init=legs` (the HiPPO-LegS matrix), which are both set by default.
Alternative inits are supported, such as `init=fout` which is the S4-FouT model described in [HTTYH](https://arxiv.org/abs/2206.12037).


### S4D

S4D is activated by the argument `mode=diag` which uses the diagonal kernel.
The default initialization (`init=legs`) does not need to be changed, which corresponds to the S4D-LegS method from the paper that approximates the original S4.
Pass in `init=diag-lin` or `init=diag-inv` for S4D-Lin or S4D-Inv.
Other options described in the S4D paper include
- `disc={'bilinear','zoh'}`: Bilinear vs. ZOH discretization
- `lr.B={0.0,None}`: frozen vs. trainable $B$ parameter (requires custom optimizer to register the hook)
- `real_transform={'exp','relu','none'}`: parameterization of real part of $A$

### Usage and Features

#### Convolution Mode
The `forward` pass of the module maps a sequence of shape `(B, H, L) -> (B, H, L)` (batch size, hidden dimension, sequence length). The forward pass first constructs a convolution kernel using the algorithm described in the S4(D) papers, then convolves using the FFT.

#### Recurrent Mode
The `step` method of the module maps `(B, H) -> (B, H)`. This represents a single step or "unroll" of the model like an RNN.

#### Sample Rate Change
The `rate` argument in the forward pass multiplies the internal step size $\Delta$.
For example, a model trained on audio signals at 16000Hz using the default `rate=1.0` can be used to process audio signals at 8000Hz *without retraining* by passing in `rate=2.0`.

#### State Forwarding
The forward pass of the model accepts an optional initial state of shape `(B, H, N)`.
The model will then compute "forward" the state through the sequence, returning the final state as well as the output.

Note that this is equivalent to using `step` repeatedly, but is much faster by combining both recurrent and convolutional mode.

It is recommended to use S4D for this feature. The S4 implementation is currently not optimized.


### Minimal S4D

`s4d.py` contains a minimal implementation of the S4D layer. This file is primarily for pedagogical purposes to illustrate the simplicity of the core principles behind S4.

This S4D layer is equivalent to using the full S4 layer with specific settings, and stripping out all extra features:

```
S4(mode='diag', init='diag-lin', bidirectional=False, disc='zoh', real_transform='exp')
```

The `example.py` script incorporates this into a simple deep neural network backbone to achieve 88% on sequential CIFAR with a model of 200K parameters. It can also be run using the standard infrastructure in this repo with the command
```
python -m train experiment=cifar/s4d-minimal-cifar
```


### LSSL (NeurIPS 2021)

![Linear State Space Layer](/assets/splash.png "Properties of State Spaces")
> **Combining Recurrent, Convolutional, and Continuous-time Models with the Linear State Space Layer**\
> Albert Gu, Isys Johnson, Karan Goel, Khaled Saab, Tri Dao, Atri Rudra, Christopher Ré\
> Paper: https://arxiv.org/abs/2110.13985

LSSL is the first version of S4 which has been preserved for historical context. The full implementation can be found at [/src/models/sequence/modules/lssl.py](/src/models/sequence/modules/lssl.py).
It can be run by adding `model/layer=lssl` to any experiment command, or `model/layer=lssl model.layer.learn=0` for the "LSSL-fixed" model from the paper which does not train $A, B, \Delta$.
