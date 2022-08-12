This folder contains several standalone implementations of S4 variants.
The file [s4.py](./s4.py) contains the full implementation of S4 with all available options, which subsumes several variants of S4.
Other standalone implementations are documented below.

## Full S4(D) Model


`s4.py` is a standalone implementation of the full S4(D) model with all options, which are documented inside the class.

The corresponding [config](/configs/model/layer/s4.yaml) also lists all available options.

### S4

S4 is characterized by the arguments `mode=nplr` (the Normal Plus Low-Rank kernel described in the original S4 paper) and `measure=legs` (the HiPPO-LegS matrix), which are both set by default.
Alternative measures are supported, such as `measure=fout` which is the S4-FouT model described in [HTTYH](https://arxiv.org/abs/2206.12037).


### S4D

S4D is activated by the argument `mode=diag` which uses the diagonal kernel.
Pass in `measure=diag-lin` or `measure=diag-inv` for S4D-Lin or S4D-Inv.
Other options described in the S4D paper include
- `disc={'bilinear','zoh'}`: Bilinear vs. ZOH discretization
- `lr.B={0.0,None}`: frozen vs. trainable $B$ parameter (requires custom optimizer to register the hook)
- `real_type={'exp','relu','none'}`: parameterization of real part of $A$

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

**It is recommended to use S4D for this feature. The S4 implementation is currently not optimized.**

### Other Variants

#### DSS

[DSS](https://arxiv.org/abs/2203.14343) is the first diagonal SSM variant. It has two main characteristics:
1. *Computation* - uses a "softmax" which combines ZOH discretization + normalization over sequence length
2. *Initialization* - uses an [approximation](https://arxiv.org/abs/2206.11893) to the HiPPO matrix (also called HiPPO-LegS)

This model is equivalent to setting the options
```
S4(mode='diag', disc='dss', measure='diag-legs')
```
Performance should be similar to S4D, but it may consume more memory.

#### GSS

[GSS](https://arxiv.org/abs/2206.13947) is another variant specialized for language modeling on TPUs.
It has two main characteristics:
1. *Gating* - Incorporates an additional multiplicative feedforward branch. Additionally, it bottlenecks the dimension of the input to the SSM. These changes are largely motivated by efficiently on TPUs, which is better suited for large feedforward matmuls rather than the FFT convolutions used by the SSM.
2. *Simplified kernel* - Matrix $A$ is randomly initialized, matrix $B=1$ and step size $\Delta=1.0$ are frozen.

These modifications can all be flexibly toggled. The full GSS layer is roughly equivalent to the following options.
```
S4(
  gate=4,                   # Multiplicative gating layer that also expands dimension by factor of 4
  bottleneck=4,             # Reduce dimension of SSM by factor of 4
  measure='diag-rand',      # Randomly initialize A
  dt_min=1.0, dt_max=1.0,   # Initialize dt to 1.0
  lr={'dt': 0.0, 'B': 0.0}, # Freeze B and dt
)
```


## Minimal S4D

`s4d.py` contains a minimal implementation of the S4D layer. This file is primarily for pedagogical purposes to illustrate the simplicity of the core SSM principles behind S4.
<!--
It is not advised to be used for tuning, as it may be less performant and lacks several features of the full model.
-->

This S4D layer is equivalent to using the full S4 layer with specific settings, and stripping out all extra features:

```
S4(mode='diag', measure='diag-lin', bidirectional=False, disc='zoh', real_type='exp')
```

The `example.py` script incorporates this into a simple deep neural network backbone to achieve 88% on sequential CIFAR with a model of 200K parameters. It can also be run using the standard infrastructure in this repo with the command
```
python -m train experiment=cifar/s4d-minimal-cifar
```

## Simple S4

TODO: Merge branch and document


## LSSL
[lssl.py](./lssl.py) is an implementation of the [predecessor](https://arxiv.org/abs/2110.13985) of S4.
