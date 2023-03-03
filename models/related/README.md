This document outlines several models related to S4, including how to use them in this codebase if available, or pointers to their official repositories.

### S5 (ICLR 2023)
> **Simplified State Space Layers for Sequence Modeling**\
> Jimmy T.H. Smith, Andrew Warrington, Scott W. Linderman\
> Paper: https://arxiv.org/abs/2208.04933

S5 (Simplified S4) makes two main changes to S4. First, it concurrently discovered the same diagonal approximation to the original S4 HiPPO matrix that DSS and S4D use. Second, it uses a multi-input multi-output (MIMO) SSM instead of single-input single-output (SISO) like S4. This also decreases the effective hidden size of the model and allows the SSM state $x$ to be materialized, so S5 is more efficiently computed by directly unrolling the recurrence, than by the convolutional view of S4.

There is no known PyTorch implementation of S5 as PyTorch currently does not support general scan functions.
The official S5 implementation is in JAX: https://github.com/lindermanlab/S5


### GSS (ICLR 2023)
> **Long Range Language Modeling via Gated State Spaces**\
> Harsh Mehta, Ankit Gupta, Ashok Cutkosky, Behnam Neyshabur\
> Paper: https://arxiv.org/abs/2206.13947

GSS (Gated State Space) is variant of DSS/S4D specialized for language modeling on TPUs.
It has two main characteristics:
1. *Gating* - Incorporates an additional multiplicative feedforward branch. Additionally, it bottlenecks the dimension of the input to the SSM. These changes are largely motivated by efficiently on TPUs, which is better suited for large feedforward matmuls rather than the FFT convolutions used by the SSM.
2. *Simplified kernel* - Matrix $A$ is randomly initialized, matrix $B=1$ and step size $\Delta=1.0$ are frozen.

These modifications can all be flexibly toggled. The full GSS layer is roughly equivalent to the following options.
```
S4(
  gate=4,                   # Multiplicative gating layer that also expands dimension by factor of 4
  bottleneck=4,             # Reduce dimension of SSM by factor of 4
  init='diag-rand',         # Randomly initialize A
  dt_min=1.0, dt_max=1.0,   # Initialize dt to 1.0
  lr={'dt': 0.0, 'B': 0.0}, # Freeze B and dt
  imag_transform='exp',     # Parameterize imag part of A under exp transform
)
```

### SGConv (ICLR 2023)

> **What Makes Convolutional Models Great on Long Sequence Modeling?**\
> Yuhong Li, Tianle Cai, Yi Zhang, Deming Chen, Debadeepta Dey\
> Paper: https://arxiv.org/abs/2210.09298

SGConv was motivated by studying S4 as a pure convolutional model, which resulted in an alternative simple way to generate a long convolution kernel with a compressed parameterization.
It is not currently supported in this codebase, but should be straightforward to implement by adding another convolution kernel to [[/src/models/sequence/kernels/kernel.py](/src/models/sequence/kernels/kernel.py)].

The official repository is at [ctlllll/SGConv](https://github.com/ctlllll/SGConv).

### Mega (ICLR 2023)

> **Mega: Moving Average Equipped Gated Attention**\
> Xuezhe Ma, Chunting Zhou, Xiang Kong, Junxian He, Liangke Gui, Graham Neubig, Jonathan May, Luke Zettlemoyer\
> Paper: https://arxiv.org/abs/2209.10655

Mega introduces a simplification of S4 motivated to look like a vanilla exponential moving average (EMA).
This component is very similar to S4D in that it can be viewed as a diagonal SSM, with some differences in the parameterization (e.g. discretization and initialization).

In addition to the EMA, Mega introduces an efficient attention variant and a DNN block design that combines EMA with attention.

The main components of the [official Mega code release](https://github.com/facebookresearch/mega) has been largely ported to this repo in a modular way.
The drop-in alternative for S4(D) is `EMAKernel` at [[/src/models/sequence/kernels/kernel.py](/src/models/sequence/kernels/kernel.py)].
The Mega block is at [[/src/models/sequence/modules/megablock.py](/src/models/sequence/modules/megablock.py)], which is written as a more generic convolution + attention block that can accept any other type of long convolution kernel (e.g. with S4 instead of EMA).

See [[/configs/experiment/mega/lra-image](/configs/experiment/mega/lra-image)] for more details of the implementation and a subset of ablations on EMA vs S4 kernels.


### Liquid S4 (ICLR 2023)

> **Liquid Structural State-Space Models**\
> Ramin Hasani, Mathias Lechner, Tsun-Hsuan Wang, Makram Chahine, Alexander Amini, Daniela Rus\
> Paper: https://arxiv.org/abs/2209.12951

Liquid S4 introduces an extension of the original (DPLR) S4 model with ideas from [liquid time-constant networks](https://arxiv.org/abs/2006.04439).
This model is not supported in this codebase, but the official repository for Liquid S4 was forked around v2 and updated for v3: https://github.com/raminmh/liquid-s4

### H3 (ICLR 2023)

> Hungry Hungry Hippos: Towards Language Modeling with State Space Models\
> Tri Dao, Daniel Y. Fu, Khaled K. Saab, Armin W. Thomas, Atri Rudra, Christopher Ré\
> Paper: https://arxiv.org/abs/2212.14052

H3 is an architecture built around SSMs designed for language modeling. The core module is a black box application of a linear S4(D) layer, and H3 also introduces a shift SSM which is very similar to a vanilla (separable) local convolution.
It also provides a more efficient CUDA implementation of FFT convolution combined with the state-passing feature of S4.

The official H3 implementation is at [HazyResearch/H3](https://github.com/HazyResearch/H3).
It is currently not supported in this repository, but there are plans to add a module for the block and port in the faster FlashConv implementation of FFT convolution.
