## Papers

### HiPPO (NeurIPS 2020)
![HiPPO Framework](/assets/hippo.png "HiPPO Framework")
> **HiPPO: Recurrent Memory with Optimal Polynomial Projections**\
> Albert Gu*, Tri Dao*, Stefano Ermon, Atri Rudra, Christopher Ré\
> Paper: https://arxiv.org/abs/2008.07669

### HTTYH (ICLR 2023)

![HTTYH](/assets/httyh.png "Basis Functions for S4 Variants")
> **How to Train Your HiPPO: State Spaces with Generalized Orthogonal Basis Projections**\
> Albert Gu*, Isys Johnson*, Aman Timalsina, Atri Rudra, Christopher Ré\
> Paper: https://arxiv.org/abs/2206.12037

## Models

### HiPPO

The original HiPPO paper produced three main methods.
1. LegT is the same as the prior method [LMU (Legendre Memory Unit)](https://proceedings.neurips.cc/paper/2019/file/952285b9b7e7a1be5aa7849f32ffff05-Paper.pdf). It defines specific formulas for A and B matrices in a time-invariant ODE $x'(t) = Ax(t) + Bu(t)$.
2. LagT is another time-invariant ODE $x'(t) = Ax(t) + Bu(t)$ meant to memorize history according to a different weighting function.
3. LegS was the main new method which produces a time-varying ODE $x'(t) = 1/t Ax(t) + 1/t Bu(t)$ meant to memorize history according to a uniform measure.

These methods were incorporated into a simple RNN called HiPPO-RNN where the measures $(A, B)$ were non-trainable.


### S4

S4 incorporated HiPPO methods into time-invariant state space models. The original version of S4 (and LSSL) primarily used the LegS matrices $(A, B)$, but used them in a *time-invariant* SSM $x'(t) = Ax(t) + Bu(t)$.
At the time, this was purely an empirical phenomenon: LSSL was originally designed to use the HiPPO-LegT matrices, and the LegS matrices were tested empirically and found to often be better.
At the time of the original S4 paper, this phenomenon was not understood.

### HTTYH


The HTTYH paper improved the HiPPO framework, simplifying the original derivations and generalizing to more methods. Some highlights include:
1. A general interpretation of the convolutional view of SSMs; in particular the correspondence between $(A, B, C)$ and the SSM's convolution kernel
2. An explanation of the time-invariant LegS method used in S4, which actually corresponds to an exponentially-decaying weight function
3. A new method HiPPO-FouT which corresponds to Fourier convolution kernels
4. Discouraging the use of the original LagT method, which was motivated to capture an exponentially-decaying weight function, but actually does not (instead, LegS does)
5. An interpretation of the discretization parameter $\Delta$ with guidelines on how to initialize it

In summary, the main HiPPO methods which are useful according to current knowledge are
1. LegS (time-varying)
2. Legs (time-invariant), which corresponds to the main S4 variant
3. LegT (corresponding to the older LMU) or FouT, both of which approximate finite-length convolution kernels


## Code

### HiPPO-RNN
The original HiPPO-RNN models are available under [[/src/models/sequence/rnns](/src/models/sequence/rnns)],
including a brief explanation with [example command line](/src/models/sequence/README.md#rnns).
These RNN methods are now deprecated as they are thin (and perhaps clumsy) wrappers around the core HiPPO methods,
which are superceded by S4 and variants.

### HiPPO
The core HiPPO methods are just a set of equations and not end-to-end models.
The specific matrices are implemented in [[/src/models/hippo/hippo.py](/src/models/hippo/hippo.py)].
The connection between HiPPO/S4 matrices $(A, B)$ and convolution kernels is illustrated in [[/notebooks/ssm_kernels.ipynb](/notebooks/ssm_kernels.ipynb)].
The *online function reconstruction* theory is illustrated in [[/notebooks/hippo_function_approximation.ipynb](/notebooks/hippo_function_approximation.ipynb)].
The animation code can also be found in a [.py file](/src/models/hippo/visualizations.py) instead of notebook.

