## DSS
> **Diagonal State Spaces are as Effective as Structured State Spaces**\
> Ankit Gupta, Albert Gu, Jonathan Berant\
> Paper: https://arxiv.org/abs/2203.14343

The official DSS repository is located at https://github.com/ag1988/dss, which contains the original code and experiments.
It was forked from this repository between version 1.0 and 2.0.
A version is also available in this repository, detailed below.

### Model

DSS is the first diagonal SSM variant. The original version has two main characteristics:
1. *Computation* - uses a "softmax" which combines ZOH discretization + normalization over sequence length
2. *Initialization* - uses a diagonal approximation to the HiPPO matrix (also called HiPPO-LegS). The reason this approximation works was described in the [S4D paper](https://arxiv.org/abs/2206.11893)

Another version of DSS called DSS-exp constrains the real part of the eigenvalues of the $A$ matrix to be negative, a technique from [SaShiMi](https://arxiv.org/abs/2202.09729).
The softmax can then be dropped.

### Implementation

DSS-exp is essentially equivalent to S4D; see S4D documentation for usage.

The original DSS is also available by setting options in the general [S4D kernel](/src/models/sequence/kernels/ssm.py):
```
SSKernelDiag(
  init='diag-legs',  # A init is HiPPO approximation
  B_init='random',   # B init is random
  real_transform='none',  # No constraint on real part of A
  disc='dss',        # Apply DSS softmax
)
```

Note that these options can be passed in on the command line for any S4D experiment, for example:
```
python -m train experiment=lra/s4-lra-cifar model.layer.mode=diag model.layer.init=diag-legs +model.layer.B_init=random +model.layer.real_transform=none +model.layer.disc=dss
```
