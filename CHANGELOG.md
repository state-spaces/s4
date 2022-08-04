## Changelog

### 2022-08-03 - [V3.0]

#### Models and Features
- Updated version of S4 module, including new measures and theory from [[How to Train Your HiPPO](https://arxiv.org/abs/2206.12037)] (https://github.com/HazyResearch/state-spaces/issues/21, https://github.com/HazyResearch/state-spaces/issues/54)
- Complete version of S4D module from [[On the Parameterization and Initialization of Diagonal State Space Models](https://arxiv.org/abs/2206.11893)]
- [State forwarding](src/models/s4/README.md#state-forwarding) (https://github.com/HazyResearch/state-spaces/issues/49, https://github.com/HazyResearch/state-spaces/issues/56)
- Support for S4 variants including DSS and GSS ([documentation](src/models/s4/README.md#other-variants))

<!--
####  Compilation of additional resources
  - Recommended resources for understanding S4-style models, including the [Simplifying S4 blog](https://hazyresearch.stanford.edu/blog/2022-06-11-simplifying-s4) ([code](https://github.com/HazyResearch/state-spaces/tree/simple/src/models/sequence/ss/s4_simple)) and a minimal pedagogical version of S4D ([code](src/models/s4/s4d.py))
  - Tips & Tricks page for getting started with tuning S4
-->

#### Bug fixes and library compatibility issues
- PyTorch 1.11 had a [Dropout bug](https://github.com/pytorch/pytorch/issues/77081) which is now avoided with a custom Dropout implementation (https://github.com/HazyResearch/state-spaces/issues/42, https://github.com/HazyResearch/state-spaces/issues/22)
- Conjugated tensors API change in PyTorch 1.10 (https://github.com/HazyResearch/state-spaces/issues/35)

#### SaShiMi
- Release of Sashimi+DiffWave model (https://github.com/HazyResearch/state-spaces/issues/46). Can be found at [albertfgu/diffwave-sashimi](https://github.com/albertfgu/diffwave-sashimi)

#### Generation
- Improved generation script for any models trained using this repository (https://github.com/HazyResearch/state-spaces/issues/38)

#### Model Checkpoints
- Re-trained SaShiMi models with the latest version of S4 (https://github.com/HazyResearch/state-spaces/issues/37, https://github.com/HazyResearch/state-spaces/issues/32)
- New WikiText-103 checkpoint with generation functionality (https://github.com/HazyResearch/state-spaces/issues/5, https://github.com/HazyResearch/state-spaces/issues/19)

#### HiPPO
- Release of new [notebook](notebooks/hippo_function_approximation.ipynb) (and equivalent .py [file](src/models/hippo/visualizations.py)) visualizing HiPPO function reconstruction. Includes animations used in HTTYH, the Annotated S4D, and various S4 talks.

#### Experiments
- Improved configs for Long Range Arena reported in HTTYH and S4D papers
- New datasets and ablation experiments from the S4D paper

Note that there have been various refactors and miscellaneous changes which may affect results slightly, but results should be close and general trends should hold. Feel free to file an issue for any results which do not match the papers.

#### Documentation
- Reorganized the [README](README.md) and added much more [documentation](README.md#readmes) for using this codebase


### 2022-05-01 - [V2.1]
- Minor updates to S4 modules
- By default, S4 no longer requires installing Pykeops or a custom CUDA kernel.
- New S4D (S4-diagonal) standalone model found at `src/models/sequence/ss/standalone/s4d.py`. Simple variant using diagonal SSMs that recovers S4's performance on most tasks. Can be run with any existing experiment config with the additional flag `model/layer=s4d` on the command line.
- New [LRA configs](#long-range-arena-lra) for updated S4 code, with an average score of ~86

### 2022-02-27 - [V2]
Code release for SaShiMi audio model

### 2022-01-29 - [V1.1]
Added configs for time series datasets from the Informer paper (https://github.com/HazyResearch/state-spaces/issues/4)

### 2021-11-18 - [V1]
First release of this repository containing the S4 module and configs to reproduce sCIFAR, Speech Commands, Long Range Arena, and WikiText-103 results

