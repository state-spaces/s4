## Changelog

### Roadmap

<!--
- Incorporate FlashConv implementation of faster FFT convolution.
- Add setup.py file for independent installation
-->


### [4.0.0] - 2023-03-31


#### Breaking Changes to Models
- The CUDA kernel has been updated and must be recompiled.
- A few parameters inside the S4(D) kernels have had their name change

To address differences between models trained on earlier versions and the current V4:
- The CUDA kernel should be re-compiled if moving between versions of this codebase.
- The script `checkpoints/port_v3_to_v4.py` can be used to convert models (see below).


#### New models
- [S4ND](models/s4nd/)
- Recent new models based on or closely related to S4, such as [GSS and Mega](models/related/)
- Other [long convolution kernels](src/models/sequence/kernels/) such as simple "wide kernel CNN" baseline (`model.layer.mode=conv`)


#### Repository Restructuring

- Information about specific papers and models (e.g. model description, overview of code, documentation of experiments) have been moved into the `models/` folder.
- Standalone S4 module has been moved from `src/models/s4/` to `models/s4/`.
- General sequence modeling framework under [src/models/sequence/](src/models/sequence/) has been reorganized. The old state space modules `src/models/sequence/ss/` have been removed; the S4 module has been broken into a generic convolution block in [src/models/sequence/modules/](src/models/sequence/modules/) and the inner linear SSM kernel moved to [src/models/sequence/kernels/](src/models/sequence/kernels/).
- More experiments have been added to [configs/experiments/](configs/experiments/) with improved structuring.


#### New CUDA Kernels
- The Cauchy CUDA kernel has been updated and must be recompiled.
- There is now a CUDA kernel for the Vandermonde operation of S4D, speeding it up over the naive and `pykeops` versions. S4D should now be faster than S4 in all versions (naive, pykeops, or CUDA kernel).

#### New Utility Scripts
- The `/checkpoints/` folder can be used to score checkpoints and contains several scripts for working with them. See `/checkpoints/README.md` for detailed usage.
- `/checkpoints/evaluate.py` takes a trained model and prints metrics on evaluation datasets.
- `/checkpoints/port_v3_to_v4.py` converts a model from V3 to V4 code.


#### S4 layer
- `model.layer.measure` has been renamed to `model.layer.init`. The name `measure` originally referred to approximation measures in the HiPPO theory, but they are only used as initialization in trainable SSM models. There are also many more initializations not based on the HiPPO theory, even the simple S4D-Lin model from the [minimal S4D standalone](models/s4/).
- TODO document some of the new features


### [3.0.0] - 2022-08-03

#### Models and Features
- Updated version of S4 module, including new measures and theory from [[How to Train Your HiPPO](https://arxiv.org/abs/2206.12037)] (https://github.com/HazyResearch/state-spaces/issues/21, https://github.com/HazyResearch/state-spaces/issues/54)
- Complete version of S4D module from [[On the Parameterization and Initialization of Diagonal State Space Models](https://arxiv.org/abs/2206.11893)]
- [State forwarding](models/s4/README.md#state-forwarding) (https://github.com/HazyResearch/state-spaces/issues/49, https://github.com/HazyResearch/state-spaces/issues/56)
- Support for S4 variants including DSS and GSS ([documentation](models/s4/README.md#other-variants))


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


### [2.1.0] - 2022-05-01
- Minor updates to S4 modules
- By default, S4 no longer requires installing Pykeops or a custom CUDA kernel.
- New S4D (S4-diagonal) standalone model found at `src/models/sequence/ss/standalone/s4d.py`. Simple variant using diagonal SSMs that recovers S4's performance on most tasks. Can be run with any existing experiment config with the additional flag `model/layer=s4d` on the command line.
- New [LRA configs](#long-range-arena-lra) for updated S4 code, with an average score of ~86

### [2.0.0] - 2022-02-27
Code release for SaShiMi audio model

### [1.1.0] - 2022-01-29
Added configs for time series datasets from the Informer paper (https://github.com/HazyResearch/state-spaces/issues/4)

### [1.0.0] - 2021-11-18
First release of this repository containing the S4 module and configs to reproduce sCIFAR, Speech Commands, Long Range Arena, and WikiText-103 results

