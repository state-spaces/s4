# Papers and Models

This repository contains the official source code and experiments of several projects (papers and associated models).
This document contains a brief overview of the history of S4-related work and how they relate to each other.
More detailed project-specific information is available in individual subfolders, including information about where to find and how to use the models, any available standalone source code, and information about experiments for the projects.

## HiPPO (NeurIPS 2020 - Spotlight)
![HiPPO Framework](/assets/hippo.png "HiPPO Framework")
> **HiPPO: Recurrent Memory with Optimal Polynomial Projections**\
> Albert Gu*, Tri Dao*, Stefano Ermon, Atri Rudra, Christopher Ré\
> Paper: https://arxiv.org/abs/2008.07669

> HiPPO was motivated by improving RNNs to address long-range dependencies. It introduced the framework of *continuous-time memorization*, or how to efficiently memorize and reconstruct a continuous signal from its history. These result in a family of specific (non-trainable) linear ODEs $\dot{x}(t) = A(t)x(t) + B(t)u(t)$, which describes how a state $x$ should be updated to memorize the history of a function $u$, in order to optimally reconstruct that history. Although motivated independently of SSMs, these equations became the motivation and basis of follow-up SSM models.

## LSSL (NeurIPS 2021)

![Linear State Space Layer](/assets/splash.png "Properties of State Spaces")
> **Combining Recurrent, Convolutional, and Continuous-time Models with the Linear State Space Layer**\
> Albert Gu, Isys Johnson, Karan Goel, Khaled Saab, Tri Dao, Atri Rudra, Christopher Ré\
> Paper: https://arxiv.org/abs/2110.13985

> LSSL built on HiPPO, extending it into a full SSM with trainable parameters. Several characteristics of SSMs for deep learning were first described here, including the trainable $\Delta$ for discretization, and the dual recurrent/convolutional modes. However, it is incomplete in some ways; it had speed and memory issues, and did not actually leverage the recurrent/convolutional duality. Rather than labeling it as a separate model, we prefer to think of LSSL as a preliminary version of S4.

## S4 (ICLR 2022 - Outstanding Paper HM)

![Structured State Spaces](/assets/s4.png "Properties of Structured State Spaces")
> **Efficiently Modeling Long Sequences with Structured State Spaces**\
> Albert Gu, Karan Goel, Christopher Ré\
> Paper: https://arxiv.org/abs/2111.00396

> S4 made HiPPO-based SSMs efficient by introducing a particular parameterization of $A$ as a diagonal plus low-rank matrix (the DPLR parameterization). S4 made it possible to apply SSMs to a wide range of problems and substantially advanced benchmarks on long range modeling, while also showing how to achieve both fast training and inference by leveraging multiple representations of SSMs. 

## SaShiMi (ICML 2022 - Long Talk)

![SaShiMi](/assets/sashimi.png "SaShiMi Architecture")
> **It's Raw! Audio Generation with State-Space Models**\
> Karan Goel, Albert Gu, Chris Donahue, Christopher Ré\
> Paper: https://arxiv.org/abs/2202.09729

> SaShiMi primarily applied S4 to audio generation, showing how the properties of SSMs yield efficient training and autoregressive generation of audio waveforms, outperforming WaveNet architectures in a variety of settings. It also proposed some improvements to the SSM parameterization to improve stability for recurrent generation.

## DSS (NeurIPS 2022 - Spotlight)
> **Diagonal State Spaces are as Effective as Structured State Spaces**\
> Ankit Gupta, Albert Gu, Jonathan Berant\
> Paper: https://arxiv.org/abs/2203.14343

> Introduces DSS, an SSM layer that restricts $A$ to the set of (complex) diagonal matrices, as well as a parameterization for this layer that relies on a softmax convolutional kernel. An important observation of this work is that a simple modification of the HiPPO initialization used in S4 can enable diagonal versions to be performant in practice. Subsequent work on S4D simplifies the parameterization, and explains why the modified initialization works well in practice.

## HTTYH (ICLR 2023)

![HTTYH](/assets/httyh.png "Basis Functions for S4 Variants")
> **How to Train Your HiPPO: State Spaces with Generalized Orthogonal Basis Projections**\
> Albert Gu*, Isys Johnson*, Aman Timalsina, Atri Rudra, Christopher Ré\
> Paper: https://arxiv.org/abs/2206.12037

> Simplifies and generalizes the HiPPO theory, and introduces a few more initializations. Most importantly, this paper filled in a gap in the original S4 paper: the $(A, B)$ matrices used were actually not explained in the original HiPPO paper and were an empirical finding. HTTYH showed how to formalize them as HiPPO methods as well, and provides more intuition for how to interpret S4 as a convolutional model. This paper was developed concurrently to S4D and was originally meant to be a companion work preceding S4D; it introduces methods and theory that were used in follow-ups including S4D and S4ND.

## S4D (NeurIPS 2022)

![S4D](/assets/s4d.png "S4D: The diagonal variant of S4")
> **On the Parameterization and Initialization of Diagonal State Space Models**\
> Albert Gu, Ankit Gupta, Karan Goel, Christopher Ré\
> Paper: https://arxiv.org/abs/2206.11893

> While the original S4 paper focused on the DPLR (diagonal plus low-rank) parameterization to be able to incorporate HiPPO, it also outlined the much simpler diagonal SSM case, which was fleshed out in S4D. These diagonal structured SSMs are much simpler than DPLR structured SSMs while generally performing on par, and have become the method of choice in most downstream work. This paper also theoretically explains the empirical effectiveness of the DSS initialization, while concurrently developing simpler initializations not based on HiPPO (e.g. S4D-Lin).


## S4ND (NeurIPS 2022)

![S4ND](/assets/s4nd.png "S4ND: Multi-dimensional S4")
> **S4ND: Modeling Images and Videos as Multidimensional Signals Using State Spaces**\
> Eric Nguyen*, Karan Goel*, Albert Gu*, Gordon W. Downs, Preey Shah, Tri Dao, Stephen A. Baccus, Christopher Ré\
> Paper: https://arxiv.org/abs/2210.06583

> Applied SSMs to vision (images and video) by generalizing S4 to S4ND, which moves beyond sequences to modeling multidimensional signals. S4ND corresponds to running an S4 layer independently along each dimension of the signal, or equivalently creating a N-D convolution kernel as the outer product of N basic 1-D S4 kernels. Also introduces a new bandlimiting regularization for learning smoother SSM convolutional kernels, which substantially improves the ability to adapt to changes in sampling rates at test time.

## Other Related Projects

S4 has inspired many more related models, many of which are also forked from this repository.
These include GSS, SGConv, Mega, H3, Liquid S4, and more.
Several of these have also been reimplemented within this repository.
More information about these models, including links to relevant resources and repositories are documented in [[related/README.md](related/)].


# Citations

If you use this codebase, or otherwise found our work valuable, please cite the relevant papers.
```
@article{gu2020hippo,
  title={HiPPO: Recurrent Memory with Optimal Polynomial Projections},
  author={Gu, Albert and Dao, Tri and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}

@article{gu2021combining,
  title={Combining Recurrent, Convolutional, and Continuous-time Models with Linear State-Space Layers},
  author={Gu, Albert and Johnson, Isys and Goel, Karan and Saab, Khaled and Dao, Tri and Rudra, Atri and R{\'e}, Christopher},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}

@inproceedings{gu2022efficiently,
  title={Efficiently Modeling Long Sequences with Structured State Spaces},
  author={Gu, Albert and Goel, Karan and R\'e, Christopher},
  booktitle={The International Conference on Learning Representations ({ICLR})},
  year={2022}
}

@article{goel2022sashimi,
  title={It's Raw! Audio Generation with State-Space Models},
  author={Goel, Karan and Gu, Albert and Donahue, Chris and R{\'e}, Christopher},
  journal={International Conference on Machine Learning ({ICML})},
  year={2022}
}

@inproceedings{gu2023hippo,
  title={How to Train Your HiPPO: State Space Models with Generalized Basis Projections},
  author={Gu, Albert and Johnson, Isys and Timalsina, Aman and Rudra, Atri and R\'e, Christopher},
  booktitle={The International Conference on Learning Representations ({ICLR})},
  year={2023}
}

@article{gu2022s4d,
  title={On the Parameterization and Initialization of Diagonal State Space Models},
  author={Gu, Albert and Gupta, Ankit and Goel, Karan and R\'e, Christopher},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  year={2022}
}

@article{nguyen2022s4nd,
  title={S4ND: Modeling Images and Videos as Multidimensional Signals Using State Spaces
},
  author={Nguyen, Eric and Goel, Karan and Gu, Albert and Downs, Gordon W. and Shah, Preey and Dao, Tri and Baccus, Stephen A. and R\'e, Christopher},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  year={2022}
}
```
