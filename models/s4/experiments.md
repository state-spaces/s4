This README provides configs for various experiments in the S4 papers.

As documented in the main README, adding `wandb=null` to any command line turns off logging.

Some of these datasets may require downloading and preparing data, documented in the [src/dataloaders](/src/dataloaders/) subdirectory.

## Long Range Arena (LRA)

The latest LRA results are reported in the [HTTYH](https://arxiv.org/abs/2206.12037) paper, which achieves over 86% average.

```
python -m train experiment=lra/s4-lra-listops
python -m train experiment=lra/s4-lra-imdb
python -m train experiment=lra/s4-lra-cifar
python -m train experiment=lra/s4-lra-aan
python -m train experiment=lra/s4-lra-pathfinder
python -m train experiment=lra/s4-lra-pathx
```

To help reproduce results and sanity check, this table lists approximate final performance, intermediate performance, and timing information.


|                        | listops  | imdb     | aan        | cifar     | pathfinder | pathx      |
| ---                    | ---      | ---      | ---        | ---       | ---        | ---        |
| **Final Accuracy**     | 59.5     | 86.5     | 91.0       | 88.5      | 94.0       | 96.0       |
| **acc @ epoch**        | 50 @ 10  | 80 @ 10  | 80 @ 10    | 80 @ 20   | 90 @ 20    | 92 @ 10    |
| **time / epoch (GPU)** | 15m (T4) | 17m (T4) | 23m (A100) | 2m (A100) | 7m (A100)  | 56m (A100) |

### V1
The configs for the original version of the S4 paper (ICLR 2022) can be run with the following commands.
```
python -m train experiment=lra/old/s4-lra-listops
python -m train experiment=lra/old/s4-lra-imdb
python -m train experiment=lra/old/s4-lra-cifar
python -m train experiment=lra/old/s4-lra-aan
python -m train experiment=lra/old/s4-lra-pathfinder
python -m train experiment=lra/old/s4-lra-pathx
```

NOTE: These configs are meant for the first version of the S4 model, which is saved in a tag: `git checkout v1`

## CIFAR-10

```
python -m train experiment=cifar/s4-cifar
```

The above command line reproduces our best sequential CIFAR model.
Note that it is possible to get fairly good results with much smaller models.
The small [ablation models](#s4d-ablations) are one example, and the
[example.py](../example.py) script is another example.

## Speech Commands (SC)

The latest SC config reported in the S4D paper can be run with
```
python -m train experiment=sc/s4-sc
```

### SC10
The original S4 paper used a smaller 10-way classification task used in [prior](https://arxiv.org/abs/2005.08926) [work](https://arxiv.org/abs/2102.02611).

This version can be toggled either with `dataset=sc dataset.all_classes=false` or `dataset=sc10`.

The original S4 config can be run using V1 of this code using
```
python -m train experiment=old/s4-sc
```

## WikiText-103

V3 re-trained the WikiText-103 experiment with the latest model and a larger context size.
The trained checkpoint can be found [here](https://https://huggingface.co/krandiash/sashimi-release/checkpoints).
```
python -m train experiment=lm/s4-wt103
```

The default settings require 8 GPUs with 40GB memory. Modifications can be made by decreasing batch size and accumulating gradients, e.g. add `loader.batch_size=4 trainer.accumulate_grad_batches=2` to the command line.

Autoregressive generation can be performed with this checkpoint following the instructions in the main [README](README.md#generation)

## Time Series Forecasting

The ETTH, ETTM, Weather, and ECL experiments originally from the [Informer]() paper are supported.
Download the [data](https://drive.google.com/file/d/1XqpxE6cthIxKYviSmR703yU45vdQ1oHT/view?usp=sharing) to `./data`, and unzip `informer.zip` inside that folder.

```
python -m train experiment=forecasting/s4-informer-{etth,ettm,ecl,weather}
```

## S4D Ablations

The [S4D](https://arxiv.org/abs/2206.11893) paper uses small models on varied tasks to perform extensive ablations.
```
python -m train experiment=cifar/s4-cifar-ablation
python -m train experiment=bidmc/s4-bidmc-ablation
python -m train experiment=sc/s4-sc-ablation
```

## SaShiMi

Configs for models and baselines from the SaShiMi paper can be found under [configs/audio/](configs/audio/) and run with
```
python -m train experiment=audio/{sashimi,samplernn,wavenet}-{sc09,youtubemix,beethoven}
```
More documentation can be found in the [SaShiMi README](sashimi/README.md).
