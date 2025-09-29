
```
config.yaml  Main config
model/       Instantiates a model backbone (see src/models/)
dataset/     Instantiates a datamodule (see src/dataloaders/)
loader/      Defines a PyTorch DataLoader
task/        Defines loss, metrics, optional encoder/decoder (see src/tasks/)
pipeline/    Combination of dataset/loader/task for convenience
optimizer/   Instantiates an optimizer
scheduler/   Instantiates a learning rate scheduler
trainer/     Flags for the PyTorch Lightning Trainer class
callbacks/   Misc options for the Trainer (see src/callbacks/)
experiment/  Defines a full experiment (combination of all of the above configs)
generate/    Additional flags used by the generate.py script
```

This README provides a brief overview of the organization of this configs folder. These configs are composed to define a full Hydra config for every experiment.

## Overview
The main config is found at `configs/config.yaml`, which is an example experiment for Permuted MNIST. Different combinations of flags can be overridden to define alternate experiments. The config files in this folder define useful combinations of flags that can be composed. Examples of full configs defining end-to-end experiments can be found in [experiment/](experiment/).

Flags can also be passed on the command line.

<!--
The end-to-end training pipeline can broken down into the following rough groups, where group XX is found under `configs/XX/`:
```
model: the sequence-to-sequence model backbone (e.g. a src.models.sequence.SequenceModel)
dataset: the raw dataset (data/target pairs) (e.g. a pytorch Dataset)
loader: how the data is loaded (e.g. a pytorch DataLoader)
encoder: defines a Module that interfaces between data and model backbone
decoder: defines a Module that interfaces between model backbone and targets
task: specifies loss and metrics
```
Default combinations of dataset+loader+encoder+decoder+task are further consolidated into groups called `pipelines`.
-->

## Helpful Tips

### Inspect the Config
- At the beginning of running `train.py`, the full Hydra config is printed. This is very useful for making sure all flags were passed in as intended. Try running `python -m train` and inspecting the full base config.

### Class Instantiation
- Generally, most dictionaries in the config correspond exactly to the arguments passed into a Python class. For example, the configs in `model/`, `dataset/`, `loader/`, `optimizer/`, `scheduler/`, `trainer/` define dictionaries which each instantiate exactly one object (a PyTorch `nn.Module`, `SequenceDataset`, PyTorch [DataLoader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html), PyTorch optimizer, PyTorch scheduler, and [PyTorch LightningTrainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html)).

### Registries
- Instantiating objects is controlled by the very useful Hydra [instantiate](https://hydra.cc/docs/advanced/instantiate_objects/overview/) utility.
- In this codebase, instead of defining a `_target_=<path>.<to>.<module>`, we use shorthand names for each desired class (wherever a `_name_` attribute appears). The file `src/utils/registry.py` lists these shorthand names found in these configs to the full class path.

### Source Code Documentation
- Check READMEs for the source code. For example, the configs in [configs/model](model) correspond to classes in [src/models](../src/models), the configs in [configs/dataset](dataset) correspond to classes in [src/dataloaders](../src/dataloaders).

<!--
It is recommended to read the overview in `src/README.md` to fully understand how models, datasets, tasks, and pipelines are put together.
-->


## Example
```
configs/optimizer/adamw.yaml

_name_: adamw
lr: 0.001
weight_decay: 0.00
```

1. When composed into a larger config, this should define a dictionary under the corresponding sub-config name. For example, the config printed by `python -m train optimizer=adamw optimizer.weight_decay=0.1` includes the following dictionary, confirming that the flags were passed in correctly.
```
├── optimizer
│   └── _name_: adamw
│       lr: 0.001
│       weight_decay: 0.1
```

2. The file `src/utils/registry.py` includes an `optimizer` dictionary mapping `adamw: torch.optim.AdamW`.

3. The full optimizer config is equivalent to instantiating `torch.optim.AdamW(lr=0.001, weight_decay=0.1)`

## Models

The `model/` configs correspond to modules in `src/models/`.
See `model/README.md`.

## Datasets

The `dataset/` configs correspond to modules in `src/dataloaders/`.

## Loader

`loader/` configs are used to instantiate a dataloader such as PyTorch's `torch.utils.data.DataLoader`.
Other configs correspond to extensions of this found in the source file `src/dataloaders/base.py`, for example dataloaders that allow sampling the data at different resolutions.

## Tasks

A task is something like "multiclass classification" or "regression", and defines *how the model interfaces with the data*.
A task defines the loss function and additional metrics, and an optional encoder and decoder.
These configs correspond to modules in `src/tasks/`.

### Encoder/Decoder

The encoder is the interface between the input data and model backbone. It defines how the input data is transformed before being fed to the model.

The decoder is the interface between the model backbone and target data. It defines how the model's outputs are transformed so that the task's loss and metrics can be calculated on it.


## Optimizer/Scheduler

`optimizer/` and `scheduler/` configs are used to instantiate an optimizer class and scheduler class respectively.


## Pipeline
A pipeline consists of a dataset + loader + encoder + decoder + task (and optionally optimizer+scheduler).
This is sometimes what people refer to as a "task", such as the "CIFAR-10 classification task".
A pipeline fully defines a training scheme; combining a pipeline with a model specifies an end-to-end experiment.

Overally, a pipeline fully defines a training experiment aside from the model backbone. This means *any pipeline* can be flexibly combined with *any* model backbone to define an experiment, regardless of the dimensions of the data and model, e.g.: `python -m train pipeline=cifar model=transformer`.

### Example: sCIFAR

```
defaults:
  - /trainer: default
  - /loader: default
  - /dataset: cifar
  - /task: multiclass_classification
  - /optimizer: adamw
  - /scheduler: plateau

train:
  monitor: val/accuracy # Needed for plateau scheduler
  mode: max

encoder: linear

decoder:
  _name_: sequence
  mode: pool
```

1. The `trainer/default.yaml` and `loader/default.yaml` specify a basic PyTorch Lightning trainer and PyTorch DataLoader

2. The `dataset/cifar.yaml` defines a dataset object that specifies data and target pairs. In this case, the data has shape `(batch size, 1024, 1)` and the target has shape `(batch size,)` which are class IDs from 0-9.

3. The model is not part of the pipeline; any model can be combined with this pipeline as long as it maps shape `(batch size, 1024, d_input) -> (batch size, 1024, d_output)`

4. The task consists of an encoder, decoder, loss function, and metrics. The `encoder` interfaces between the input data and model backbone; this example specifies that the data will pass through an `nn.Linear` mapping dimension the data from `(batch size, 1024, 1) -> (batch size, 1024, d_input)`. The `decoder` will map the model's outputs from `(batch size, 1024, d_output) -> (batch size,)` by pooling over the sequence length and passing through another `nn.Linear`. Finally, the `multiclass_classification` task defines a cross entropy loss and Accuracy metric.

5. This pipeline also defines a target optimizer and scheduler, which are optional.


## Experiment

An experiment combines every type of config into a complete end-to-end experiment.
Generally, this consists of a pipeline and model, together with training details such as the optimizer and scheduler.
See [experiment/README.md](experiment/).
