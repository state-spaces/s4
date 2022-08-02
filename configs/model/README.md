
The `model/` configs largely follow the structure of the `src/models/` code folder.

## Backbones
Top-level configs use the model backbone structure specified by this repository.
These models consist of **backbones** that are composed of repeatable blocks of core **layers**.
The backbones include a simple isotropic residual backbone (in the style of ResNets and Transformers) (`base.yaml`) and variations of UNet structures (`unet.yaml`, `sashimi.yaml`).

## Layers
Layers configs are defined in `model/layer/`. Each one of these instantiates a `src.models.sequence.base.SequenceModule` which maps an input sequence to output sequence, and can be passed into the various backbones.
Older versions of HiPPO focused on RNNs, and defined a flexible RNN layer in `model/layer/rnn.yaml`. This RNN accepts any RNN cell, with example configs in `model/layer/cell/`.

## Examples
Some examples of full models are provided which combine a backbone with a choice of inner layer, such as `convnet1d.yaml` (a simple 1D residual convnet), `s4.yaml` (basic isotropic S4 model), and `transformer.yaml` (isotropic Transformer model composed of alternating layers of self-attention and feed-forward network).

## Other Baselines

Other baseline models are included that do not necessarily follow this structure.
```
baseline/      Miscellaneous baselines from the literature
nonaka/        1-D CNN models ported from the paper [Nonaka, Seita]
               "In-depth Benchmarking of Deep Neural Network Architectures for ECG Diagnosis"
               (https://github.com/seitalab/dnn_ecg_comparison)
segmentation/  Segmentation models (preliminary)
timm/          Ports of timm ResNet and ConvNext models
vit/           Ports of vit models
```
