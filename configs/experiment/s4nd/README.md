# S4ND experiments and configs

This directory contains the main classification experiments for S4ND, which include ImageNet, Cifar-10, Celeb-A, and HMDB-51 (video), and the resolution experiments for CIFAR-10.

## Download ImageNet if needed

The ImageNet dataset will need to be downloaded and extracted into the `data/` directory.

To download ImageNet, download each dataset in data/. eg., commands to download each consecutively (a few hrs or so)

```wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar && wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar && wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz```

Download this script [here](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh) (into data/) and run it:

`sh extract_ILSVRC.sh`  # this will extract and parse files into the appropriate dirs

## ImageNet (1D and 2D)

The main 2D experiments use ViT and ConvNeXt. Here are sample commands to launch experiments to train both models from scratch on ImageNet with and without S4ND.

#### ViT (baseline)

```python -m train wandb=null experiment=s4nd/vit/vit_b_16_imagenet trainer.devices=8 loader.batch_size=128 loader.num_workers=12 optimizer.weight_decay=0.03 model.dropout=0.0```

#### ViT-S4ND

```python -m train wandb=null experiment=s4nd/vit/vit_b_16_s4_imagenet_v2 trainer.devices=8 loader.batch_size=128 loader.num_workers=12 optimizer.weight_decay=0.03 model.dropout=0.0 model.use_cls_token=false model.use_pos_embed=false model.layer.postact=glu model.layer.channels=2```

#### ConvNeXt (baseline)

In the ConvNeXt architecture, you can specify the type of conv layers in the stem (which is the initial downsample layer), (all other) downsample layers, and blocks.

`stem_type` 2 main options (see model script for more options):
- "patch" (standard 2D conv)
- "new_s4nd_patch" (S4ND)

`downsample_type`:
- null (standard 2d conv)
- "s4nd"

`layer` controls the main conv layers in the blocks:
- null (standard 2D conv)
- "s4nd"
    
Example ConvNeXt-2D baseline

```python -m train wandb=null experiment=s4nd/convnext/convnext_timm_tiny_imagenet loader.batch_size=512 loader.batch_size_eval=512 trainer.devices=8 optimizer.lr=4e-3 loader.num_workers=12 trainer.max_epochs=300 optimizer.weight_decay=0.05```

#### ConvNeXt-S4ND

Example ConvNeXt-2D-S4ND

```python -m train wandb=null experiment=s4nd/convnext/convnext_timm_tiny_s4nd_imagenet loader.batch_size=240 loader.batch_size_eval=240 trainer.device=8 optimizer.lr=4e-3 train.global_batch_size=3840 +model.layer.contract_version=0 trainer.max_epochs=300```

## CIFAR-10

The CIFAR dataset will be automatically downloaded to `data/` if not already there.

#### Conv2D ISO

Sample command for Conv2D model (baseline)

```python -m train experiment=s4nd/cifar/cnn-cifar-2d wandb=null```

#### S4ND ISO

Sample command for S4ND model

```python -m train experiment=s4nd/cifar/s4-cifar-2d wandb=null dataset.augment=True model.layer.l_max=[32,32] loader.train_resolution=1 loader.img_size=32 model.layer.d_state=64 model.layer.final_act=glu model.layer.bidirectional=True trainer.max_epochs=100 scheduler.num_training_steps=90000 model.d_model=512 model.n_layers=6 model.dropout=0.1 optimizer.weight_decay=0.05 model.layer.dt_max=1.0 model.layer.dt_min=0.1 model.layer.bandlimit=null model.layer.init=legs model.layer.n_ssm=1 model.layer.rank=1```

#### Zero-shot resolution

To run the zero-shot resolution experiment on CIFAR, we use the flags:
 - `loader.train_resolution`, int, where it divides the input base resolution.  So `1` means full res, `2` means half res
 `loader.eval_resolutions`, a list of ints, where it'll evaluate at each resolution factor. eg [1, 2] is evaluating at full and half res.

See config in this command for a sample zero-shot resolution experiment:

```python -m train experiment=s4nd/cifar/s4-cifar-2d-16x16 wandb=null loader.train_resolution=2 loader.eval_resolutions=[1,2]```

#### Progressive resizing

For progressive resizing experiments, we provide 2 examples:

#### Conv2d (baseline)

```python -m train experiment=s4nd/progres/cnn-cifar-2d wandb=null```

#### S4ND

```python -m train experiment=s4nd/progres/s4-cifar-2d wandb=null```

## Celeb-A

The Celeb-A dataset will be automatically downloaded to `data/` if not already there, but the shared drive might be over it's download limit. 

You can also try this [link](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset?resource=download) manually: , and unzip it into `data/celeba/celeba`.

#### ConvNeXt-Micro (baseline) 

```python -m train wandb=null experiment=s4nd/celeba/convnext-celeba-all trainer.devices=1 loader.batch_size=8 loader.num_workers=8```

#### ConvNeXt-Micro-S4ND

```python -m train wandb=null experiment=s4nd/celeba/convnext-s4nd-celeba-all trainer.devices=1 loader.batch_size=8 loader.num_workers=8```

## HMDB-51 (3D) video experiments

The HMDB-51 dataset will need to be downloaded and extracted to `data/`.  You can follow the procedure from this [link](https://cv.gluon.ai/build/examples_datasets/hmdb51.html#sphx-glr-download-build-examples-datasets-hmdb51-py).

Or follow our detailed notes for downloading [here.](https://docs.google.com/document/d/1AU9CnKyrhs4OidcYXsl82WlwvVEZr2wkZmEUTdd3sm0/edit?usp=sharing)

#### ConvNeXt-3D (baseline): I3D inflation

We provide sample launch commands for the video experiments, which leverage 2D to 3D kernel inflation using ImageNet pretrained weights. 

ConvNeXt-3D baseline (S3D/I3D inflation), with temporal kernel inflation using a pretrained 2D model

```python -m train wandb=null experiment=s4nd/convnext/convnext_timm_tiny_inflate3d_hmdb trainer.devices=1 loader.num_workers=12 dataset.split_dir=testTrainMulti_7030_splits dataset.video_dir=videos dataset.clip_duration=2 dataset.num_frames=30 model.tempor_patch_size=2 train.pretrained_model_state_hook._name_=convnext_timm_tiny_s4nd_2d_to_3d train.pretrained_model_path=/home/workspace/hippo/outputs/2022-11-12/08-55-04-224200/checkpoints/val/accuracy.ckpt loader.batch_size=8 train.global_batch_size=64 optimizer.lr=0.0001 scheduler.warmup_t=0 scheduler.t_initial=50 trainer.max_epochs=50 model.drop_path_rate=0.2 model.drop_head=0.2 dataset.augment=randaug dataset.randaug.num_layers=1 dataset.randaug.magnitude=5 optimizer.weight_decay=0.2 model.factor_3d=False```

Notes on dataloader/model hyperparameters:
- To train from scratch, set `train.pretrained_model_state_hook._name_=null` and `train.pretrained_model_path=null`
- For the ConvNeXt-3D baseline, we use 2 types of temporal 3D kernels, S3D and I3D (see paper). To use S3D inflation, set `model.factor_3d=True`, to use I3D, set `model.factor_3d=False`.
- `dataset.clip_duration` is in seconds, and controls the clip duration (see Pytorch video library for sampling details)
- `dataset.num_frames` control number of frames in the clip
- similar to ConvNeXt-2D, you can select which Conv layer type (S4ND or local Conv2D) for the `Stem`, `Downsample` or `Block`.
- `train.pretrained_model_state_hook` indicates using the kernel inflation, which also requires passing a pretrained ConvNeXt-2D model path, via `train.pretrained_model_path`

#### ConvNeXt-3D-S4ND

For the S4ND video model, we show an example of how to customize the temporal kernel initialization via the `post_init_hook` and its hyperparameters.

```python -m train wandb=null experiment=s4nd/convnext/convnext_timm_tiny_inflate3d_s4nd_hmdb trainer.devices=1 loader.num_workers=12 dataset.split_dir=testTrainMulti_7030_splits dataset.video_dir=videos train.remove_test_loader_in_eval=True dataset.clip_duration=2 dataset.num_frames=30 model.tempor_patch_size=2 train.pretrained_model_path=/home/workspace/hippo/outputs/2023-02-24/01-50-15-857995/checkpoints/val/accuracy.ckpt model.stem_type=patch model.downsample_type=null model.stem_l_max=null train.pretrained_model_state_hook._name_=convnext_timm_tiny_s4nd_2d_to_3d loader.batch_size=8 train.global_batch_size=64 optimizer.lr=0.0001 scheduler.warmup_t=0 scheduler.t_initial=50 trainer.max_epochs=50 model.drop_path_rate=0.2 model.drop_head=0.2 dataset.augment=randaug dataset.randaug.num_layers=1 dataset.randaug.magnitude=7 optimizer.weight_decay=0.2 +train.pretrained_model_reinit_hook._name_=_reinit +train.pretrained_model_reinit_hook.measure=hippo +train.pretrained_model_reinit_hook.n_ssm=null +train.pretrained_model_reinit_hook.deterministic=True +train.pretrained_model_reinit_hook.normalize=False +train.pretrained_model_reinit_hook.dt_min=2.0 +train.pretrained_model_reinit_hook.dt_max=2.0 train.pretrained_model_reinit_hook.dt_min=2.0 train.pretrained_model_reinit_hook.dt_max=2.0```




