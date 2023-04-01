Scripts for working with model checkpoints. Also serves as a convenient place to store specific checkpoints.


### Generation script
The generation script lies outside this folder and is documented in the main README.
An example usage is
```
python -m generate experiment=lm/s4-wt103 checkpoint_path=checkpoints/s4-wt103.ckpt n_samples=1 l_sample=16384 l_prefix=8192 decode=text

```

### Evaluation script

The evaluation script `evaluate.py` follows a similar interface to the generation script.
```
python -m checkpoints.evaluate wandb=null experiment=lm/s4-wt103 train.ckpt='/dfs/scratch1/albertgu/projects/hippo/checkpoints/new_wt103_test_new.ckpt' trainer.devices=1 loader.batch_size=1
```
Note that the numbers reported in papers are those logged during training, not numbers reported by this script, which may differ slightly.

### Converting .ckpt (PyTorch Lightning) checkpoint to .pt (PyTorch)
```
python -m checkpoints.convert_pl_to_pt checkpoints/<name>.ckpt
```
This example creates a file `checkpoints/<name>.pt`.

### Converting V3 model to V4
```
python -m checkpoints.port_v3_to_v4 checkpoint_path=checkpoints/s4-wt103-v3.ckpt
```
This script follows the structure of the generation script and supports a few more advanced options. You can convert the model and test it on a batch immediately by passing in `test_model=true`. This requires a valid experiment configuration so that a model and dataloader can be constructed. The two options for loading the `generate.py` script from either a `checkpoint_path` or `experiment_path` argument also apply here.
```
python -m checkpoints.port_v3_to_v4 test_model=true checkpoint_path=checkpoints/s4-wt103-v3.ckpt experiment=lm/s4-wt103 trainer.devices=1 loader.batch_size=1
```
