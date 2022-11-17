## Mega: SSM Ablations (EMA and S4D)

Results:

| Model              | Params | s/epoch | Val Acc |
|--------------------|--------|---------|---------|
| (large) Mega-EMA   | 2.65M  |     166 |   82.60 |
| (large) Mega-S4D   | 2.65M  |     120 |   84.68 |
| (small) Mega-EMA   | 279K   |      48 |   79.98 |
| (small) Mega-S4D   | 279K   |      55 |   80.80 |
| (small) EMA        | 267    |      30 |   70.74 |
| (small) S4D        | 200    |      31 |   84.40 |

These runs correspond to the experiment files
`{large-mega,small-mega,small}-{ema-with-s4,s4d}.yaml`
described below.

```
python -m train experiment=mega/lra-image/large-mega-ema
```
Attempted reproduction of full LRA-Image model described in Mega paper, using the official PyTorch module, with all hyperparameters reproduced.


```
python -m train experiment=mega/lra-image/large-mega-ema-with-s4
```
Version of this model with minor changes matching the parameter count of S4.
This does not affect the speed or accuracy of the model; same performance as above.

```
python -m train experiment=mega/lra-image/large-mega-s4d
```
Same model but replacing the EMA component with original (complex) S4D.

----------

Small Mega models and additional details about these variants:

```
python -m train experiment=mega/lra-image/small-mega-ema
```
Small version of the same model (using the original Mega EMA layer), with the following changes:
- Model depth halved from 8 to 4
- Model width halved (`d_model`, `d_ffn`, `v` in Table 8)
- Weight decay halved from 0.02 to 0.01
- Training time halved from 200 to 100 epochs

Details: This calls the Mega block (https://github.com/HazyResearch/state-spaces/blob/mega/src/models/sequence/mega.py) which uses the MultiHeadEMA module (https://github.com/HazyResearch/state-spaces/blob/mega/src/models/sequence/ss/ema.py).
Both of these are transcribed from the official Mega code.

```
python -m train experiment=mega/lra-image/small-mega-ema-with-s4
```
Same as above, but computing the EMA module with a codepath that goes through the S4 block (does not use the actual S4 layer). Should be exactly the same as the above model.
This config only ablates that the alternate codepath behaves as expected, so that we can confidently replace smaller modules (the convolution kernel).

Details: This breaks the MultiHeadEMA module into two parts, the kernel construction and the convolution. The kernel construction module is `EMAKernel` ([code](https://github.com/HazyResearch/state-spaces/blob/17663f26f7e91f88757e1d61318ed216dfb8a8a5/src/models/sequence/ss/kernel.py#L869)) which can be seen as a drop-in replacement of alternative convolution kernels such as S4D's kernel.

The convolution goes through the [S4 block](https://github.com/HazyResearch/state-spaces/blob/mega/src/models/sequence/ss/s4.py), which is just a generic block implementing FFT convolution. The kernel inside is controllable by the `mode=ema` [option](https://github.com/HazyResearch/state-spaces/blob/17663f26f7e91f88757e1d61318ed216dfb8a8a5/src/models/sequence/ss/kernel.py#L1012)) to use the `EMAKernel`.

The swap from `MultiHeadEMA` to `S4Block`+`EMAKernel` is performed [here](https://github.com/HazyResearch/state-spaces/blob/17663f26f7e91f88757e1d61318ed216dfb8a8a5/src/models/sequence/mega.py#L95).


```
python -m train experiment=mega/lra-image/small-mega-s4d
```
Same as above, but replaces EMA with an S4D layer.

Details: Exactly the same as above but replaces `EMAKernel` with `SSKernelDiag`. Note that the latter has more features, but the minimal version of it ([here](https://github.com/HazyResearch/state-spaces/blob/17663f26f7e91f88757e1d61318ed216dfb8a8a5/src/models/s4/s4d.py#L16)) is nearly identical to the EMA kernel.

----------

The `small-{ema,s4d}.yaml` experiments use a block with only an SSM convolution.
This is a variant of the small ablation models from the S4D paper
(`configs/experiment/cifar/s4-cifar-ablation.yaml`).

```
python -m train experiment=mega/lra-image/small-s4d
```
Pure S4D module from the S4D paper (no Attention).

To make the models more directly comparable, some architecture flags were tweaked to match the Mega models (namely using pre-batch-norm rather than post-layer-norm),
which might lower these results slightly (~84) compared to the original S4D results (~86) for these model sizes.

```
python -m train experiment=mega/lra-image/small-ema
```
Same as above, but EMA instead of S4D (no Attention).

```
python -m train experiment=mega/lra-image/small-ema-with-s4d
```
Same as above, but use settings to match the parameter count of S4D.
