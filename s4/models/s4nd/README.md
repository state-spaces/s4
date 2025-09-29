## S4ND (NeurIPS 2022)

![S4ND](/assets/s4nd.png "S4ND: Multi-dimensional S4")
> **S4ND: Modeling Images and Videos as Multidimensional Signals Using State Spaces**\
> Eric Nguyen*, Karan Goel*, Albert Gu*, Gordon W. Downs, Preey Shah, Tri Dao, Stephen A. Baccus, Christopher Ré\
> Paper: https://arxiv.org/abs/2210.06583


### Model

The main S4ND model can be found in [[/src/models/sequence/modules/s4nd.py](/src/models/sequence/modules/s4nd.py)].
It is very similar to the [general CNN block](/src/models/sequence/modules/s4block.py) but is specialized to multi-dimensional inputs, while calling the S4 kernel as a black box.

### Experiments

The main S4ND experiments are located and documented at [[/configs/experiment/s4nd](/configs/experiment/s4nd)].
