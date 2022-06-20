# S4 Simple

This is the code for the blog post [Simplifying S4](https://hazyresearch.stanford.edu/blog/2022-06-11-simplifying-s4).
We present a simplified version of the S4 kernel with diagonal matrices and fewer learnable parameters.

You can find the kernel in the `s4_simple.py` file.

## Running the Code

To run the code, first follow the install instructions of the overall repo.

Then, running the code is as simple as (from the root directory of this repo):
```
python -m train experiment=s4-simple-cifar wandb=null
```
(You can remove `wandb=null` if you want to log the run to WandB.)
This code should reach 83-84% val accuracy on CIFAR10.

By default, the kernel ignores the initial state (fusing `b` and `c`), and only trains the `a` parameters (leaving `theta` fixed to the initialization).
You can play with these parameters in the training run:
* Adding `use_initial=true` will add a learnable initial state, and learn the `b` and `c` parameters separately.
* Setting `learn_theta=true` will make the `theta` parameters learnable (we usually see a decrease in performance of about 3 points from this).
* Setting `learn_a=false` will make the `a` parameters not learnable. We don't see much of a performance degradation on CIFAR in this case, which speaks to the utility of the Chebyshev initialization!
* Setting `zero_order_hold=false` will switch from Zero-Order Hold to left-end-point quadrature. Additionally setting `trap_rule=true` will switch to the trapezoid rule (when `zero_order_hold` is set to `false`).

## Other Files

There are two other files in this folder, `s4_wrapper.py` and `utils.py`.
They contain some standard wrapper classes and utils for integrating into the state spaces code base.
