if __name__ == '__main__':
    import sys
    import pathlib
    p = pathlib.Path().absolute()
    print("Adding path: ", p)
    sys.path.append(str(p))

import math
from types import new_class
from src.models.sequence.ss.s4 import StateSpace
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
from einops import rearrange, repeat
from omegaconf import DictConfig
import opt_einsum as oe

optimized = True

if optimized:
    contract = oe.contract
else:
    contract = torch.einsum

from src.models.nn.krylov import HippoKrylov
# from models.nn.components import TransposedLinear
from src.models.nn import LinearActivation, Activation
import src.utils.train

log = src.utils.train.get_logger(__name__)

class StateSpace2D(nn.Module):
    # transposed = True
    requires_length = True

    def __init__(
            self,
            H,
            l_max=None,
            d_model=64, # overloading this term, same as memory_order or N
            measure='legs', # 'legs', 'legt' main ones; can also try 'lagt'
            dt_min=0.001,
            dt_max=0.1,
            trainable=None,
            lr=None,
            rank=1,
            stride=1,
            w_bias=0.0,
            dropout=0.0,
            cache=False,
            weight_decay=0.0,
            # return_state=True,
            transposed=True,
            activation='gelu',
            postact=None,
            weight_norm=False,
            # glu=False,
            initializer=None,
            train_state=False,
            slow=False, # Use slow Krylov function for debugging
            test_resolution=False, # Assume all sequences are same length and different length sequences are subsampled differently
            tie=False,
            # absorb_c=True,
        ):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        """
        super().__init__()

        self.s4_1 = StateSpace(
            H,
            l_max=int(math.sqrt(l_max)),
            d_model=d_model,
            measure=measure,
            dt_min=dt_min,
            dt_max=dt_max,
            trainable=trainable,
            lr=lr,
            rank=rank,
            stride=stride,
            w_bias=w_bias,
            dropout=dropout,
            cache=cache,
            weight_decay=weight_decay,
            transposed=transposed,
            activation=activation,
            postact=postact,
            weight_norm=weight_norm,
            initializer=initializer,
            train_state=train_state,
            slow=slow,
            test_resolution=test_resolution,
        )

        self.s4_2 = StateSpace(
            H,
            l_max=int(math.sqrt(l_max)),
            d_model=d_model,
            measure=measure,
            dt_min=dt_min,
            dt_max=dt_max,
            trainable=trainable,
            lr=lr,
            rank=rank,
            stride=stride,
            w_bias=w_bias,
            dropout=dropout,
            cache=cache,
            weight_decay=weight_decay,
            transposed=transposed,
            activation=activation,
            postact=postact,
            weight_norm=weight_norm,
            initializer=initializer,
            train_state=train_state,
            slow=slow,
            test_resolution=test_resolution,
        ) if not tie else self.s4_1
        self.transposed = transposed
        log.info(f"Constructing s4 (H, N, L) = ({H}, {d_model}, {l_max})")

    def pre_forward(self, u):
        return u.reshape(u.shape[0], int(math.sqrt(u.shape[1])), int(math.sqrt(u.shape[1])), u.shape[2])

    def post_forward(self, y):
        return y.reshape(y.shape[0], y.shape[1] * y.shape[2], y.shape[3])

    def forward(self, u, state=None, cache=None, **kwargs): # absorbs return_output and transformer src mask
        """
        u: (B H L)
        Returns: (B H L)
        
        u: (B, H, W, C)
        Returns: (B, H, W, C)        
        """
        # TODO: simplify the transposes
        u = u.transpose(-1, -2)
        u = self.pre_forward(u)
        b, h, w, c = u.shape
        new_state = [None, None]
        y, new_state[0] = self.s4_1(u.reshape(b * h, w, c).transpose(-1, -2), state[0] if state else None)
        y = y.transpose(-1, -2)
        y, new_state[1] = self.s4_2(y.reshape(b, h, w, c).transpose(-2, -3).reshape(b * w, h, c).transpose(-1, -2), state[1] if state else None)
        y = y.transpose(-1, -2)
        y = y.reshape(b, w, h, c).transpose(-2, -3)
        y = self.post_forward(y)
        y = y.transpose(-1, -2)
        return y, new_state
        

    def step(self, u, state):
        raise NotImplementedError
    
    def default_state(self, *batch_shape, device=None):
        return [self._initial_state.repeat(*batch_shape, 1, 1)] * 2

    def loss(self):
        """ Extra train loss (implements weight decay for the filter).

        This is probably better than naive weight decay on the individual parameters A, B, C, dt, although we have not tested that.
        Prior work that parameterizes convolution filters implicitly (i.e. CKConv) also implement it this way.
        """
        return self.s4_1.loss() + self.s4_2.loss()

    @property
    def d_state(self):
        return self.s4_1.d_state

    @property
    def d_output(self):
        return self.s4_1.d_output

    @property
    def state_to_tensor(self):
        return lambda state: [self.s4_1.state_to_tensor(state[0]), self.s4_2.state_to_tensor(state[1])]


if __name__ == '__main__':
    from benchmark import utils

    device = 'cuda' # 'cpu'
    device = torch.device(device)

    ssm = StateSpace2D(
        H=256,
        l_max=28,
        d_model=64,
    )
    ssm.to(device)
    for module in ssm.modules():
        if hasattr(module, 'setup'): module.setup()
    x = torch.randn(2, 784, 256, device=device)

    y, state = ssm(x)
    print(y.shape)
