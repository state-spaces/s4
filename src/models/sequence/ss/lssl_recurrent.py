""" DEPRECATED implementation of recurrent version of LSSL with variable step sizes """

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
from einops import rearrange, repeat
from omegaconf import DictConfig

from src.models.hippo import transition
from src.models.sequence.ss.linear_system_recurrence import linearsystem
from src.models.functional.toeplitz import causal_convolution
from src.models.sequence.base import SequenceModule


class RecurrentLSSL(SequenceModule):
    """ Compute LSSL recurrently

    - currently not used
    - could be useful for handling variable step sizes
    """

    def __init__(
            self, d,
            # memory_order,
            d_model, # overloading this term
            dt_min=0.01,
            dt_max=1.0,
            measure='legt',
            channels=None,
            # discretization='bilinear',
            init='normal', # for debugging, but might be useful?
            dropout=0.0,
        ):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        """

        if dropout > 0.0:
            raise NotImplementedError("Dropout currently not supported for Recurrent LSSL")

        super().__init__()
        self.N = d_model
        self.d = d
        self.channels = channels

        dt = torch.exp(torch.linspace(math.log(dt_min), math.log(dt_max), self.d))
        self.register_buffer('dt', dt)
        # self.dt = dt

        # Construct transition
        if measure == 'identity':
            A, B = torch.eye(self.N), torch.ones(self.N)
            self.transition = transition.ManualAdaptiveTransition(self.N, A, B)
        elif measure == 'legt':
            self.transition = transition.LegTAdaptiveTransition(self.N)
        elif measure == 'lagt':
            self.transition = transition.LagTCumsumAdaptiveTransition(self.N)
        else:
            raise NotImplementedError

        if self.channels is None:
            self.m = 1
        else:
            self.m = self.channels

        if init == 'normal':
            self.C = nn.Parameter(torch.randn(self.d, self.m, self.N))
            self.D = nn.Parameter(torch.randn(self.d, self.m))
        elif init == 'constant':
            self.C = nn.Parameter(torch.ones(self.d, self.m, self.N))
            self.D = nn.Parameter(torch.ones(self.d, self.m))

    def forward(self, u, return_output=True):
        """
        u: (L, B, D)
        """
        dt = self.dt.repeat((u.shape[0], u.shape[1], 1))
        y = linearsystem(None, dt, u, self.C, self.D, self.transition) # (L, B, D, M)

        if self.channels:
            output = y.sum(dim=-2) # (L, B, M)
        else:
            output = y.squeeze(-1) # (L, B, D)
        return output, output[-1]

    def default_state(self, *batch_shape, device=None):
        return torch.zeros(*batch_shape, self.N, device=device)

    def step(self, x, state):
        raise NotImplementedError("Needs to be implemented.")

    @property
    def d_state(self):
        return self.d

    @property
    def d_output(self):
        return self.d

    @property
    def state_to_tensor(self):
        return lambda state: state
