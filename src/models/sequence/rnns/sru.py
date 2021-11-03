""" Implementation of the Simple Recurrent Unit

https://arxiv.org/abs/1709.02755
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.models.sequence.rnns.cells import CellBase
from src.models.nn import LinearActivation
from src.models.sequence.base import SequenceModule

class SRUCell(CellBase):
    """ Implementation of the pure SRU cell that works with the models.rnn.RNN class """
    name = 'sru'

    valid_keys = ['fx', 'rx', 'bias']

    @property
    def default_initializers(self):
        return {
            'fx': 'xavier',
            'rx': 'xavier',
        }

    @property
    def default_architecture(self):
        return {
            'bias': True,
        }

    def __init__(
            self, d_input, d_model,
            skip='H', # Highway, Residual, None
            offset=True, # whether to use previous or current cell to compute highway gate
            **kwargs
        ):

        self.offset = offset
        self.skip = skip
        assert self.skip in ['H', 'R', 'N']

        super().__init__(d_input, d_model, **kwargs)

    def reset_parameters(self):
        self.W = LinearActivation(self.d_input, self.d_model, bias=self.architecture['bias'])
        # gate
        self.W_fx = LinearActivation(self.d_input, self.d_model, bias=True, initializer=self.initializers['fx'], activation='sigmoid')
        self.W_fc = nn.Parameter(torch.randn(self.d_model))

        # highway
        if self.skip == 'H':
            self.W_rx = LinearActivation(self.d_input, self.d_model, bias=True, initializer=self.initializers['rx'], activation='sigmoid')
            self.W_rc = nn.Parameter(torch.randn(self.d_model))

        # resize input
        if self.d_input != self.d_model:
            self.skip_transform = nn.Linear(self.d_input, self.d_model)
        else:
            self.skip_transform = nn.Identity()


    def forward(self, x, c):
        ### Update hidden state
        g = torch.sigmoid(self.W_fx(x) + self.W_fc * c)
        c_ = (1.-g) * c + g * self.W(x)

        if self.skip == 'H':
            if self.offset:
                r = torch.sigmoid(self.W_rx(x) + self.W_rc * c)
            else:
                r = torch.sigmoid(self.W_rx(x) + self.W_rc * c_)
            h = (1-r) * self.skip_transform(x) + r * c_
        elif self.skip == 'R':
            h = c_ + self.skip_transform(x)
        else:
            h = c_

        return h, c_

class SRURNNGate(nn.Module):
    """ The gate/cell state computation of SRU """
    def __init__(self, d_model, feedback=True):
        """
        feedback: control whether cell state feeds back into itself. If False, this is essentially a QRNN reduce
        """
        super().__init__()
        self.d_model = d_model
        self.feedback = feedback
        if self.feedback:
            self.W_fc = nn.Parameter(torch.randn(self.d_model))

    def forward(self, f, u):
        """
        f, u: (batch, length, dim)
        """

        # If no feedback, batch the sigmoid computation
        if not self.feedback:
            f = torch.sigmoid(f)

        c = f.new_zeros(f.shape[..., 1:, :], requires_grad=False)
        cs = []
        for f_, u_ in zip(torch.unbind(f, dim=-2), torch.unbind(u, dim=-2)):
            if self.feedback:
                f_ = torch.sigmoid(f_ + self.W_fc * c)
            c = (1.-f_) * c + f_ * u_
            cs.append(c)
        return torch.stack(cs, dim=0)


class SRURNN(SequenceModule):
    """ Full RNN layer implementing the SRU (not just a Cell) """

    def __init__(self, d_input, d_model, feedback=True, return_output=True, dropout=0.0):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.return_output = return_output

        self.W_fused = LinearActivation(d_input, 2*self.d_model, bias=True)
        self.C = SRURNNGate(d_model, feedback)

        if dropout > 0.0:
            raise NotImplementedError("Dropout currently not supported for SRU")

    def forward(self, x, return_output=True):
        ufr = self.W_fused(x)
        ufr = rearrange(ufr, 'b l (c d) -> b l c d', c=2)
        u, fx = torch.unbind(ufr, dim=2) # (B, L, H)
        c = self.C(fx, u) # (B, L, H)
        state = c[..., -1, :]
        if self.return_output:
            return c, state
        else:
            return None, state

    @property
    def d_state(self):
        return self.d_model

    @property
    def d_output(self):
        return self.d_model

    @property
    def state_to_tensor(self):
        return lambda state: state
    # TODO haven't checked the default_state, step functions
