""" Implements downsampling and upsampling on sequences """

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

from src.models.sequence import SequenceModule
from src.models.nn import LinearActivation

""" Simple pooling functions that just downsample or repeat

pool: Subsample on the layer dimension
expand: Repeat on the feature dimension
"""

def downsample(x, pool=1, expand=1, transposed=False):
    if x is None: return None
    if pool > 1:
        if transposed:
            x = x[..., 0::pool]
        else:
            x = x[..., 0::pool, :]

    if expand > 1:
        if transposed:
            x = repeat(x, '... d l -> ... (d e) l', e=expand)
        else:
            x = repeat(x, '... d -> ... (d e)', e=expand)

    return x

def upsample(x, pool=1, expand=1, transposed=False):
    if x is None: return None
    if expand > 1:
        if transposed:
            x = reduce(x, '... (d e) l -> ... d l', 'mean', e=expand)
        else:
            x = reduce(x, '... (d e) -> ... d', 'mean', e=expand)
    if pool > 1:
        if transposed:
            x = repeat(x, '... l -> ... (l e)', e=pool)
        else:
            x = repeat(x, '... l d -> ... (l e) d', e=pool)
    return x

class DownSample(SequenceModule):
    def __init__(self, d_input, pool=1, expand=1, transposed=True):
        super().__init__()
        self.d_input = d_input
        self.pool = pool
        self.expand = expand
        self.transposed = transposed

    def forward(self, x):
        return downsample(x, self.pool, self.expand, self.transposed)

    def step(self, x, state, **kwargs):
        if self.pool > 1 or self.expand > 1:
            raise NotImplementedError
        return x, state

    @property
    def d_output(self):
        return self.d_input * self.expand

class UpSample(nn.Module):
    def __init__(self, d_input, pool=1, expand=1, transposed=True):
        super().__init__()
        self.d_input = d_input
        self.pool = pool
        self.expand = expand
        self.transposed = transposed

    def forward(self, x):
        return upsample(x, self.pool, self.expand, self.transposed)

    @property
    def d_output(self):
        return self.d_input // self.expand
    def step(self, x, state, **kwargs):
        if self.pool > 1 or self.expand > 1:
            raise NotImplementedError
        return x, state


""" Pooling functions with trainable parameters """ # TODO make d_output expand instead
class DownPool(SequenceModule):
    def __init__(self, d_input, d_output, pool, transposed=True, weight_norm=True, initializer=None, activation=None):
        super().__init__()
        self._d_output = d_output
        self.pool = pool
        self.transposed = transposed

        self.linear = LinearActivation(
            d_input * pool,
            d_output,
            transposed=transposed,
            initializer=initializer,
            weight_norm = weight_norm,
            activation=activation,
            activate=True if activation is not None else False,
        )

    def forward(self, x):
        if self.transposed:
            x = rearrange(x, '... h (l s) -> ... (h s) l', s=self.pool)
        else:
            x = rearrange(x, '... (l s) h -> ... l (h s)', s=self.pool)
        x = self.linear(x)
        return x, None

    def step(self, x, state, **kwargs): # TODO needs fix in transpose ca, **kwargsse
        """
        x: (..., H)
        """

        if x is None: return None, state
        state.append(x)
        if len(state) == self.pool:
            x = rearrange(torch.stack(state, dim=-1), '... h s -> ... (h s)')
            if self.transposed: x = x.unsqueeze(-1)
            x = self.linear(x)
            if self.transposed: x = x.squeeze(-1)
            return x, []
        else:
            return None, state

    def default_state(self, *args, **kwargs):
        return []

    @property
    def d_output(self): return self._d_output


class UpPool(SequenceModule): # TODO subclass SequenceModule
    def __init__(self, d_input, d_output, pool, transposed=True, weight_norm=True, initializer=None, activation=None):
        super().__init__()
        self.d_input = d_input
        self._d_output = d_output
        self.pool = pool
        self.transposed = transposed

        self.linear = LinearActivation(
            d_input,
            d_output * pool,
            transposed=transposed,
            initializer=initializer,
            weight_norm = weight_norm,
            activation=activation,
            activate=True if activation is not None else False,
        )

    def forward(self, x, skip=None):
        x = self.linear(x)
        if self.transposed:
            x = F.pad(x[..., :-1], (1, 0)) # Shift to ensure causality
            x = rearrange(x, '... (h s) l -> ... h (l s)', s=self.pool)
        else:
            x = F.pad(x[..., :-1, :], (0, 0, 1, 0)) # Shift to ensure causality
            x = rearrange(x, '... l (h s) -> ... (l s) h', s=self.pool)
        if skip is not None:
            x = x + skip
        return x, None

    def step(self, x, state, **kwargs):
        """
        x: (..., H)
        """

        assert len(state) > 0
        y, state = state[0], state[1:]
        if len(state) == 0:
            assert x is not None
            if self.transposed: x = x.unsqueeze(-1)
            x = self.linear(x)
            if self.transposed: x = x.squeeze(-1)
            x = rearrange(x, '... (h s) -> ... h s', s=self.pool)
            state = list(torch.unbind(x, dim=-1))
        else: assert x is None
        return y, state

    def default_state(self, *batch_shape, device=None):
        state = torch.zeros(batch_shape + (self.d_output, self.pool), device=device) # (batch, h, s)
        state = list(torch.unbind(state, dim=-1)) # List of (..., H)
        return state

    @property
    def d_output(self): return self._d_output

registry = {
    'sample': DownSample,
    'pool': DownPool,
}
