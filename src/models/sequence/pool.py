""" Implements downsampling and upsampling on sequences """

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

from src.models.sequence import SequenceModule
from src.models.nn import LinearActivation

""" Simple pooling functions that just downsample or repeat

stride: Subsample on the layer dimension
expand: Repeat on the feature dimension
"""

def downsample(x, stride=1, expand=1, average=False, transposed=False):
    if x is None: return None
    if stride > 1:
        # TODO higher dimension stuff
        if transposed:
            # einops appears slower than F
            # if average: x = reduce(x, '... (l s) -> ... l', 'mean', s=stride)
            if average: x = F.avg_pool1d(x, stride, stride)
            else: x = x[..., 0::stride]
        else:
            if average: x = reduce(x, '... (l s) h -> ... l h', 'mean', s=stride)
            else: x = x[..., 0::stride, :]

    if expand > 1:
        if transposed:
            x = repeat(x, '... d l -> ... (d e) l', e=expand)
        else:
            x = repeat(x, '... d -> ... (d e)', e=expand)

    return x

def upsample(x, stride=1, expand=1, transposed=False):
    if x is None: return None
    if expand > 1:
        if transposed:
            x = reduce(x, '... (d e) l -> ... d l', 'mean', e=expand)
        else:
            x = reduce(x, '... (d e) -> ... d', 'mean', e=expand)
    if stride > 1:
        if transposed:
            x = repeat(x, '... l -> ... (l e)', e=stride)
        else:
            x = repeat(x, '... l d -> ... (l e) d', e=stride)
    return x

class DownSample(SequenceModule):
    def __init__(self, d_input, stride=1, expand=1, transposed=True):
        super().__init__()
        self.d_input = d_input
        self.stride = stride
        self.expand = expand
        # self.average = average
        self.transposed = transposed

    def forward(self, x):
        return downsample(x, self.stride, self.expand, False, self.transposed)

    def step(self, x, state, **kwargs):
        if self.stride > 1 or self.expand > 1:
            raise NotImplementedError
        return x, state

    @property
    def d_output(self):
        return self.d_input * self.expand

class DownAvgPool(SequenceModule):
    def __init__(self, d_input, stride=1, expand=1, transposed=True):
        super().__init__()
        self.d_input = d_input
        self.stride = stride
        self.expand = expand
        # self.average = average
        self.transposed = transposed

    def forward(self, x):
        return downsample(x, self.stride, self.expand, True, self.transposed)

    def step(self, x, state, **kwargs):
        if self.stride > 1 or self.expand > 1:
            raise NotImplementedError
        return x, state

    @property
    def d_output(self):
        return self.d_input * self.expand

class UpSample(nn.Module):
    def __init__(self, d_input, stride=1, expand=1, transposed=True):
        super().__init__()
        self.d_input = d_input
        self.stride = stride
        self.expand = expand
        self.transposed = transposed

    def forward(self, x):
        return upsample(x, self.stride, self.expand, self.transposed)

    @property
    def d_output(self):
        return self.d_input // self.expand
    def step(self, x, state, **kwargs):
        if self.stride > 1 or self.expand > 1:
            raise NotImplementedError
        return x, state

""" Pooling functions with trainable parameters """ # For the flexible backbone SequenceModel
class DownLinearPool(SequenceModule):
    def __init__(self, d_input, stride=1, expand=1, transposed=True):
        super().__init__()

        self.d_input = d_input
        self.stride = stride
        self.expand = expand
        self.transposed = transposed

        self.linear = LinearActivation(
            d_input * stride,
            d_input * expand,
            transposed=transposed,
            # initializer=initializer,
            # weight_norm = weight_norm,
            # activation=activation,
            # activate=True if activation is not None else False,
        )

    def forward(self, x):
        if self.transposed:
            x = rearrange(x, '... h (l s) -> ... (h s) l', s=self.stride)
        else:
            x = rearrange(x, '... (l s) h -> ... l (h s)', s=self.stride)
        x = self.linear(x)
        return x

    def step(self, x, state, **kwargs):
        if self.stride > 1 or self.expand > 1:
            raise NotImplementedError
        return x, state

    @property
    def d_output(self):
        return self.d_input * self.expand


""" Pooling functions with trainable parameters """ # TODO make d_output expand instead

class DownPool2d(SequenceModule):

    def __init__(self, d_input, d_output, stride=1, transposed=True, weight_norm=True):
        super().__init__()

        self.linear = LinearActivation(
            d_input,
            d_output,
            transposed=transposed,
            weight_norm=weight_norm,
        )

        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride),

    def forward(self, x):
        if self.transposed:
            x = self.pool(x)

class DownPool(SequenceModule):
    def __init__(self, d_input, d_output=None, expand=None, stride=1, transposed=True, weight_norm=True, initializer=None, activation=None):
        super().__init__()
        assert (d_output is None) + (expand is None) == 1
        if d_output is None: d_output = d_input * expand

        self._d_output = d_output
        self.stride = stride
        self.transposed = transposed

        self.linear = LinearActivation(
            d_input * stride,
            d_output,
            transposed=transposed,
            initializer=initializer,
            weight_norm = weight_norm,
            activation=activation,
            activate=True if activation is not None else False,
        )

    def forward(self, x):
        if self.transposed:
            x = rearrange(x, '... h (l s) -> ... (h s) l', s=self.stride)
        else:
            x = rearrange(x, '... (l s) h -> ... l (h s)', s=self.stride)
        x = self.linear(x)
        return x, None

    def step(self, x, state, **kwargs): # TODO needs fix in transpose ca, **kwargsse
        """
        x: (..., H)
        """

        if x is None: return None, state
        state.append(x)
        if len(state) == self.stride:
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
    def __init__(self, d_input, d_output, stride, transposed=True, weight_norm=True, initializer=None, activation=None):
        super().__init__()

        self.d_input = d_input
        self._d_output = d_output
        self.stride = stride
        self.transposed = transposed

        self.linear = LinearActivation(
            d_input,
            d_output * stride,
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
            x = rearrange(x, '... (h s) l -> ... h (l s)', s=self.stride)
        else:
            x = F.pad(x[..., :-1, :], (0, 0, 1, 0)) # Shift to ensure causality
            x = rearrange(x, '... l (h s) -> ... (l s) h', s=self.stride)
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
            x = rearrange(x, '... (h s) -> ... h s', s=self.stride)
            state = list(torch.unbind(x, dim=-1))
        else: assert x is None
        return y, state

    def default_state(self, *batch_shape, device=None):
        state = torch.zeros(batch_shape + (self.d_output, self.stride), device=device) # (batch, h, s)
        state = list(torch.unbind(state, dim=-1)) # List of (..., H)
        return state

    @property
    def d_output(self): return self._d_output

registry = {
    'sample': DownSample,
    'pool': DownAvgPool,
    'linear': DownLinearPool,
    # 'pool': DownPool,
}

if __name__ == '__main__':
    from benchmark import utils

    a = torch.ones(50, 256, 1024)
    a, = utils.convert_data(a)
    stride = 4

    y0 = downsample(a, stride=stride, average=True, transposed=True)
    y1 = F.avg_pool1d(a, stride, stride)

    print(y0.shape, y1.shape)
    print(y0 - y1)

    utils.benchmark(downsample, a, stride, 1, True, True, repeat=100, desc='einops')
    utils.benchmark(F.avg_pool1d, a, stride, stride, repeat=100, desc='torch')
