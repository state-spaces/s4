if __name__ == '__main__':
    import sys
    import pathlib
    p = pathlib.Path().absolute()
    print("Adding path: ", p)
    sys.path.append(str(p))

import math
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

from src.models.sequence.ss.kernel import HippoSSKernel
from src.models.nn import LinearActivation, Activation

class StateSpace(nn.Module):
    requires_length = True

    def __init__(
            self,
            H,
            l_max=None,
            # Arguments for SSM Kernel
            d_state=64,
            measure='legs',
            dt_min=0.001,
            dt_max=0.1,
            rank=1,
            trainable=None,
            lr=None,
            length_correction=False,
            stride=1,
            weight_decay=0.0, # weight decay on the SS Kernel
            precision=1,
            cache=False, # Cache the SS Kernel during evaluation
            # Arguments for FF
            activation='gelu', # activation in between SS and FF
            postact=None, # activation after FF
            weight_norm=False, # weight normalization on FF
            initializer=None, # initializer on FF
            input_linear=False,
            hyper_act=None,
            dropout=0.0,
            transposed=True, # axis ordering (B, L, D) or (B, D, L)
            resample=False,
            use_state=False,
            verbose=False,
            mode='nplr',
            keops=False,
        ):
        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum sequence length, also denoted by L
          if this is not known at model creation, or inconvenient to pass in,
          set l_max=None and length_correction=True
        dropout: standard dropout argument
        transposed: choose backbone axis ordering of (B, L, D) or (B, D, L) [B=batch size, L=sequence length, D=feature dimension]

        Other options are all experimental and should not need to be configured
        """

        super().__init__()
        if verbose:
            import src.utils.train
            log = src.utils.train.get_logger(__name__)
            log.info(f"Constructing s4 (H, N, L) = ({H}, {d_state}, {l_max})")

        self.h = H
        self.n = d_state if d_state > 0 else H
        self.stride = stride
        if l_max is not None and stride > 1:
            assert l_max % stride == 0
            l_max = l_max // self.stride
        self.cache = cache
        self.weight_decay = weight_decay
        self.transposed = transposed
        self.resample = resample

        self.D = nn.Parameter(torch.randn(self.h))

        # Optional (position-wise) input transform
        if input_linear:
            self.input_linear = LinearActivation(
                self.h,
                self.h,
                transposed=self.transposed,
                initializer=initializer,
                activation=postact,
                activate=True,
                weight_norm=weight_norm,
            )
        else:
            self.input_linear = nn.Identity()

        # SSM Kernel
        self.kernel = HippoSSKernel(self.n, self.h, l_max, dt_min=dt_min, dt_max=dt_max, measure=measure, rank=rank, trainable=trainable, lr=lr, length_correction=length_correction, precision=precision, cache=cache, mode=mode, resample=resample, keops=keops)
        self.K = None # Cache the computed convolution filter if possible (during evaluation)

        # optional multiplicative modulation
        self.hyper = hyper_act is not None
        if self.hyper:
            self.hyper_linear = LinearActivation(
                self.h,
                self.h,
                transposed=True,
                initializer=initializer,
                activation=hyper_act,
                activate=True,
                weight_norm=weight_norm,
            )


        self.activation = Activation(activation)
        dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = LinearActivation(
            self.h,
            self.h,
            transposed=self.transposed,
            initializer=initializer,
            activation=postact,
            activate=True,
            weight_norm=weight_norm,
        )

        if use_state:
            self._initial_state = nn.Parameter(torch.zeros(self.h, self.n))


    def forward(self, u, state=None, cache=None, **kwargs): # absorbs return_output and transformer src mask
        """
        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as u
        """
        u = self.input_linear(u)
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SS Kernel
        if state is not None:
            assert self.stride == 1, "Striding not supported with states"
            k, k_state = self.kernel(state=state, L=L)
        else:
            k = self.kernel(L=L)

        # Stride the filter if needed
        if self.stride > 1:
            k = k[..., :L // self.stride] # (H, L/S)
            k = F.pad(k.unsqueeze(-1), (0, self.stride-1)) # (H, L/S, S)
            k = rearrange(k, '... h s -> ... (h s)') # (H, L)
        else:
            k = k[..., :L]

        # Convolution
        k_f = torch.fft.rfft(k, n=2*L) # (H L)
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y_f = k_f * u_f
        y = torch.fft.irfft(y_f, n=2*L)[..., :L] # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        # Compute state update
        if state is not None:
            y = y + k_state[..., :L]
            next_state = self.kernel.next_state(state, u)
        else:
            next_state = None

        # Optional hyper-network multiplication
        if self.hyper:
            hyper = self.hyper_linear(u)
            y = hyper * y

        y = self.dropout(self.activation(y))

        if not self.transposed: y = y.transpose(-1, -2)

        y = self.output_linear(y)

        return y, next_state

    def step(self, u, state):
        """ Step one time step as a recurrent model. Intended to be used during validation.

        u: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """
        assert not self.training
        y, next_state = self.kernel.step(u, state)
        y = y + u * self.D
        y = self.output_linear(self.activation(y).unsqueeze(-1)).squeeze(-1)
        return y, next_state

    def default_state(self, *batch_shape, device=None):
        return self._initial_state.repeat(*batch_shape, 1, 1)

    @property
    def d_state(self):
        return self.h * self.n

    @property
    def d_output(self):
        return self.h

    @property
    def state_to_tensor(self):
        return lambda state: rearrange('... h n -> ... (h n)', state)


def test_state():
    B = 1
    H = 64
    N = 64
    L = 1024
    s4 = StateSpace(H, d_state=N, l_max=L, use_state=True, mode='nplr')
    s4.to(device)
    for module in s4.modules():
        if hasattr(module, '_setup'): module._setup()

    u = torch.ones(B, H, L).to(device)
    # initial_state = torch.zeros(B, H, N)
    initial_state = torch.randn(B, H, N//2, dtype=torch.cfloat, device=device)

    state = initial_state.clone()
    y, final_state = s4(u, state)
    print("output:\n", y, y.shape)
    print("final state:\n", final_state, final_state.shape)

    state = initial_state.clone()
    chunks = 2
    outs = []
    for u_ in u.chunk(chunks, dim=-1):
        y_, state = s4(u_, state=state)
        outs.append(y_)
        # print("step output:", y_, y_.shape)
        # print("step state:", state, state.shape)
    outs = torch.cat(outs, dim=-1)
    print("step outputs:\n", outs)
    print("step final state:\n", state)
    print("step output error:")
    utils.compare_outputs(y, outs)
    print("step final state error:")
    utils.compare_outputs(final_state, state)

def test_recurrence():
    B = 2
    H = 3
    N = 4
    L = 6
    s4 = StateSpace(H, d_state=N, l_max=L, use_state=True, mode='slow')
    s4.to(device)
    s4.eval()


    u = torch.ones(B, H, L).to(device)
    # initial_state = torch.zeros(B, H, N//2, dtype=torch.cfloat, device=device)
    initial_state = torch.randn(B, H, N//2, dtype=torch.cfloat, device=device)
    # initial_state = torch.zeros(B, H, N, device=device) # real case
    # initial_state = torch.randn(B, H, N, device=device)
    state = initial_state.clone()
    y, state = s4(u, state=state)
    print(y, y.shape)
    print("state", state, state.shape)

    # set up dC
    for module in s4.modules():
        if hasattr(module, '_setup'): module._setup()

    # state = s4.default_state(*u.shape[:-2], device=device)
    state = initial_state.clone()
    ys = []
    for u_ in torch.unbind(u, dim=-1):
        y_, state = s4.step(u_, state=state)
        ys.append(y_)
    y = torch.stack(ys, dim=-1)
    print(y, y.shape)
    print("state", state, state.shape)

if __name__ == '__main__':
    from benchmark import utils
    torch.manual_seed(42)

    device = 'cuda' # 'cpu'
    device = torch.device(device)

    test_state()
    test_recurrence()
