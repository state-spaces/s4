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
import numpy as np

optimized = True

if optimized:
    contract = oe.contract
else:
    contract = torch.einsum

from src.models.sequence.ss.kernel import HippoSSKernel
from src.models.nn import LinearActivation, Activation, Normalization

class S4(nn.Module):
    requires_length = True

    def __init__(
            self,
            d_model,
            d_state=64,
            l_max=1, # Maximum length of sequence. Fine if not provided: the kernel will keep doubling in length until longer than sequence. However, this can be marginally slower if the true length is not a power of 2
            channels=1, # maps 1-dim to C-dim
            bidirectional=False,
            # Arguments for FF
            activation='gelu', # activation in between SS and FF
            ln=False, # Extra normalization
            postact=None, # activation after FF
            initializer=None, # initializer on FF
            weight_norm=False, # weight normalization on FF
            hyper_act=None, # Use a "hypernetwork" multiplication
            dropout=0.0,
            transposed=True, # axis ordering (B, L, D) or (B, D, L)
            verbose=False,
            shift=False,
            linear=False,
            # SSM Kernel arguments
            **kernel_args,
        ):
        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum sequence length, also denoted by L
          if this is not known at model creation, set l_max=1
        channels: can be interpreted as a number of "heads"
        bidirectional: bidirectional
        dropout: standard dropout argument
        transposed: choose backbone axis ordering of (B, L, H) or (B, H, L) [B=batch size, L=sequence length, H=hidden dimension]

        Other options are all experimental and should not need to be configured
        """

        super().__init__()
        if verbose:
            import src.utils.train
            log = src.utils.train.get_logger(__name__)
            log.info(f"Constructing S4 (H, N, L) = ({d_model}, {d_state}, {l_max})")

        self.h = d_model
        self.n = d_state
        self.bidirectional = bidirectional
        self.ln = ln
        self.channels = channels
        self.transposed = transposed
        self.shift = shift
        self.linear = linear

        # optional multiplicative modulation GLU-style
        # https://arxiv.org/abs/2002.05202
        self.hyper = hyper_act is not None
        if self.hyper:
            channels *= 2
            self.hyper_activation = Activation(hyper_act)

        self.D = nn.Parameter(torch.randn(channels, self.h))

        if self.bidirectional:
            channels *= 2


        # SSM Kernel
        self.kernel = HippoSSKernel(self.h, N=self.n, L=l_max, channels=channels, verbose=verbose, **kernel_args)

        # Pointwise
        if not self.linear:
            self.activation = Activation(activation)
            dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout
            self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
            if self.ln:
                self.norm = Normalization(self.h*self.channels, transposed=transposed)
            else:
                self.norm = nn.Identity()

        # position-wise output transform to mix features
        if not self.linear:
            self.output_linear = LinearActivation(
                self.h*self.channels,
                self.h,
                transposed=self.transposed,
                initializer=initializer,
                activation=postact,
                activate=True,
                weight_norm=weight_norm,
            )


    def forward(self, u, state=None, **kwargs): # absorbs return_output and transformer src mask
        """
        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as u
        """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SS Kernel
        k, k_state = self.kernel(L=L, state=state) # (C H L) (B C H L)

        # Convolution
        if self.bidirectional:
            k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2)
            k = F.pad(k0, (0, L)) \
                    + F.pad(k1.flip(-1), (L, 0)) \

        if self.shift:
            # Try flip and pad to correct for potential off-by-one
            k_f = torch.fft.rfft(F.pad(k.flip(-1), (L, 0)), n=2*L) # (C H L)
            u_f = torch.fft.rfft(F.pad(u.flip(-1), (L, 0)), n=2*L) # (B H L)
            y_f = contract('bhl,chl->bchl', u_f, k_f) # k_f.unsqueeze(-4) * u_f.unsqueeze(-3) # (B C H L)
            y = torch.fft.irfft(y_f, n=2*L)[..., L:].flip(-1) # (B C H L)
        else:
            k_f = torch.fft.rfft(k, n=2*L) # (C H L)
            u_f = torch.fft.rfft(u, n=2*L) # (B H L)
            y_f = contract('bhl,chl->bchl', u_f, k_f) # k_f.unsqueeze(-4) * u_f.unsqueeze(-3) # (B C H L)
            y = torch.fft.irfft(y_f, n=2*L)[..., :L] # (B C H L)



        # Compute D term in state space equation - essentially a skip connection
        y = y + contract('bhl,ch->bchl', u, self.D) # u.unsqueeze(-3) * self.D.unsqueeze(-1)

        # Compute state update
        if state is not None:
            assert not self.bidirectional, "Bidirectional not supported with state forwarding"
            y = y + k_state
            next_state = self.kernel.forward_state(u, state)
        else:
            next_state = None

        # Optional hyper-network multiplication
        if self.hyper:
            y, yh = rearrange(y, 'b (s c) h l -> s b c h l', s=2)
            y = self.hyper_activation(yh) * y

        # Reshape to flatten channels
        y = rearrange(y, '... c h l -> ... (c h) l')

        if not self.linear:
            y = self.dropout(self.activation(y))

        if not self.transposed: y = y.transpose(-1, -2)

        if not self.linear:
            y = self.norm(y)
            y = self.output_linear(y)

        return y, next_state

    def setup_step(self):
        self.kernel.setup_step()

    def step(self, u, state):
        """ Step one time step as a recurrent model. Intended to be used during validation.

        u: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """
        assert not self.training

        y, next_state = self.kernel.step(u, state) # (B C H)
        y = y + u.unsqueeze(-2) * self.D
        y = rearrange(y, '... c h -> ... (c h)')
        y = self.activation(y)
        if self.transposed:
            y = self.output_linear(y.unsqueeze(-1)).squeeze(-1)
        else:
            y = self.output_linear(y)
        return y, next_state

    def default_state(self, *batch_shape, device=None):
        return self.kernel.default_state(*batch_shape)

    @property
    def d_state(self):
        return self.h * self.n

    @property
    def d_output(self):
        return self.h

    @property
    def state_to_tensor(self):
        return lambda state: rearrange('... h n -> ... (h n)', state)


def test_state(random_init=False, **kwargs):
    # B = 1
    # H = 64
    # N = 64
    # L = 1024
    B = 2
    H = 3
    N = 4
    L = 8
    s4 = S4(H, d_state=N, l_max=L, **kwargs)
    s4.to(device)
    s4.eval()
    for module in s4.modules():
        if hasattr(module, 'setup_step'): module.setup_step()

    u = torch.ones(B, H, L).to(device)
    initial_state = s4.default_state(B)
    if random_init:
        if initial_state.size(-1) == N:
            initial_state = initial_state[..., :N//2]
        initial_state = torch.randn_like(initial_state)
        initial_state = torch.cat([initial_state, initial_state.conj()], dim=-1)

    state = initial_state.clone()
    y, final_state = s4(u, state=state)
    print("output:\n", y, y.shape)
    print("final state:\n", final_state, final_state.shape)

    # Use Stepping
    state = initial_state.clone()
    ys = []
    for u_ in torch.unbind(u, dim=-1):
        y_, state = s4.step(u_, state=state)
        ys.append(y_)
    ys = torch.stack(ys, dim=-1)
    print("step outputs:\n", ys)
    print("step final state:\n", state)

    # Use Chunking

    chunks = 4
    state = initial_state.clone()
    ys = []
    for u_ in u.chunk(chunks, dim=-1):
        y_, state = s4(u_, state=state)
        ys.append(y_)
    ys = torch.cat(ys, dim=-1)
    print("chunk outputs:\n", ys)
    print("chunk final state:\n", state)
    print("chunk output error:")
    utils.compare_outputs(y, ys)
    print("chunk final state error:")
    utils.compare_outputs(final_state, state)


if __name__ == '__main__':
    from benchmark import utils
    torch.manual_seed(42)

    device = 'cuda' # 'cpu'
    device = torch.device(device)

    test_state(random_init=True, mode='nplr', measure='legt', rank=2)
