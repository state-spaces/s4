""" Different deep backbone that is essentially a 1-D UNet instead of ResNet/Transformer backbone.

Sequence length gets downsampled through the depth of the network while number of feature increases.
Then sequence length gets upsampled again (causally) and blocks are connected through skip connections.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from einops import rearrange, repeat, reduce
from opt_einsum import contract

from src.models.sequence.base import SequenceModule
from src.models.sequence.pool import DownPool, UpPool
from src.models.sequence.block import SequenceResidualBlock



class SequenceUNet(SequenceModule):
    """
    layer is a Namespace that specifies '_name_', referring to a constructor, and a list of arguments to that layer constructor. This layer must subscribe to the interface (i) takes a hidden dimension H and sequence length L (ii) forward pass transforms input sequence of shape (B, H, L) to output (B, H, L)
    """

    def __init__(
        self,
        d_model, n_layers, pool=[], expand=1, ff=2, cff=0,
        prenorm=False,
        dropout=0.0,
        dropres=0.0,
        layer=None,
        residual=None,
        norm=None,
        initializer=None,
        l_max=-1,
        transposed=True,
    ):
        super().__init__()
        self.d_model = d_model
        H = d_model
        L = l_max
        self.L = L
        self.transposed = transposed
        assert l_max > 0, "UNet must have length passed in"

        # Layer arguments
        layer_cfg = layer.copy()
        layer_cfg['dropout'] = dropout
        layer_cfg['transposed'] = self.transposed
        layer_cfg['initializer'] = initializer
        layer_cfg['l_max'] = L
        print("layer config", layer_cfg)

        ff_cfg = {
            '_name_': 'ff',
            'expand': ff,
            'transposed': self.transposed,
            'activation': 'gelu',
            'initializer': initializer, # TODO
            'dropout': dropout, # TODO untie dropout
        }

        def _residual(d, i, layer):
            return SequenceResidualBlock(
                d,
                i, # temporary placeholder for i_layer
                prenorm=prenorm,
                dropout=dropres,
                layer=layer,
                residual=residual if residual is not None else 'R',
                norm=norm,
                pool=None,
            )

        # Down blocks
        d_layers = []
        for p in pool:
            for i in range(n_layers):
                d_layers.append(_residual(H, i+1, layer_cfg))
                if ff > 0: d_layers.append(_residual(H, i+1, ff_cfg))

            # Add sequence downsampling and feature expanding
            d_layers.append(DownPool(H, H*expand, pool=p, transposed=self.transposed)) # TODO take expansion argument instead
            L //= p
            layer_cfg['l_max'] = L
            H *= expand
        self.d_layers = nn.ModuleList(d_layers)

        # Center block
        c_layers = [ ]
        for i in range(n_layers):
            c_layers.append(_residual(H, i+1, layer_cfg))
            if cff > 0: c_layers.append(_residual(H, i+1, ff_cfg))
        self.c_layers = nn.ModuleList(c_layers)

        # Up blocks
        u_layers = []
        for p in pool[::-1]:
            H //= expand
            L *= p
            layer_cfg['l_max'] = L
            u_layers.append(UpPool(H*expand, H, pool=p, transposed=self.transposed)) # TODO

            for i in range(n_layers):
                u_layers.append(_residual(H, i+1, layer_cfg))
                if ff > 0: u_layers.append(_residual(H, i+1, ff_cfg))
        self.u_layers = nn.ModuleList(u_layers)

        assert H == d_model

        self.norm = nn.LayerNorm(H)

    # @property
    # def transposed(self):
    #     return len(self.d_layers) > 0 and self.d_layers[0].transposed
    @property
    def d_output(self):
        return self.d_model

    def forward(self, x, state=None):
        """
        input: (batch, length, d_input)
        output: (batch, length, d_output)
        """
        if self.transposed: x = x.transpose(1, 2)

        # Down blocks
        outputs = [] # Store all layers for SequenceUNet structure
        for layer in self.d_layers:
            outputs.append(x)
            x, _ = layer(x)

        # Center block
        outputs.append(x)
        for layer in self.c_layers:
            x, _ = layer(x)
        x = x + outputs.pop()

        for layer in self.u_layers:
            x, _ = layer(x)
            x = x + outputs.pop()

        # feature projection
        if self.transposed: x = x.transpose(1, 2) # (batch, length, expand)
        x = self.norm(x)

        return x, None # required to return a state

    def default_state(self, *args, **kwargs):
        """ x: (batch) """
        layers = list(self.d_layers) + list(self.c_layers) + list(self.u_layers)
        return [layer.default_state(*args, **kwargs) for layer in layers]

    def step(self, x, state, **kwargs):
        """
        input: (batch, d_input)
        output: (batch, d_output)
        """
        # States will be popped in reverse order for convenience
        state = state[::-1]

        # Down blocks
        outputs = [] # Store all layers for SequenceUNet structure
        next_state = []
        for layer in self.d_layers:
            outputs.append(x)
            x, _next_state = layer.step(x, state=state.pop(), **kwargs)
            next_state.append(_next_state)
            if x is None: break

        # Center block
        if x is None:
            # Skip computations since we've downsized
            skipped = len(self.d_layers) - len(outputs)
            for _ in range(skipped+len(self.c_layers)+skipped):
                next_state.append(state.pop())
            u_layers = list(self.u_layers)[skipped:]
        else:
            outputs.append(x)
            for layer in self.c_layers:
                x, _next_state = layer.step(x, state=state.pop(), **kwargs)
                next_state.append(_next_state)
            x = x + outputs.pop()
            u_layers = self.u_layers

        for layer in u_layers:
            x, _next_state = layer.step(x, state=state.pop(), **kwargs)
            next_state.append(_next_state)
            x = x + outputs.pop()

        # feature projection
        x = self.norm(x)
        return x, next_state

    def cache_all(self):
        modules = self.modules()
        next(modules)
        for layer in modules:
            if hasattr(layer, 'cache_all'): layer.cache_all()

def prepare_generation(model):
    model.eval()
    if hasattr(model, 'cache_all'): model.cache_all()

@torch.inference_mode()
def generate_recurrent(model, batch_size=None, x=None):
    from src.tasks.mixture import mixture_sample
# TODO incorporate normalization function for dataset
# TODO handle or document non-mixture case
    """ generate remaining L-L' samples given x: (B, L', C) a context for the model """

    if x is None:
        assert batch_size is not None
        x = torch.zeros(batch_size, model.d_model, device=device)
        state = model.default_state(batch_size, device=device)
    else: raise NotImplementedError("Conditional generation not implemented yet")

    xs = []
    for i in range(model.L):
        print("pixel", i)
        x, state = model.step(x, state)
        x = mixture_sample(x)
        # TODO postprocess: clamp, divide into buckets, renormalize
        x = x.unsqueeze(-1)
        xs.append(x)
    sample = torch.stack(xs, dim=1)
    print("recurrent sample shape", sample.shape)

@torch.no_grad()
def generate_global(model, batch_size=None, x=None, length=None):
    from tasks.mixture import mixture_sample
    """ generate remaining L-L' samples given x: (B, L', C) a context for the model """

    if x is None:
        assert batch_size is not None
        x = torch.zeros(batch_size, model.L, model.d_input, device=device)
    else: raise NotImplementedError("Conditional generation not implemented yet")

    if length is None: length = model.L
    for i in range(length):
        print("pixel", i)
        y = model(x)
        y = torch.cat([y, y.new_zeros(batch_size, 1, model.d_output)], dim=1) # TODO handle sequence shape properly
        z = mixture_sample(y[:, i, :])
        # TODO postprocess: clamp, divide into buckets, renormalize
        z = z.unsqueeze(-1)
        x[:, i, :] = z
    print("global sample shape", x.shape)

@torch.inference_mode()
def test():
    import time
    B = 256 # 384
    L = 3072
    # x = torch.ones(B, L, C).to(device)

    H = 192
    model = SequenceUNet(
        d_model=H,
        n_layers=16,
        l_max=L,
        pool=[3, 4, 4],
        expand=2,
        ff=2,
        # dropout=0.1,
        layer={'_name_': 's4'},
        transposed=True,
    )
    # print(model)

    # Setup the model
    model.to(device)
    for module in model.modules():
        if hasattr(module, 'setup'): module.setup()

    # Parallelized forward pass
    # print("input shape", x.shape)
    # x = torch.ones(B, L, H).to(device)
    # y, _ = model(x)
    # print("output shape", y.shape)


    # Test stepping
    model.eval()
    for module in model.modules():
        if hasattr(module, 'setup_step'): module.setup_step()
    state = model.default_state(B, device=device)
    t = time.time()
    # for i, _x in enumerate(torch.unbind(x, dim=-2)):
    _x = torch.zeros(B, H).to(device)
    for i in range(L):
        print(i)
        _y, state = model.step(_x, state)
        # print("step output", _y, _y.shape)
        # print("step diff", _y - y[:,i,:])
    print("time", time.time() - t)

    # Test generation TODO [21-09-25] needs to be hooked up to encoder/decoder
    # generate_recurrent(model, batch_size=4)
    # generate_global(model, batch_size=4)

def benchmark():
    import time
    L = 3072
    C = 1

    H = 64
    k = 12
    layer = default_cfgs['s4']
    layer.d_model = 64
    net = SequenceUNet(
        d_input=C,
        d_output=3*k,
        d_model=H,
        n_layers=2,
        l_max=L,
        pool=[3, 4, 4],
        expand=2,
        ff=2,
        dropout=0.0,
        layer=layer,
    )
    net.to(device)
    for module in net.modules():
        if hasattr(module, 'setup'): module.setup()
    prepare_generation(net)

    t = time.time()
    # generate_recurrent(net, 8000)
    generate_global(net, 800, length=64)
    print("time", time.time() - t)
    # Utils uses extra memory
    # from benchmark import utils
    # utils.benchmark_forward(1, generate_recurrent, net, 8192, desc='Recurrent generation')
    # utils.benchmark_forward(1, generate_global, net, 512, desc='Global generation')

if __name__ == '__main__':
    device = torch.device('cuda')
    test()
    # benchmark()
