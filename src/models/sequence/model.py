""" Core deep sequence model backbone, in the style of ResNets / Transformers.

The SequenceModel class implements a generic (batch, length, d_input) -> (batch, length, d_output) transformation
"""

import functools

import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig

from src.utils.config import to_list, to_dict
# from src.models.sequence.rnns import rnn # [21-09-13] I get a baffling error where hydra claims circular import if I _remove_ this line. This import doesn't even appear to be used at all in this file
from src.models.sequence.block import SequenceResidualBlock
from src.models.sequence.base import SequenceModule
from src.models.nn.components import Normalization
from src.models.nn.initialization import weights_init
from src.tasks import encoders, decoders


class SequenceModel(SequenceModule):
    def __init__(
        self,
        d_model, # Resize input (useful for deep models with residuals)
        n_layers=1, # Number of layers
        transposed=False,
        dropout=0.0, # Residual dropout parameter
        prenorm=True,
        layer=None, # layer config, must be specified
        residual=None, # Residual config
        norm=None, # Normalization config (e.g. layer vs batch)
        pool=None,
        init=None,
        verbose=False,
        track_norms=True,
        dropinp=0.0,
    ):
        super().__init__()
        # Save arguments needed for forward pass
        self.d_model = d_model
        self.transposed = transposed
        self.verbose = verbose
        self.track_norms = track_norms
        self._forward = False

        if dropinp > 0.0:
            self.drop = nn.Dropout2d(dropinp) if self.transposed else nn.Dropout(dropinp)
        else:
            self.drop = nn.Identity()

        layer = to_list(layer, recursive=False)

        # Some special arguments are passed into each layer
        for _layer in layer:
            # If layers don't specify dropout, add it
            if _layer.get('dropout', None) is None:
                _layer['dropout'] = dropout
            # Ensure all layers are shaped the same way
            _layer['transposed'] = transposed

        # Duplicate layers
        layers = layer * n_layers

        # Instantiate layers
        _layers = []
        d = d_model
        for l, layer in enumerate(layers):
            block = SequenceResidualBlock(d, l+1, prenorm=prenorm, dropout=dropout, layer=layer, residual=residual, norm=norm, pool=pool)
            _layers.append(block)
            d = block.d_output

        self.d_output = d
        self.layers = nn.ModuleList(_layers)

        if prenorm:
            if norm is None:
                self.norm = None
            elif isinstance(norm, str):
                self.norm = Normalization(self.d_output, transposed=self.transposed, _name_=norm)
            else:
                self.norm = Normalization(self.d_output, transposed=self.transposed, **norm)
        else:
            self.norm = nn.Identity()

        # Initializer hook
        if init is not None:
            self.apply(functools.partial(weights_init, init_cfg=init))

    def forward(self, inputs, *args, state=None, **kwargs):
        """ Inputs assumed to be (batch, sequence, dim) """
        # Debug
        if self.verbose and not self._forward:
            print("Model: unused kwargs", kwargs)
            self._forward = True

        if self.transposed: inputs = rearrange(inputs, 'b l d -> b d l')
        inputs = self.drop(inputs)

        # Track norms
        if self.track_norms: output_norms = [torch.mean(inputs.detach() ** 2)]

        # Apply layers
        outputs = inputs
        prev_states = [None] * len(self.layers) if state is None else state
        next_states = []
        for layer, prev_state in zip(self.layers, prev_states):
            outputs, state = layer(outputs, *args, state=prev_state, **kwargs) # TODO handle state
            next_states.append(state)
            if self.track_norms: output_norms.append(torch.mean(outputs.detach() ** 2))
        outputs = self.norm(outputs)

        if self.transposed: outputs = rearrange(outputs, 'b d l -> b l d')

        if self.track_norms:
            metrics = to_dict(output_norms, recursive=False)
            self.metrics = {f'norm/{i}': v for i, v in metrics.items()}

        return outputs, next_states

    @property
    def d_state(self):
        d_states = [layer.d_state for layer in self.layers]
        return sum([d for d in d_states if d is not None])

    @property
    def state_to_tensor(self):
        # Slightly hacky way to implement this in a curried manner (so that the function can be extracted from an instance)
        # Somewhat more sound may be to turn this into a @staticmethod and grab subclasses using hydra.utils.get_class
        def fn(state):
            x = [_layer.state_to_tensor(_state) for (_layer, _state) in zip(self.layers, state)]
            x = [_x for _x in x if _x is not None]
            return torch.cat( x, dim=-1)
        return fn

    def default_state(self, *batch_shape, device=None):
        return [layer.default_state(*batch_shape, device=device) for layer in self.layers]

    def step(self, x, state):
        """ 
        Step one time step as a recurrent model. Intended to be used during validation.

        u: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """

        # if self.transposed: x = rearrange(x, 'b l d -> b d l')

        # Apply layers
        prev_states = [None] * len(self.layers) if state is None else state
        next_states = []
        for layer, prev_state in zip(self.layers, prev_states):
            x, state = layer.step(x, state=prev_state)
            next_states.append(state)
            
        x = self.norm(x)

        # if self.transposed: x = rearrange(x, 'b d l -> b l d')

        return x, next_states
