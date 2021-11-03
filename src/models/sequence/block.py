""" Implements a full residual block around a black box layer

Configurable options include:
normalization position: prenorm or postnorm
normalization type: batchnorm, layernorm etc.
subsampling/pooling
residual options: feedforward, residual, affine scalars, depth-dependent scaling, etc.
"""

from torch import nn

import src.utils as utils
from src.models.nn.components import Normalization
from src.models.sequence import SequenceModule
from src.models.sequence.pool import registry as pool_registry
from src.models.nn.residual import registry as residual_registry
import src.utils.registry as registry


class SequenceResidualBlock(SequenceModule):
    def __init__(
            self,
            d_input,
            i_layer=None, # Only needs to be passed into certain residuals like Decay
            prenorm=True,
            dropout=0.0,
            layer=None, # Config for black box module
            residual=None, # Config for residual function
            norm=None, # Config for normalization layer
            pool=None,
        ):
        super().__init__()

        self.i_layer = i_layer
        self.d_input = d_input
        self.layer = utils.instantiate(registry.layer, layer, d_input)
        self.prenorm = prenorm

        # Residual
        # d_residual is the output dimension after residual
        if residual is None:
            self.residual = None
            self.d_residual = self.layer.d_output
        else:
            self.residual = utils.instantiate(residual_registry, residual, i_layer, d_input, self.layer.d_output)
            self.d_residual = self.residual.d_output

        # Normalization
        d_norm = d_input if self.prenorm else self.d_residual
        # We don't use config to directly instantiate since Normalization has some special cases
        if norm is None:
            self.norm = None
        elif isinstance(norm, str):
            self.norm = Normalization(d_norm, transposed=self.transposed, _name_=norm)
        else:
            self.norm = Normalization(d_norm, transposed=self.transposed, **norm)

        # Pool
        self.pool = utils.instantiate(pool_registry, pool, self.d_residual, transposed=self.transposed)

        # Dropout
        drop_cls = nn.Dropout2d if self.transposed else nn.Dropout
        self.drop = drop_cls(dropout) if dropout > 0.0 else nn.Identity()


    @property
    def transposed(self):
        return getattr(self.layer, 'transposed', False)

    @property
    def d_output(self):
        return self.pool.d_output if self.pool is not None else self.d_residual

    @property
    def d_state(self):
        return self.layer.d_state

    @property
    def state_to_tensor(self):
        return self.layer.state_to_tensor

    def default_state(self, *args, **kwargs):
        return self.layer.default_state(*args, **kwargs)

    def forward(self, x, *args, state=None, **kwargs):
        y = x

        # Pre-norm
        if self.norm is not None and self.prenorm: y = self.norm(y)

        # Black box module
        y, state = self.layer(y, *args, state=state, **kwargs)

        # Residual
        if self.residual is not None: x = self.residual(x, self.drop(y), self.transposed)

        # Post-norm
        if self.norm is not None and not self.prenorm: x = self.norm(x)

        # Pool
        # x = pool.downpool(x, self.pool, self.expand, self.transposed)
        if self.pool is not None: x = self.pool(x)

        return x, state

    def step(self, x, state, *args, **kwargs): # TODO needs fix for transpose logic
        y = x

        # Pre-norm
        if self.norm is not None and self.prenorm:
            if self.transposed: y = y.unsqueeze(-1)
            y = self.norm(y) # TODO transpose seems wrong
            if self.transposed: y = y.squeeze(-1)

        # Black box module
        y, state = self.layer.step(y, state, *args, **kwargs)

        # Residual
        if self.residual is not None: x = self.residual(x, y, transposed=False) # TODO this would not work with concat

        # Post-norm
        if self.norm is not None and not self.prenorm:
            if self.transposed: y = y.unsqueeze(-1)
            x = self.norm(x)#.step(x)
            if self.transposed: y = y.squeeze(-1)

        # Pool
        if self.pool is not None: x = self.pool(x)

        return x, state
