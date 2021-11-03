""" Utility wrappers around modules to let them handle Tuples and extra arguments """

# import torch
from torch import nn


def TupleModule(module):
    """ Wrap a nn.Module class with two features:
        - discard extra arguments in the forward pass
        - return a tuple
    """
    # TODO maybe possible with functools.wraps
    class WrappedModule(module):
        def forward(self, x, *args, **kwargs):
            y = super().forward(x)
            return y if isinstance(y, tuple) else (y,)
    # https://stackoverflow.com/questions/5352781/how-to-set-class-names-dynamically
    WrappedModule.__name__ = module.__name__
    return WrappedModule

def Squeeze(module, dim=-1):
    """ Wrap a nn.Module to squeeze a dimension.
    Use for e.g. Embeddings, because our sequence API assumes a feature dimension while nn.Embedding does not
    """
    class WrappedModule(module):
        def forward(self, x, *args, **kwargs):
            assert x.size(dim) == 1
            x = x.squeeze(dim)
            y = super().forward(x)
            return y
    # https://stackoverflow.com/questions/5352781/how-to-set-class-names-dynamically
    WrappedModule.__name__ = module.__name__
    return WrappedModule

# TODO maybe call these TupleIdentity etc. instead?
Identity = TupleModule(nn.Identity)
Embedding = TupleModule(nn.Embedding)
# Embedding = TupleModule(Squeeze(nn.Embedding))
Linear = TupleModule(nn.Linear)

def TupleSequential(*modules):
    """ Similar to TupleModule:
    - Discard extra arguments in forward pass
    - Return a Tuple

    Semantics are the same as nn.Sequential, with extra convenience features:
    - Discard None modules
    - Flatten inner Sequential modules
    - Discard extra Identity modules
    - If only one Module, extract it to top level
    """
    def flatten(module):
        if isinstance(module, nn.Sequential):
            return sum([flatten(m) for m in module], [])
        else:
            return [module]

    modules = flatten(nn.Sequential(*modules))
    modules = [module for module in modules if module if not None and not isinstance(module, nn.Identity)]

    class Sequential(nn.Sequential):
        def forward(self, x, *args, **kwargs):
            # layer_args = []
            x = x,
            for layer in self:
                x = layer(*(x + args), **kwargs) # Always a tuple
                # args = tuple(layer_args) + args
            return x # Returns a tuple

    if len(modules) == 0:
        return Identity()
    elif len(modules) == 1:
        return modules[0]
    else:
        return Sequential(*modules)

def Transpose(module_cls):
    class TransposedModule(module_cls):
        def __init__(self, *args, transposed=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.transposed = transposed

        def forward(self, x, *args, **kwargs):
            if self.transposed: x = x.transpose(-1, -2)
            y, *z = super().forward(x, *args, **kwargs)
            if self.transposed: y = y.transpose(-1, -2)
            return y, *z
    TransposedModule.__name__ = module_cls.__name__
    return TransposedModule
