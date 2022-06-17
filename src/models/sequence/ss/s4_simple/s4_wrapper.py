import torch
import torch.nn as nn
from src.models.nn import Activation, LinearActivation
from src.models.sequence.ss.s4_simple.utils import DropoutNd
from src.models.sequence.ss.s4_simple.s4_simple import SimpleS4

# Below here are standard wrapper classes to handle
# (1) Non-linearity
# (2) Integration with the State Spaces Code base
class NonLinear(nn.Module):
    def __init__(self, h, channels, 
                ln=False, # Extra normalization
                transposed=True,
                dropout=0.0, 
                postact=None, # activation after FF
                activation='gelu', # activation in between SS and FF
                initializer=None, # initializer on FF
                weight_norm=False, # weight normalization on FF
                ):
            super().__init__()
            dropout_fn = DropoutNd # nn.Dropout2d bugged in PyTorch 1.11
            dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
            #norm = Normalization(h*channels, transposed=transposed) if ln else nn.Identity()

            activation_fn = Activation(activation)

            output_linear = LinearActivation(
                h*channels,
                h,
                transposed=transposed, 
                initializer=initializer,
                activation=postact,
                activate=True,
                weight_norm=weight_norm,
            )
            #self.f = nn.Sequential(activation_fn, dropout, norm, output_linear)
            self.f = nn.Sequential(activation_fn, dropout, output_linear)
    def forward(self,x):  # Always (B H L)
        return self.f(x)

class SimpleS4Wrapper(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=64,
            channels=1,
            bidirectional=False,
            dropout=0.0,
            transposed=True, # axis ordering (B, L, D) or (B, D, L)
            ln=True, # IGNORED: Extra normalization
            postact=None, # activation after FF
            activation='gelu', # activation in between SS and FF
            initializer=None, # initializer on FF
            weight_norm=False, # weight normalization on FF
            linear=False,
            # SSM Kernel arguments
            **kernel_args,
        ):
        super().__init__()
        self.h = d_model
        self.d = d_state
        self.channels = channels
        #self.shift = shift
        #self.linear = linear
        self.out_d = self.h
        self.transposed = transposed
        self.bidirectional = bidirectional
        assert not bidirectional, f"Bidirectional NYI"
        self.s4 = SimpleS4(nHippos=d_model, d_state=d_state, 
                            channels=channels, **kernel_args)
        # the mapping
        # We transpose if it's not in the forward.
        nl         = NonLinear(self.h, channels=self.channels, ln=ln, # Extra normalization
                        dropout=dropout, postact=postact, activation=activation, transposed=True,
                        initializer=initializer, weight_norm=weight_norm)
        self.out = nn.Identity() if linear else nl

    def forward(self, u, state=None):
        #  u: (B H L) if self.transposed else (B L H)
        if not self.transposed: u = u.transpose(-1, -2)
        # We only pass BHL, and it is as if transposed is True.
        ret = self.out(self.s4(u))
        if not self.transposed: ret = ret.transpose(-1, -2)
        return ret, state

    @property
    def d_state(self): return self.h * self.d 

    @property
    def d_output(self): return self.out_d  