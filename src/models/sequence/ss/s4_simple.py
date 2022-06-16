import torch
import torch.nn as nn
from einops import rearrange, repeat
import opt_einsum as oe
from src.models.nn import Activation, LinearActivation

import math

# Replacement for Dropout in PyTorch 1.11.0
class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True):
        """ tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        For some reason tie=False is dog slow, prob something wrong with torch.distribution
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, X):
        """ X: (batch, dim, lengths...) """
        if self.training:
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
            mask_shape = X.shape[:2] + (1,)*(X.ndim-2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1.-self.p
            return X * mask * (1.0/(1-self.p))
        return X

class OurModule(nn.Module):
    def __init__(self): super().__init__()

    def register(self, name, tensor, trainable=False, lr=None, wd=None):
        """Utility method: register a tensor as a buffer or trainable parameter"""

        if trainable:
            self.register_parameter(name, nn.Parameter(tensor))
        else:
            self.register_buffer(name, tensor)

        optim = {}
        if trainable and lr is not None: optim["lr"] = lr
        if trainable and wd is not None: optim["weight_decay"] = wd
        if len(optim) > 0: setattr(getattr(self, name), "_optim", optim)

#
# This is intended to match np.convolve(x,w)[:len(w)]
# That is, (u \ast v)[k] = sum_{j} u[k-j]v[j]
# Here y = (u \ask v) on return.
# We assume the inputs are:
# u (B H L)
# v (C H L)
# and we want to produce y that is (B C H L)
#
def fft_conv(u,v):
    L   = u.shape[-1]
    u_f = torch.fft.rfft(u, n=2*L) # (B H L)
    v_f = torch.fft.rfft(v, n=2*L) # (C H L)
   
    y_f = oe.contract('bhl,chl->bchl', u_f, v_f) 
    y   = torch.fft.irfft(y_f, n=2*L)[..., :L] # (B C H L)
    return y

class SimpleS4(OurModule):
    def __init__(self,
            nHippos,
            d_state=64,
            channels=1, 
            use_initial=True, # Use the initial state?
            zero_order_hold=True, # Use zero-order hold approximation
            trap_rule=False,
            dt_min=0.001,
            dt_max=0.1,
            lr=None, # Hook to set LR of SSM parameters differently
            learn_a=True,
            learn_theta=True,
            theta_scale=False,
            **kernel_args,): # Use the trapezoid rule
        super().__init__()
        # H is number of hippos
        # D is the dimension (also shockingly n other places)
        # B is the batch
        # L is the length
        self.h = nHippos
        self.d = d_state // 2    
        self.channels = channels
        self.use_initial = use_initial
        self.zero_order_hold = zero_order_hold
        #
        # Use the trapezoid rule correct or just do zero-order hold.
        self.trap_rule = trap_rule

        _fp    = (self.channels, self.h, self.d)
        
        # Chebyshev initialization
        h_scale  = torch.exp(torch.arange(self.h)/self.h * math.log(dt_max/dt_min))
        angles   = torch.arange(self.d)*torch.pi
        theta_scale = h_scale if theta_scale else torch.ones(self.h)
        theta    = oe.contract('c,h,d->chd', torch.ones(self.channels), h_scale, angles)
        a        = -repeat(h_scale, 'h -> c h d', c=self.channels, d=self.d)
        #a = -oe.contract('c,h,d->chd', torch.ones(self.channels), _log_T,
        #                               torch.ones(self.d))
                                            
        self.register("theta", theta,learn_theta,lr=lr, wd=None)
        self.register("a", a, learn_a,lr=lr, wd=None)
        # The other maps 
        self.D = nn.Parameter(torch.randn(channels, self.h))
        
        if use_initial:
            self.b = nn.Parameter(torch.randn(*_fp))
            self.c = nn.Parameter(torch.randn(*_fp))
            self.x0 = nn.Parameter(torch.randn(channels, self.h, self.d))
        else:
            # This is an optimization that we combine q = c * b
            # It's as if we're setting x0 = 0.
            self.q = nn.Parameter(torch.randn(*_fp))

    def zoh_method(self, u):
        l  = u.size(-1)
        T = 1/(l-1) 
        zk        = T*torch.arange(u.size(-1), device=u.device).view(1,1,-1,1)
        ls        = torch.complex(-self.a.abs(), self.theta)
        term_0    = (torch.exp(ls*T) - 1)/ls
        base_term = (2*term_0.unsqueeze(2)*torch.exp(ls.unsqueeze(2)* zk)).real
        q  = self.b*self.c if self.use_initial else self.q
        f  = (q.unsqueeze(2)*base_term).sum(-1)
        y  = fft_conv(u,f)
        y  = y + oe.contract('bhl,ch->bchl', u, self.D)
        if self.use_initial:
            # This the cosine formula from the note
            cos_term = 2*T*torch.exp(-self.a.abs().unsqueeze(2) * zk)*torch.cos(   self.theta.unsqueeze(2) * zk)
            y = y + (2*(self.c*self.x0).unsqueeze(2)*cos_term).sum(-1)
        return rearrange(y, 'b c h l-> b (c h) l') # flatten the channels.

    def quadrature_method(self, u):
        # The input is now Batch x Hippos x Length
        l  = u.size(-1)
        T  = 1/(l-1) # the step size
        zk = T*torch.arange(l, device=u.device).view(1,1,-1,1)
        #T  = torch.exp(self.log_T).to(u.device).unsqueeze(-1).unsqueeze(-1)
        #zk = T*torch.arange(l, device=u.device).view(1,1,-1,1)
        # q and a are both C x H x D
        # zk is of length l we want a C x H x L matrix
        # From the note, we have 
        # f[k] = 2 sum_{j=1}^{d} q_j e^{a_j z_k} cos( z_k * theta_j )
        # we construct the body of the sum
        base_term = 2*T*torch.exp(-self.a.abs().unsqueeze(2) * zk)*torch.cos(   self.theta.unsqueeze(2) * zk)
        q  = self.b*self.c if self.use_initial else self.q
        f  = (q.unsqueeze(2)*base_term).sum(-1)

        # after summing f it is now an C H L matrix
        g  = u  # this is a B H L matrix 
        # we want to convolve on L and produce a B H C L
        #
        y = fft_conv(g,f)
        if self.trap_rule:  
            y = y - T*(oe.contract('ch,bhl -> bchl', f[:,:,0], g) + oe.contract('chl,bh -> bchl', f, g[:,:,0]))/2
    
        # Add in the skip connection with per-channel D matrix
        y = y + oe.contract('bhl,ch->bchl', u, self.D)
        # Add back the initial state
        if self.use_initial:
            y = y + (2*(self.c*self.x0).unsqueeze(2)*base_term).sum(-1)
        return rearrange(y, 'b c h l-> b (c h) l') # flatten the channels.

    def forward(self, u):
        return self.zoh_method(u) if self.zero_order_hold else self.quadrature_method(u)

# Below here are standard wrapper classes to handle
# (1) Non-linearity
# (2) Integration with the Hippo Code base
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