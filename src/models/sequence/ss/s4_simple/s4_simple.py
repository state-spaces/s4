import torch
import torch.nn as nn
from einops import rearrange, repeat
import opt_einsum as oe
from src.models.sequence.ss.s4_simple.utils import OurModule, fft_conv

import math

class SimpleS4(OurModule):
    def __init__(self,
            nHippos,
            d_state =64,
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
        self.d = d_state // 2 # Adjustment for conjugate pairs
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