import torch
import torch.nn as nn
import opt_einsum as oe

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

# Utility class for registering the learning rate in the state spaces repo
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