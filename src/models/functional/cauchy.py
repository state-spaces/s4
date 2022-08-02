"""pykeops implementations of the Cauchy matrix multiplication used in the S4 algorithm.

The interface of the Cauchy multiplication is:
    v: (N)
    z: (N)
    w: (L)
    Return: y (L)
      y_k = \sum_i v_i / (z_i - w_k)
"""
import math
import torch

from einops import rearrange

import os

try:
    import pykeops
    from pykeops.torch import LazyTensor, Genred
except:
    pass

_conj = lambda x: torch.cat([x, x.conj()], dim=-1)
def _broadcast_dims(*tensors):
    max_dim = max([len(tensor.shape) for tensor in tensors])
    tensors = [tensor.view((1,)*(max_dim-len(tensor.shape))+tensor.shape) for tensor in tensors]
    return tensors

def _c2r(x): return torch.view_as_real(x)
def _r2c(x): return torch.view_as_complex(x)

def cauchy_naive(v, z, w, conj=True):
    """
    v: (..., N)
    z: (..., L)
    w: (..., N)
    returns: (..., L) \sum v/(z-w)
    """
    if conj:
        v = _conj(v)
        w = _conj(w)
    cauchy_matrix = v.unsqueeze(-1) / (z.unsqueeze(-2) - w.unsqueeze(-1)) # (... N L)
    return torch.sum(cauchy_matrix, dim=-2)

def cauchy(v, z, w, conj=False):
    expr = 'ComplexDivide(v, z-w)'
    cauchy_mult = Genred(
        expr,
        [
            'v = Vj(2)',
            'z = Vi(2)',
            'w = Vj(2)',
        ],
        reduction_op='Sum',
        axis=1,
    )

    if conj:
        v = _conj(v)
        w = _conj(w)
    v, z, w = _broadcast_dims(v, z, w)
    v = _c2r(v)
    z = _c2r(z)
    w = _c2r(w)

    r = cauchy_mult(v, z, w, backend='GPU')
    return _r2c(r)

def cauchy_real(v, z, w):
    expr = 'v / (z - w)'
    cauchy_mult = Genred(
        expr,
        [
            'v = Vj(1)',
            'z = Vi(1)',
            'w = Vj(1)',
        ],
        reduction_op='Sum',
        axis=1,
    )
    v, z, w = _broadcast_dims(v, z, w)
    v = v.unsqueeze(-1)
    z = z.unsqueeze(-1)
    w = w.unsqueeze(-1)

    r = cauchy_mult(v, z, w, backend='GPU')
    return r


def cauchy_conj(v, z, w, num=2, denom=2):
    if num == 1:
        expr_num = 'z * ComplexReal(v) - Real2Complex(ComplexReal(v)*ComplexReal(w) + ComplexImag(v)*ComplexImag(w))'
    elif num == 2:
        expr_num = 'z * ComplexReal(v) - Real2Complex(Sum(v * w))'
    else: raise NotImplementedError

    if denom == 1:
        expr_denom = 'ComplexMult(z-Real2Complex(ComplexReal(w)), z-Real2Complex(ComplexReal(w))) + Real2Complex(Square(ComplexImag(w)))'
    elif denom == 2:
        expr_denom = 'ComplexMult(z-w, z-Conj(w))'
    else: raise NotImplementedError

    cauchy_mult = Genred(
        f'ComplexDivide({expr_num}, {expr_denom})',
        [
            'v = Vj(2)',
            'z = Vi(2)',
            'w = Vj(2)',
        ],
        reduction_op='Sum',
        axis=1,
    )

    v, z, w = _broadcast_dims(v, z, w)
    v = _c2r(v)
    z = _c2r(z)
    w = _c2r(w)

    r = 2*cauchy_mult(v, z, w, backend='GPU')
    return _r2c(r)
