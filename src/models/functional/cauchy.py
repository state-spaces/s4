""" pykeops implementations of the core Cauchy kernel used in the S3 algorithm.

The interface of the Cauchy multiplication is:
    v: (N)
    z: (N)
    w: (L)
    Return: y (L)
      y_k = \sum_i v_i / (z_i - w_k)
"""
if __name__ == '__main__':
    import sys
    import pathlib
    p = pathlib.Path().absolute()
    print("Adding path: ", p)
    sys.path.append(str(p))

import math
import torch

from einops import rearrange
from benchmark import utils

import os
import pykeops
from pykeops.torch import LazyTensor, Genred

def _conj(x): return rearrange(torch.stack([x, x.conj()], dim=-1), '... i j -> ... (i j)')
def _broadcast_dims(*tensors):
    max_dim = max([len(tensor.shape) for tensor in tensors])
    tensors = [tensor.view((1,)*(max_dim-len(tensor.shape))+tensor.shape) for tensor in tensors]
    return tensors

def _c2r(x):
    # return torch.stack([x.real, x.imag], dim=-1)
    return torch.view_as_real(x)

def _r2c(x):
    # return torch.complex(x[..., 0], x[..., 1])
    return torch.view_as_complex(x)

def mult(B, C, z, w):
    r = (B.conj()*C).unsqueeze(-1) / (z.unsqueeze(-2) - w.unsqueeze(-1)) # (... N L)
    return torch.sum(r, dim=-2)


def mult_fast(B, C, z, w):
    B, C, z, w = _broadcast_dims(B, C, z, w)
    B_l = LazyTensor(rearrange(B, '... N -> ... N 1 1'))
    C_l = LazyTensor(rearrange(C, '... N -> ... N 1 1'))
    # prod = (B_l.conj() * C_l)
    prod = B_l * C_l
    w_l = LazyTensor(rearrange(w, '... N -> ... N 1 1'))
    z_l = LazyTensor(rearrange(z, '... L -> ... 1 L 1'))
    sub = z_l - w_l  # (b N L 1), for some reason it doesn't display the last dimension
    div = prod / sub
    s = div.sum(dim=len(B_l.shape)-2)
    return s.squeeze(-1)

def mult_genred(B, C, z, w):
    cauchy_mult = Genred(
        'ComplexDivide(ComplexMult(Conj(B), C), z-w)', # 'B * C / (z - w)',
        [
            'B = Vj(2)',
            'C = Vj(2)',
            'z = Vi(2)',
            'w = Vj(2)',
        ],
        reduction_op='Sum',
        axis=1,
    )

    B, C, z, w = _broadcast_dims(B, C, z, w)
    B = _c2r(B)
    C = _c2r(C)
    z = _c2r(z)
    w = _c2r(w)
    r = cauchy_mult(B, C, z, w, backend='GPU')
    return _r2c(r)

def cauchy(v, z, w):
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

def mult_pure(B, C, z, w):
    return cauchy(B.conj() * C, z, w, complex=True)

def cauchy_conj_slow(v, z, w):
    z = z.unsqueeze(-1)
    v = v.unsqueeze(-2)
    w = w.unsqueeze(-2)
    r = (z*v.real - (v*w.conj()).real) / ((z-w.real)**2 + w.imag**2)
    # r =  ((z-w.real)**2 + w.imag**2)
    return 2 * torch.sum(r, dim=-1)

def mult_conj_slow(B, C, z, w):
    v = B.conj() * C
    return cauchy_conj_slow(v[..., 0::2].contiguous(), z, w[..., 0::2].contiguous())


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
        # expr_num,
        # expr_denom,
        [
            'v = Vj(2)',
            'z = Vi(2)',
            'w = Vj(2)',
        ],
        reduction_op='Sum',
        axis=1,
        dtype='float32' if v.dtype == torch.cfloat else 'float64',
    )

    v, z, w = _broadcast_dims(v, z, w)
    v = _c2r(v)
    z = _c2r(z)
    w = _c2r(w)

    r = 2*cauchy_mult(v, z, w, backend='GPU')
    return _r2c(r)

def mult_conj(B, C, z, w, **kwargs):
    v = B.conj() * C
    return cauchy_conj(v[..., 0::2].contiguous(), z, w[..., 0::2].contiguous(), **kwargs)

def cauchy_conj_components(v_r, v_i, w_r, w_i, z_i):
    expr_num = 'Imag2Complex(zi*vr) - Real2Complex(vr*wr + vi*wi)'
    expr_denom = 'Real2Complex(Square(wr)+Square(wi)-Square(zi)) - Imag2Complex(IntCst(2)*zi*wr)'
    # expr_denom = 'IntCst(2)*Imag2Complex(zi*wr)'
    cauchy_mult = Genred(
        f'ComplexDivide({expr_num}, {expr_denom})',
        # expr_num,
        # expr_denom,
        [
            'vr = Vj(1)',
            'vi = Vj(1)',
            'wr = Vj(1)',
            'wi = Vj(1)',
            'zi = Vi(1)',
        ],
        reduction_op='Sum',
        axis=1,
    )

    v_r, v_i, w_r, w_i, z_i = _broadcast_dims(v_r, v_i, w_r, w_i, z_i)
    v_r = v_r.unsqueeze(-1)
    v_i = v_i.unsqueeze(-1)
    w_r = w_r.unsqueeze(-1)
    w_i = w_i.unsqueeze(-1)
    z_i = z_i.unsqueeze(-1)

    r = 2*cauchy_mult(v_r, v_i, w_r, w_i, z_i, backend='GPU')
    return _r2c(r)

def mult_conj_components(B, C, z, w, **kwargs):
    v = B.conj() * C
    v = v[..., 0::2].contiguous()
    w = w[..., 0::2].contiguous()
    r = cauchy_conj_components(v.real.contiguous(), v.imag.contiguous(), w.real.contiguous(), w.imag.contiguous(), z.imag.contiguous())
    return r

def cauchy_conj_components_lazy(v_r, v_i, w_r, w_i, z_i, type=1):
    v_r, v_i, w_r, w_i, z_i = _broadcast_dims(v_r, v_i, w_r, w_i, z_i)
    v_r = LazyTensor(rearrange(v_r, '... N -> ... 1 N 1'))
    v_i = LazyTensor(rearrange(v_i, '... N -> ... 1 N 1'))
    w_r = LazyTensor(rearrange(w_r, '... N -> ... 1 N 1'))
    w_i = LazyTensor(rearrange(w_i, '... N -> ... 1 N 1'))
    z_i = LazyTensor(rearrange(z_i, '... L -> ... L 1 1'))

    if type == 1:
        num = -v_r*w_r-v_i*w_i + 1j* z_i*v_r
        denom = w_r**2+w_i**2-z_i**2 - 2j*w_r*z_i
    else:
        # z = torch.complex(-w_r, z_i) # Not supported
        z = -w_r + 1j* z_i
        num = v_r * z - v_i*w_i
        denom = z*z + w_i**2 # z**2 is bugged for complex

    r = num / denom
    r = 2*r.sum(dim=len(z_i.shape)-1)
    return r.squeeze(-1)

def mult_conj_components_lazy(B, C, z, w, **kwargs):
    v = B.conj() * C
    v = v[..., 0::2].contiguous()
    w = w[..., 0::2].contiguous()
    return cauchy_conj_components_lazy(v.real.contiguous(), v.imag.contiguous(), w.real.contiguous(), w.imag.contiguous(), z.imag.contiguous(), **kwargs)

def cauchy_conj2(v, z, w):
    expr = 'ComplexDivide(v, z-w) + ComplexDivide(Conj(v), z-Conj(w))'
    # expr = 'ComplexDivide(v, z-w)'
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

    v, z, w = _broadcast_dims(v, z, w)
    if complex:
        v = _c2r(v)
        z = _c2r(z)
        w = _c2r(w)

    r = cauchy_mult(v, z, w, backend='GPU')
    return _r2c(r)

def mult_conj2(B, C, z, w, **kwargs):
    v = B.conj() * C
    return cauchy_conj2(v[..., 0::2].contiguous(), z, w[..., 0::2].contiguous(), **kwargs)

def resolvent_diagonalized(w, V, B, C, z):
    """
    w: (..., N)
    V: (..., N, N) represents diagonalization of A = V w V^{-1}
    B: (..., N)
    C: (..., N)
    z: (... L)
    Returns: (... L) represents C^T (z-A)^{-1} B
    """
    B = (B.to(V).unsqueeze(-2) @ V).squeeze(-2) # TODO try einsum for simplicity/efficiency
    C = (C.to(V).unsqueeze(-2) @ V).squeeze(-2)
    r = mult(B.conj(), C, z, w)
    return r

def trigger_compilation():
    """ Small function to trigger the compilation of a pykeops kernel

    Used in scenarios where we must manually control compilation, e.g. the multi-gpu case (https://github.com/getkeops/keops/issues/168) """
    B = 2
    N = 4
    L = 16

    w = torch.randn(B, N//2, dtype=torch.cfloat, device='cuda')
    v = torch.randn(B, N//2, dtype=torch.cfloat, device='cuda')
    z = torch.randn(B, L, dtype=torch.cfloat, device='cuda')
    w.requires_grad = True
    v.requires_grad = True

    cauchy_conj(v, z, w)


# Handle cache folder in a janky way for multi-gpu training
# print(pykeops.config.bin_folder)  # display default build_folder
# cache_folders = [
# ]
# has_cache_folder = False
# for f in cache_folders:
#     if os.path.isdir(f):
#         pykeops.set_bin_folder(f)  # change the build folder
#         has_cache_folder = True
# if not has_cache_folder:
#     # https://github.com/getkeops/keops/issues/168
#     import tempfile
#     with tempfile.TemporaryDirectory() as dirname:
#         pykeops.set_bin_folder(dirname)

#         # Run code that triggers compilation.
#         trigger_compilation()
# print(pykeops.config.bin_folder)  # display new build_folder


def data(B, N, L):
    dtype = torch.cfloat

    bs = B
    # bs = 1
    # N = 4
    # L = 8


    w = torch.randn(bs, N//2, dtype=dtype)
    B = torch.randn(bs, N//2, dtype=dtype)
    C = torch.randn(bs, N//2, dtype=dtype)
    z = torch.randn(bs, L, dtype=torch.float)
    if w.is_complex(): z = z * 1j


    w = _conj(w)
    B = _conj(B)
    C = _conj(C)

    w, B, C, z = utils.convert_data(w, B, C, z)
    return w, B, C, z

def profile_cauchy():
    B = 1024
    N = 64
    L = 16384

    w = torch.randn(N).contiguous()
    v = torch.randn(B, N).contiguous()
    z = torch.randn(L, dtype=torch.float).contiguous()

    # y = cauchy(v, z, w)
    y = cauchy_real(v, z, w)
    # utils.benchmark(cauchy,v, z, w, T=10)
    utils.benchmark(cauchy_real, v, z, w, T=10)
    # gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)

def profile_mults():
    B = 1024
    N = 64
    L = 16384
    w, B, C, z = data(B, N, L)

    # Test correctness
    utils.compare_outputs(
        # mult(B, C, z, w),
        mult_fast(B, C, z, w),
        mult_genred(B, C, z, w),
        mult_pure(B, C, z, w),
        # mult_conj_slow(B, C, z, w),
        mult_conj(B, C, z, w, num=1),
        mult_conj(B, C, z, w, num=2),
        mult_conj(B, C, z, w, denom=2),
        mult_conj(B, C, z, w, num=2, denom=2),
        mult_conj_components(B, C, z, w),
        mult_conj_components_lazy(B, C, z, w, type=1),
        mult_conj_components_lazy(B, C, z, w, type=2),
        mult_conj2(B, C, z, w),
        full=False,
        relative=True,
    )


    # Measure speed
    # utils.benchmark_forward(100, mult, B, C, z, w, desc='slow cauchy')
    utils.benchmark_forward(1000, mult_fast, B, C, z, w, desc='fast lazy')
    utils.benchmark_forward(1000, mult_genred, B, C, z, w, desc='fast genred')
    utils.benchmark_forward(1000, mult_pure, B, C, z, w, desc='fast cauchy')
    # utils.benchmark_forward(100, mult_conj_slow, B, C, z, w, desc='slow cauchy conj')
    utils.benchmark_forward(1000, mult_conj, B, C, z, w, num=1, desc='fast cauchy conj')
    utils.benchmark_forward(1000, mult_conj, B, C, z, w, num=2, desc='fast cauchy conj alternate num')
    utils.benchmark_forward(1000, mult_conj, B, C, z, w, denom=2, desc='fast cauchy conj alternate denom')
    utils.benchmark_forward(1000, mult_conj, B, C, z, w, num=2, denom=2, desc='fast cauchy conj alternate')
    utils.benchmark_forward(1000, mult_conj_components, B, C, z, w, desc='fast cauchy conj components')
    utils.benchmark_forward(1000, mult_conj_components_lazy, B, C, z, w, type=1, desc='lazy cauchy conj')
    utils.benchmark_forward(1000, mult_conj_components_lazy, B, C, z, w, type=2, desc='lazy cauchy conj alternate')
    utils.benchmark_forward(1000, mult_conj2, B, C, z, w, desc='fast cauchy')

    # utils.benchmark_backward(100, mult, B, C, z, w, desc='slow cauchy')
    # utils.benchmark_backward(100, mult_fast, B, C, z, w, desc='fast cauchy')
    # utils.benchmark_backward(100, mult_genred, B, C, z, w, desc='fast cauchy')
    # utils.benchmark_backward(100, mult_pure, B, C, z, w, desc='fast cauchy')


    # Measure memory
    # mem0 = utils.benchmark_memory(mult, B, C, z, w, desc='slow cauchy')
    mem1 = utils.benchmark_memory(mult_fast, B, C, z, w, desc='fast cauchy')
    mem1 = utils.benchmark_memory(mult_genred, B, C, z, w, desc='fast cauchy')
    mem1 = utils.benchmark_memory(mult_pure, B, C, z, w, desc='fast cauchy')
    # utils.benchmark_memory(100, mult_conj_slow, B, C, z, w, desc='slow cauchy conj')
    utils.benchmark_memory(mult_conj, B, C, z, w, num=1, desc='fast cauchy conj')
    utils.benchmark_memory(mult_conj, B, C, z, w, num=2, desc='fast cauchy conj alternate')
    utils.benchmark_memory(mult_conj_components, B, C, z, w, desc='fast cauchy conj components')
    utils.benchmark_memory(mult_conj_components_lazy, B, C, z, w, type=1, desc='lazy cauchy conj')
    utils.benchmark_memory(mult_conj_components_lazy, B, C, z, w, type=2, desc='lazy cauchy conj alternate')
    # print(f'mem savings: {mem0 / mem1}x')

if __name__ == '__main__':
    device = 'cuda'
    profile_cauchy()
