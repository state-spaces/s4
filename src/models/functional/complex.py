""" Custom implementation of fast complex operations.

This was written during earlier versions of Pytorch.
Later versions have native support for complex numbers and much of this is no longer necessary.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.torch.utils.dlpack import to_dlpack, from_dlpack


use_cupy = True
try:
    import cupy as cp
except:
    use_cupy = False
use_pt_native = hasattr(torch, 'view_as_complex')


def complex_mul_native(X, Y):
    return torch.view_as_real(torch.view_as_complex(X) * torch.view_as_complex(Y))


def conjugate_native(X):
    return torch.view_as_real(torch.view_as_complex(X).conj())


def torch2numpy(X):
    """Convert a torch float32 tensor to a numpy array, sharing the same memory.
    """
    return X.detach().numpy()


def torch2cupy(tensor):
    return cp.fromDlpack(to_dlpack(tensor.cuda()))


def cupy2torch(tensor):
    return from_dlpack(tensor.toDlpack())


def real_to_complex(X):
    """A version of X that's complex (i.e., last dimension is 2).
    Parameters:
        X: (...) tensor
    Return:
        X_complex: (..., 2) tensor
    """
    return torch.stack((X, torch.zeros_like(X)), dim=-1)


def conjugate_torch(X):
    assert X.shape[-1] == 2, 'Last dimension must be 2'
    return X * torch.tensor((1, -1), dtype=X.dtype, device=X.device)


class Conjugate(torch.autograd.Function):
    '''X is a complex64 tensors but stored as float32 tensors, with last dimension = 2.
    '''
    @staticmethod
    def forward(ctx, X):
        assert X.shape[-1] == 2, 'Last dimension must be 2'
        if X.is_cuda:
            if use_cupy:
                # TODO: do we need .contiguous here? I think it doesn't work if the last dimension isn't contiguous
                return cupy2torch(torch2cupy(X).view('complex64').conj().view('float32'))
            else:
                return conjugate_torch(X)
        else:
            return torch.from_numpy(np.ascontiguousarray(torch2numpy(X)).view('complex64').conj().view('float32'))

    @staticmethod
    def backward(ctx, grad):
        return Conjugate.apply(grad)


conjugate = conjugate_native if use_pt_native else Conjugate.apply


def complex_mul_torch(X, Y):
    assert X.shape[-1] == 2 and Y.shape[-1] == 2, 'Last dimension must be 2'
    return torch.stack(
        (X[..., 0] * Y[..., 0] - X[..., 1] * Y[..., 1],
         X[..., 0] * Y[..., 1] + X[..., 1] * Y[..., 0]),
        dim=-1)


def complex_mul_numpy(X, Y):
    assert X.shape[-1] == 2 and Y.shape[-1] == 2, 'Last dimension must be 2'
    X_np = np.ascontiguousarray(torch2numpy(X)).view('complex64')
    Y_np = np.ascontiguousarray(torch2numpy(Y)).view('complex64')
    return torch.from_numpy((X_np * Y_np).view('float32'))


class ComplexMul(torch.autograd.Function):
    '''X and Y are complex64 tensors but stored as float32 tensors, with last dimension = 2.
    '''
    @staticmethod
    def forward(ctx, X, Y):
        assert X.shape[-1] == 2 and Y.shape[-1] == 2, 'Last dimension must be 2'
        ctx.save_for_backward(X, Y)
        if X.is_cuda:
            assert Y.is_cuda, 'X and Y must both be torch.cuda.FloatTensor'
            if use_cupy:
                # TODO: do we need .contiguous here? I think it doesn't work if the last dimension isn't contiguous
                return cupy2torch((torch2cupy(X).view('complex64') * torch2cupy(Y).view('complex64')).view('float32'))
            else:
                return complex_mul_torch(X, Y)
        else:
            assert not Y.is_cuda, 'X and Y must both be torch.FloatTensor'
            X_np = np.ascontiguousarray(torch2numpy(X)).view('complex64')
            Y_np = np.ascontiguousarray(torch2numpy(Y)).view('complex64')
            return torch.from_numpy((X_np * Y_np).view('float32'))

    @staticmethod
    def backward(ctx, grad):
        X, Y = ctx.saved_tensors
        grad_X, grad_Y = None, None
        if ctx.needs_input_grad[0]:
            grad_X = ComplexMul.apply(grad, conjugate(Y)).sum_to_size(*X.shape)
        if ctx.needs_input_grad[1]:
            grad_Y = ComplexMul.apply(grad, conjugate(X)).sum_to_size(*Y.shape)
        return grad_X, grad_Y

complex_mul = ComplexMul.apply if use_cupy else complex_mul_torch
if use_pt_native:
    complex_mul = complex_mul_native
