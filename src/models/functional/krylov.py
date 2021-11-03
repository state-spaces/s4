""" Compute a Krylov function efficiently. (S3 renames the Krylov function to a "state space kernel")

A : (N, N)
b : (N,)
c : (N,)
Return: [c^T A^i b for i in [L]]
"""

if __name__ == '__main__':
    import sys
    import pathlib
    p = pathlib.Path().absolute()
    print("Adding path: ", p)
    sys.path.append(str(p))


import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from src.models.functional.toeplitz import causal_convolution

def krylov_sequential(L, A, b, c=None):
    """ Constant matrix A

    A : (..., N, N)
    b : (..., N)
    c : (..., N)

    Returns
    if c:
    x : (..., L)
    x[i, l] = c[i] @ A^l @ b[i]

    else:
    x : (..., N, L)
    x[i, l] = A^l @ b[i]
    """

    # Check which of dim b and c is smaller to save memory
    if c is not None and c.numel() < b.numel():
        return krylov_sequential(L, A.transpose(-1, -2), c, b)

    b_ = b
    x = []
    for _ in range(L):
        if c is not None:
            x_ = torch.sum(c*b_, dim=-1) # (...) # could be faster with matmul or einsum?
        else:
            x_ = b_
        x.append(x_)
        b_ = (A @ b_.unsqueeze(-1)).squeeze(-1)

    x = torch.stack(x, dim=-1)
    return x


def krylov(L, A, b, c=None, return_power=False):
    """
    Compute the Krylov matrix (b, Ab, A^2b, ...) using the squaring trick.

    If return_power=True, return A^{L-1} as well
    """
    # TODO There is an edge case if L=1 where output doesn't get broadcasted, which might be an issue if caller is expecting broadcasting semantics... can deal with it if it arises

    x = b.unsqueeze(-1) # (..., N, 1)
    A_ = A

    AL = None
    if return_power:
        AL = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
        _L = L-1

    done = L == 1
    # loop invariant: _L represents how many indices left to compute
    while not done:
        if return_power:
            if _L % 2 == 1: AL = A_ @ AL
            _L //= 2

        # Save memory on last iteration
        l = x.shape[-1]
        if L - l <= l:
            done = True
            _x = x[..., :L-l]
        else: _x = x

        _x = A_ @ _x
        x = torch.cat([x, _x], dim=-1) # there might be a more efficient way of ordering axes
        if not done: A_ = A_ @ A_

    assert x.shape[-1] == L

    if c is not None:
        x = torch.einsum('...nl, ...n -> ...l', x, c)
    x = x.contiguous() # WOW!!
    if return_power:
        return x, AL
    else:
        return x

def power(L, A, v=None):
    """ Compute A^L and the scan sum_i A^i v_i

    A: (..., N, N)
    v: (..., N, L)
    """

    I = torch.eye(A.shape[-1]).to(A) # , dtype=A.dtype, device=A.device)

    powers = [A]
    l = 1
    while True:
        if L % 2 == 1: I = powers[-1] @ I
        L //= 2
        if L == 0: break
        l *= 2
        powers.append(powers[-1] @ powers[-1])

    if v is None: return I

    # Invariants:
    # powers[-1] := A^l
    # l := largest po2 at most L

    # Note that an alternative divide and conquer to compute the reduction is possible and can be embedded into the above loop without caching intermediate powers of A
    # We do this reverse divide-and-conquer for efficiency reasons:
    # 1) it involves fewer padding steps for non-po2 L
    # 2) it involves more contiguous arrays

    # Take care of edge case for non-po2 arrays
    # Note that this initial step is a no-op for the case of power of 2 (l == L)
    k = v.size(-1) - l
    v_ = powers.pop() @ v[..., l:]
    v = v[..., :l]
    v[..., :k] = v[..., :k] + v_

    # Handle reduction for power of 2
    while v.size(-1) > 1:
        v = rearrange(v, '... (z l) -> ... z l', z=2)
        v = v[..., 0, :] + powers.pop() @ v[..., 1, :]
    return I, v.squeeze(-1)

def krylov_toeplitz(L, A, b, c=None):
    """ Specializes to lower triangular Toeplitz matrix A represented by its diagonals

    A : (..., N)
    b : (..., N)
    c : (..., N)

    Returns
    x : (..., N, L)
    x[i, l] = A^l @ b[i]
    """
    x = b.unsqueeze(0) # (1, ..., N)
    A_ = A
    while x.shape[0] < L:
        xx = causal_convolution(A_, x)
        x = torch.cat([x, xx], dim=0) # there might be a more efficient way of ordering axes
        A_ = causal_convolution(A_, A_)
    x = x[:L, ...] # (L, ..., N)
    if c is not None:
        x = torch.einsum('l...n, ...n -> ...l', x, c)
    else:
        x = rearrange(x, 'l ... n -> ... n l')
    x = x.contiguous()
    return x

def krylov_toeplitz_(L, A, b, c=None):
    """ Padded version of krylov_toeplitz that saves some fft's

    TODO currently not faster than original version, not sure why
    """
    N = A.shape[-1]

    x = b.unsqueeze(0) # (1, ..., N)
    x = F.pad(x, (0, N))
    A = F.pad(A, (0, N))
    done = L == 1
    while not done:
        l = x.shape[0]
        # Save memory on last iteration
        if L - l <= l:
            done = True
            _x = x[:L-l]
        else: _x = x
        Af = torch.fft.rfft(A, n=2*N, dim=-1)
        xf = torch.fft.rfft(_x, n=2*N, dim=-1)
        xf_ = Af * xf
        x_ = torch.fft.irfft(xf_, n=2*N, dim=-1)
        x_[..., N:] = 0
        x = torch.cat([x, x_], dim=0) # there might be a more efficient way of ordering axes
        if not done:
            A = torch.fft.irfft(Af*Af, n=2*N, dim=-1)
            A[..., N:] = 0
    x = x[:L, ..., :N] # (L, ..., N)
    if c is not None:
        x = torch.einsum('l...n, ...n -> ...l', x, c)
    else:
        x = rearrange(x, 'l ... n -> ... n l')
    x = x.contiguous()
    return x


def generate_data(L, B, N, random=False):
    from models.nn.toeplitz import construct_toeplitz
    from models.hippo.transition import transition

    if random:
        # A = torch.eye(N) + torch.normal(0, 1, size=(B, N, N)) / (N**.5) / L
        A = torch.normal(0, 1, size=(B, N)) / (N**.5) # / L
        A = construct_toeplitz(A)
        b = torch.normal(0, 1, size=(B, N))
        c = torch.normal(0, 1, size=(B, N))
    else:
        A, B = transition('lagt', N)
        A = torch.eye(N) + .01 * torch.Tensor(A)
        b = torch.Tensor(B[:, 0])
        c = torch.ones(N)

    A = A.to(device)
    b = b.to(device)
    c = c.to(device)
    A.requires_grad = True
    b.requires_grad = True
    c.requires_grad = True
    return A, b, c

def test_power():
    L = 16
    B = 2
    N = 4

    A = 2 * torch.eye(N)
    b = torch.ones(B, N, L)

    AL, v = power(L, A, b)
    print(AL, v)


def test_krylov():
    from .unroll import unroll, parallel_unroll_iterative, variable_unroll_sequential, variable_unroll
    L = 10
    B = 2
    N = 4

    A = 2 * torch.eye(N)
    b = torch.ones(B, N)
    c = torch.ones(N)

    # Check unroll methods
    print("Checking Unroll")
    u = torch.cat([b.unsqueeze(0), b.new_zeros(L-1, *b.shape)], dim=0) # (L, B, N)
    print(u.shape)

    y = unroll(A, u)
    y = torch.sum(c * y, dim=-1) # (L, B)
    print(y)

    y = parallel_unroll_iterative(A, u)
    y = torch.sum(c * y, dim=-1) # (L, B)
    print(y)

    A_ = repeat(A, '... -> l ...', l=L)

    y = variable_unroll_sequential(A_, u)
    y = torch.sum(c * y, dim=-1) # (L, B)
    print(y)

    y = variable_unroll(A_, u)
    y = torch.sum(c * y, dim=-1) # (L, B)
    print(y)

    y = variable_unroll(A_, u).permute(1, 2, 0) # (B, N, L)
    print(y)

    # # Check new krylov methods
    print("Checking Krylov")

    y = krylov_sequential(L, A, b)
    print(y)

    y = krylov(L, A, b, return_power=False)
    print(y)

    y, AL = krylov(L, A, b, return_power=True)
    print(y, AL)

def test_toeplitz():
    """ Test correctness of Krylov Toeplitz variants """

    L = 10
    B = 2
    N = 4

    A, b, _ = generate_data(L, B, N, random=False)

    y = krylov_sequential(L, A, b, c=None)
    print(y)

    y = krylov(L, A, b)
    print(y)

    y = krylov_toeplitz(L, A[..., 0], b)
    print(y)

    y = krylov_toeplitz_(L, A[..., 0], b)
    print(y)

def benchmark_krylov(test=True):
    """ Benchmark the Krylov functions with shapes as in an LSSL layer """

    from models.nn.unroll import parallel_unroll_recursive, variable_unroll_toeplitz
    from models.nn.toeplitz import construct_toeplitz

    if test:
        L, B, H, M, N = 10, 2, 3, 4, 5 # for testing
    else:
        L = 1024
        B = 50
        # H = 32
        H = 256
        # H = 512
        M = 1
        # N = 32
        N = 256
        # N = 512

    # A = torch.ones(H, N)
    A = 1e-3 * torch.ones(N)
    A[..., 0] += 1
    b = torch.ones(H, N)
    c = torch.ones(H, M, N)
    u = torch.ones(L, B, H)

    A = A.to(device)
    b = b.to(device)
    c = c.to(device)
    u = u.to(device)
    A_explicit = construct_toeplitz(A).to(device)


    def krylov_to_lssl(y):
        y = causal_convolution(y.unsqueeze(1), u.unsqueeze(-1).transpose(0,-1)) # (M, B, H, L)
        y = y.transpose(0, -1) # rearrange(y, 'm b h l -> l b h m') # (L, B, H, M)
        return y

    def method0():
        k = krylov(L, A_explicit, b, c.transpose(0, 1)) # (M, H, L)
        return krylov_to_lssl(k) # (L, B, H, M)
    def method1():
        k = krylov_toeplitz(L, A, b, c.transpose(0, 1)) # (M, H, L)
        return krylov_to_lssl(k) # (L, B, H, M)
    def method2():
        k = krylov_toeplitz_(L, A, b, c.transpose(0, 1)) # (M, H, L)
        return krylov_to_lssl(k) # (L, B, H, M)
    def method3():
        k = parallel_unroll_recursive(A_explicit, b*u.unsqueeze(-1)) # (L, B, H, N)
        y = torch.sum(c * k.unsqueeze(-2), dim=-1)
        return y
    def method4():
        k = variable_unroll_toeplitz(A, b*u.unsqueeze(-1), variable=False) # (L, B, H, N)
        y = torch.sum(c * k.unsqueeze(-2), dim=-1)
        return y

    # Check outputs to test correctness
    if test:
        print("Deviation from vanilla Krylov")
        y = method0()
        print(torch.linalg.norm(method0()-y))
        print(torch.linalg.norm(method1()-y))
        print(torch.linalg.norm(method2()-y))
        print(torch.linalg.norm(method3()-y))
        print(torch.linalg.norm(method4()-y))

    utils.benchmark_forward(10, method0, desc='Matrix Krylov')
    utils.benchmark_forward(10, method1, desc='Toeplitz Krylov')
    utils.benchmark_forward(10, method2, desc='Toeplitz Krylov fast')
    ## Note that scan runs out of memory on large problems since it expands to (L, B, H, N)
    # utils.benchmark_forward(10, method3, desc='Matrix scan')
    # utils.benchmark_forward(10, method4, desc='Toeplitz scan')
    # utils.pytorch_profiler(10, method2)

    ### Results
    # (L, B, H, M, N)
    # CPU
    # (100, 50, 32, 1, 256) 9ms, 7ms, 900ms
    # (100, 50, 256, 1, 256) 87, 122
    # (100, 50, 512, 1, 512) 508, 643
    # (1024, 50, 512, 1, 512) 4.5, 5.7 s
    # T4
    # (100, 50, 32, 1, 256) 1.2, 2.7
    # (100, 50, 256, 1, 256) 5.5, 6.3 ms
    # (1024, 50, 256, 1, 256) 32.6, 47.4, 41.1 / 2.4, 3.7, 4.5
    # (100, 50, 512, 1, 512) 16.3, 19.5 ms / 1.7, 2.5 Gb
    # (1024, 50, 512, 1, 512) 138, 172, 170 ms / 6.3, 11.4 Gb

    ### 2021-06-02
    # full LSSL computation
    # (256, 20, 32, 1, 32) 1.0, 2.6, 1.9, 3.4, 7.2 ms
    # (256, 20, 64, 1, 64) 1.2, 2.8, 1.9, 10.5, 23.4 ms
    # (1024, 20, 32, 1, 32) 1.3, 3.4, 2.3, 10.4, 24.6 ms

    # After fixing Krylov speed: pure Krylov computation
    # (1024, 50, 256, 1, 256) 26.6, 34.8, 36.8 ms
    # (1024, 50, 512, 1, 512) 110, 139, 149 ms


if __name__ == '__main__':
    from benchmark import utils

    device = 'cuda' # 'cpu'
    device = torch.device(device)

    # test_krylov()
    # test_toeplitz()
    # benchmark_krylov(test=True)
    test_power()
