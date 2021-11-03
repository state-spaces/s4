""" Old utilities for parallel scan implementation of Linear RNNs. """
# TODO this file could use much cleanup

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from src.models.functional.toeplitz import triangular_toeplitz_multiply, triangular_toeplitz_multiply_padded
from src.utils.permutations import bitreversal_po2, bitreversal_permutation



### Utilities


def shift_up(a, s=None, drop=True, dim=0):
    assert dim == 0
    if s is None:
        s = torch.zeros_like(a[0, ...])
    s = s.unsqueeze(dim)
    if drop:
        a = a[:-1, ...]
    return torch.cat((s, a), dim=dim)

def interleave(a, b, uneven=False, dim=0):
    """ Interleave two tensors of same shape """
    # assert(a.shape == b.shape)
    assert dim == 0 # TODO temporary to make handling uneven case easier
    if dim < 0:
        dim = N + dim
    if uneven:
        a_ = a[-1:, ...]
        a = a[:-1, ...]
    c = torch.stack((a, b), dim+1)
    out_shape = list(a.shape)
    out_shape[dim] *= 2
    c = c.view(out_shape)
    if uneven:
        c = torch.cat((c, a_), dim=dim)
    return c

def batch_mult(A, u, has_batch=None):
    """ Matrix mult A @ u with special case to save memory if u has additional batch dim

    The batch dimension is assumed to be the second dimension
    A : (L, ..., N, N)
    u : (L, [B], ..., N)
    has_batch: True, False, or None. If None, determined automatically

    Output:
    x : (L, [B], ..., N)
      A @ u broadcasted appropriately
    """

    if has_batch is None:
        has_batch = len(u.shape) >= len(A.shape)

    if has_batch:
        u = u.permute([0] + list(range(2, len(u.shape))) + [1])
    else:
        u = u.unsqueeze(-1)
    v = (A @ u)
    if has_batch:
        v = v.permute([0] + [len(u.shape)-1] + list(range(1, len(u.shape)-1)))
    else:
        v = v[..., 0]
    return v



### Main unrolling functions

def unroll(A, u):
    """
    A : (..., N, N) # TODO I think this can't take batch dimension?
    u : (L, ..., N)
    output : x (..., N) # TODO a lot of these shapes are wrong
    x[i, ...] = A^{i} @ u[0, ...] + ... + A @ u[i-1, ...] + u[i, ...]
    """

    m = u.new_zeros(u.shape[1:])
    outputs = []
    for u_ in torch.unbind(u, dim=0):
        m = F.linear(m, A) + u_
        outputs.append(m)

    output = torch.stack(outputs, dim=0)
    return output


def parallel_unroll_recursive(A, u):
    """ Bottom-up divide-and-conquer version of unroll. """

    # Main recursive function
    def parallel_unroll_recursive_(A, u):
        if u.shape[0] == 1:
            return u

        u_evens = u[0::2, ...]
        u_odds = u[1::2, ...]

        # u2 = F.linear(u_evens, A) + u_odds
        u2 = (A @ u_evens.unsqueeze(-1)).squeeze(-1) + u_odds
        A2 = A @ A

        x_odds = parallel_unroll_recursive_(A2, u2)
        # x_evens = F.linear(shift_up(x_odds), A) + u_evens
        x_evens = (A @ shift_up(x_odds).unsqueeze(-1)).squeeze(-1) + u_evens

        x = interleave(x_evens, x_odds, dim=0)
        return x

    # Pad u to power of 2
    n = u.shape[0]
    m = int(math.ceil(math.log(n)/math.log(2)))
    N = 1 << m
    u = torch.cat((u, u.new_zeros((N-u.shape[0],) + u.shape[1:] )), dim=0)

    return parallel_unroll_recursive_(A, u)[:n, ...]



def parallel_unroll_recursive_br(A, u):
    """ Same as parallel_unroll_recursive but uses bit reversal for locality. """

    # Main recursive function
    def parallel_unroll_recursive_br_(A, u):
        n = u.shape[0]
        if n == 1:
            return u

        m = n//2
        u_0 = u[:m, ...]
        u_1 = u[m:, ...]

        u2 = F.linear(u_0, A) + u_1
        A2 = A @ A

        x_1 = parallel_unroll_recursive_br_(A2, u2)
        x_0 = F.linear(shift_up(x_1), A) + u_0

        # x = torch.cat((x_0, x_1), dim=0) # is there a way to do this with cat?
        x = interleave(x_0, x_1, dim=0)
        return x

    # Pad u to power of 2
    n = u.shape[0]
    m = int(math.ceil(math.log(n)/math.log(2)))
    N = 1 << m
    u = torch.cat((u, u.new_zeros((N-u.shape[0],) + u.shape[1:] )), dim=0)

    # Apply bit reversal
    br = bitreversal_po2(N)
    u = u[br, ...]

    x = parallel_unroll_recursive_br_(A, u)
    return x[:n, ...]

def parallel_unroll_iterative(A, u):
    """ Bottom-up divide-and-conquer version of unroll, implemented iteratively """

    # Pad u to power of 2
    n = u.shape[0]
    m = int(math.ceil(math.log(n)/math.log(2)))
    N = 1 << m
    u = torch.cat((u, u.new_zeros((N-u.shape[0],) + u.shape[1:] )), dim=0)

    # Apply bit reversal
    br = bitreversal_po2(N)
    u = u[br, ...]

    # Main recursive loop, flattened
    us = [] # stores the u_0 terms in the recursive version
    N_ = N
    As = [] # stores the A matrices
    for l in range(m):
        N_ = N_ // 2
        As.append(A)
        u_0 = u[:N_, ...]
        us.append(u_0)
        u = F.linear(u_0, A) + u[N_:, ...]
        A = A @ A
    x_0 = []
    x = u # x_1
    for l in range(m-1, -1, -1):
        x_0 = F.linear(shift_up(x), As[l]) + us[l]
        x = interleave(x_0, x, dim=0)

    return x[:n, ...]


def variable_unroll_sequential(A, u, s=None, variable=True):
    """ Unroll with variable (in time/length) transitions A.

    A : ([L], ..., N, N) dimension L should exist iff variable is True
    u : (L, [B], ..., N) updates
    s : ([B], ..., N) start state
    output : x (..., N)
    x[i, ...] = A[i]..A[0] @ s + A[i..1] @ u[0] + ... + A[i] @ u[i-1] + u[i]
    """

    if s is None:
        s = torch.zeros_like(u[0])

    if not variable:
        A = A.expand((u.shape[0],) + A.shape)
    has_batch = len(u.shape) >= len(A.shape)

    outputs = []
    for (A_, u_) in zip(torch.unbind(A, dim=0), torch.unbind(u, dim=0)):
        # s = F.linear(s, A_) + u_
        # print("shapes", A_.shape, s.shape, has_batch)
        s = batch_mult(A_.unsqueeze(0), s.unsqueeze(0), has_batch)[0]
        # breakpoint()
        s = s + u_
        outputs.append(s)

    output = torch.stack(outputs, dim=0)
    return output



def variable_unroll(A, u, s=None, variable=True, recurse_limit=16):
    """ Bottom-up divide-and-conquer version of variable_unroll. """

    if u.shape[0] <= recurse_limit:
        return variable_unroll_sequential(A, u, s, variable)

    if s is None:
        s = torch.zeros_like(u[0])

    uneven = u.shape[0] % 2 == 1
    has_batch = len(u.shape) >= len(A.shape)

    u_0 = u[0::2, ...]
    u_1  = u[1::2, ...]

    if variable:
        A_0 = A[0::2, ...]
        A_1  = A[1::2, ...]
    else:
        A_0 = A
        A_1 = A

    u_0_ = u_0
    A_0_ = A_0
    if uneven:
        u_0_ = u_0[:-1, ...]
        if variable:
            A_0_ = A_0[:-1, ...]

    u_10 = batch_mult(A_1, u_0_, has_batch)
    u_10 = u_10 + u_1
    A_10 = A_1 @ A_0_

    # Recursive call
    x_1 = variable_unroll(A_10, u_10, s, variable, recurse_limit)

    x_0 = shift_up(x_1, s, drop=not uneven)
    x_0 = batch_mult(A_0, x_0, has_batch)
    x_0 = x_0 + u_0


    x = interleave(x_0, x_1, uneven, dim=0) # For some reason this interleave is slower than in the (non-multi) unroll_recursive
    return x

def variable_unroll_general_sequential(A, u, s, op, variable=True):
    """ Unroll with variable (in time/length) transitions A with general associative operation

    A : ([L], ..., N, N) dimension L should exist iff variable is True
    u : (L, [B], ..., N) updates
    s : ([B], ..., N) start state
    output : x (..., N)
    x[i, ...] = A[i]..A[0] s + A[i..1] u[0] + ... + A[i] u[i-1] + u[i]
    """

    if not variable:
        A = A.expand((u.shape[0],) + A.shape)

    outputs = []
    for (A_, u_) in zip(torch.unbind(A, dim=0), torch.unbind(u, dim=0)):
        s = op(A_, s)
        s = s + u_
        outputs.append(s)

    output = torch.stack(outputs, dim=0)
    return output

def variable_unroll_matrix_sequential(A, u, s=None, variable=True):
    if s is None:
        s = torch.zeros_like(u[0])

    if not variable:
        A = A.expand((u.shape[0],) + A.shape)
    # has_batch = len(u.shape) >= len(A.shape)

    # op = lambda x, y: batch_mult(x.unsqueeze(0), y.unsqueeze(0), has_batch)[0]
    op = lambda x, y: batch_mult(x.unsqueeze(0), y.unsqueeze(0))[0]

    return variable_unroll_general_sequential(A, u, s, op, variable=True)

def variable_unroll_toeplitz_sequential(A, u, s=None, variable=True, pad=False):
    if s is None:
        s = torch.zeros_like(u[0])

    if not variable:
        A = A.expand((u.shape[0],) + A.shape)
    # has_batch = len(u.shape) >= len(A.shape)

    # op = lambda x, y: batch_mult(x.unsqueeze(0), y.unsqueeze(0), has_batch)[0]
    # op = lambda x, y: batch_mult(x.unsqueeze(0), y.unsqueeze(0))[0]

    if pad:
        n = A.shape[-1]
        # print("shapes", A.shape, u.shape)
        A = F.pad(A, (0, n))
        u = F.pad(u, (0, n))
        s = F.pad(s, (0, n))
        # print("shapes", A.shape, u.shape)
        ret = variable_unroll_general_sequential(A, u, s, triangular_toeplitz_multiply_padded, variable=True)
        ret = ret[..., :n]
        return ret

    return variable_unroll_general_sequential(A, u, s, triangular_toeplitz_multiply, variable=True)



### General parallel scan functions with generic binary composition operators

def variable_unroll_general(A, u, s, op, compose_op=None, sequential_op=None, variable=True, recurse_limit=16):
    """ Bottom-up divide-and-conquer version of variable_unroll.

    compose is an optional function that defines how to compose A without multiplying by a leaf u
    """

    if u.shape[0] <= recurse_limit:
        if sequential_op is None:
            sequential_op = op
        return variable_unroll_general_sequential(A, u, s, sequential_op, variable)

    if compose_op is None:
        compose_op = op

    uneven = u.shape[0] % 2 == 1
    has_batch = len(u.shape) >= len(A.shape)

    u_0 = u[0::2, ...]
    u_1 = u[1::2, ...]

    if variable:
        A_0 = A[0::2, ...]
        A_1 = A[1::2, ...]
    else:
        A_0 = A
        A_1 = A

    u_0_ = u_0
    A_0_ = A_0
    if uneven:
        u_0_ = u_0[:-1, ...]
        if variable:
            A_0_ = A_0[:-1, ...]

    u_10 = op(A_1, u_0_) # batch_mult(A_1, u_0_, has_batch)
    u_10 = u_10 + u_1
    A_10 = compose_op(A_1, A_0_)

    # Recursive call
    x_1 = variable_unroll_general(A_10, u_10, s, op, compose_op, sequential_op, variable=variable, recurse_limit=recurse_limit)

    x_0 = shift_up(x_1, s, drop=not uneven)
    x_0 = op(A_0, x_0) # batch_mult(A_0, x_0, has_batch)
    x_0 = x_0 + u_0


    x = interleave(x_0, x_1, uneven, dim=0) # For some reason this interleave is slower than in the (non-multi) unroll_recursive
    return x

def variable_unroll_matrix(A, u, s=None, variable=True, recurse_limit=16):
    if s is None:
        s = torch.zeros_like(u[0])
    has_batch = len(u.shape) >= len(A.shape)
    op = lambda x, y: batch_mult(x, y, has_batch)
    sequential_op = lambda x, y: batch_mult(x.unsqueeze(0), y.unsqueeze(0), has_batch)[0]
    matmul = lambda x, y: x @ y
    return variable_unroll_general(A, u, s, op, compose_op=matmul, sequential_op=sequential_op, variable=variable, recurse_limit=recurse_limit)

def variable_unroll_toeplitz(A, u, s=None, variable=True, recurse_limit=8, pad=False):
    """ Unroll with variable (in time/length) transitions A with general associative operation

    A : ([L], ..., N) dimension L should exist iff variable is True
    u : (L, [B], ..., N) updates
    s : ([B], ..., N) start state
    output : x (L, [B], ..., N) same shape as u
    x[i, ...] = A[i]..A[0] s + A[i..1] u[0] + ... + A[i] u[i-1] + u[i]
    """
    # Add the batch dimension to A if necessary
    A_batch_dims = len(A.shape) - int(variable)
    u_batch_dims = len(u.shape)-1
    if u_batch_dims > A_batch_dims:
        # assert u_batch_dims == A_batch_dims + 1
        if variable:
            while len(A.shape) < len(u.shape):
                A = A.unsqueeze(1)
        # else:
        #     A = A.unsqueeze(0)

    if s is None:
        s = torch.zeros_like(u[0])

    if pad:
        n = A.shape[-1]
        # print("shapes", A.shape, u.shape)
        A = F.pad(A, (0, n))
        u = F.pad(u, (0, n))
        s = F.pad(s, (0, n))
        # print("shapes", A.shape, u.shape)
        op = triangular_toeplitz_multiply_padded
        ret = variable_unroll_general(A, u, s, op, compose_op=op, variable=variable, recurse_limit=recurse_limit)
        ret = ret[..., :n]
        return ret

    op = triangular_toeplitz_multiply
    ret = variable_unroll_general(A, u, s, op, compose_op=op, variable=variable, recurse_limit=recurse_limit)
    return ret



### Testing

def test_correctness():
    print("Testing Correctness\n====================")

    # Test sequential unroll
    L = 3
    A = torch.Tensor([[1, 1], [1, 0]])
    u = torch.ones((L, 2))
    x = unroll(A, u)
    assert torch.isclose(x, torch.Tensor([[1., 1.], [3., 2.], [6., 4.]])).all()

    # Test utilities
    assert torch.isclose(shift_up(x), torch.Tensor([[0., 0.], [1., 1.], [3., 2.]])).all()
    assert torch.isclose(interleave(x, x), torch.Tensor([[1., 1.], [1., 1.], [3., 2.], [3., 2.], [6., 4.], [6., 4.]])).all()

    # Test parallel unroll
    x = parallel_unroll_recursive(A, u)
    assert torch.isclose(x, torch.Tensor([[1., 1.], [3., 2.], [6., 4.]])).all()

    # Powers
    L = 12
    A = torch.Tensor([[1, 0, 0], [2, 1, 0], [3, 3, 1]])
    u = torch.ones((L, 3))
    x = parallel_unroll_recursive(A, u)
    print("recursive", x)
    x = parallel_unroll_recursive_br(A, u)
    print("recursive_br", x)
    x = parallel_unroll_iterative(A, u)
    print("iterative_br", x)


    A = A.repeat((L, 1, 1))
    s = torch.zeros(3)
    print("A shape", A.shape)
    x = variable_unroll_sequential(A, u, s)
    print("variable_unroll", x)
    x = variable_unroll(A, u, s)
    print("parallel_variable_unroll", x)


def generate_data(L, N, B=None, cuda=True):
    A = torch.eye(N) + torch.normal(0, 1, size=(N, N)) / (N**.5) / L
    u = torch.normal(0, 1, size=(L, B, N))


    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda:0') if cuda else torch.device('cpu')
    A = A.to(device)
    u = u.to(device)
    return A, u

def test_stability():
    print("Testing Stability\n====================")
    L = 256
    N = L // 2
    B = 100
    A, u = generate_data(L, N, B)

    x = unroll(A, u)
    x1 = parallel_unroll_recursive(A, u)
    x2 = parallel_unroll_recursive_br(A, u)
    x3 = parallel_unroll_iterative(A, u)
    print("norm error", torch.norm(x-x1))
    print("norm error", torch.norm(x-x2))
    print("norm error", torch.norm(x-x3))
    # print(x-x1)
    # print(x-x2)
    # print(x-x3)
    print("max error", torch.max(torch.abs(x-x1)))
    print("max error", torch.max(torch.abs(x-x2)))
    print("max error", torch.max(torch.abs(x-x3)))

    A = A.repeat((L, 1, 1))
    x = variable_unroll_sequential(A, u)
    x_ = variable_unroll(A, u)
    # x_ = variable_unroll_matrix_sequential(A, u)
    x_ = variable_unroll_matrix(A, u)
    print(x-x_)
    abserr = torch.abs(x-x_)
    relerr = abserr/(torch.abs(x)+1e-8)
    print("norm abs error", torch.norm(abserr))
    print("max abs error", torch.max(abserr))
    print("norm rel error", torch.norm(relerr))
    print("max rel error", torch.max(relerr))

def test_toeplitz():
    from model.toeplitz import construct_toeplitz
    def summarize(name, x, x_, showdiff=False):
        print(name, "stats")
        if showdiff:
            print(x-x_)
        abserr = torch.abs(x-x_)
        relerr = abserr/(torch.abs(x)+1e-8)
        print("  norm abs error", torch.norm(abserr))
        print("  max abs error", torch.max(abserr))
        print("  norm rel error", torch.norm(relerr))
        print("  max rel error", torch.max(relerr))

    print("Testing Toeplitz\n====================")
    L = 512
    N = L // 2
    B = 100
    A, u = generate_data(L, N, B)

    A = A[..., 0]
    A = construct_toeplitz(A)

    # print("SHAPES", A.shape, u.shape)

    # Static A
    x = unroll(A, u)
    x_ = variable_unroll(A, u, variable=False)
    summarize("nonvariable matrix original", x, x_, showdiff=False)
    x_ = variable_unroll_matrix(A, u, variable=False)
    summarize("nonvariable matrix general", x, x_, showdiff=False)
    x_ = variable_unroll_toeplitz(A[..., 0], u, variable=False)
    summarize("nonvariable toeplitz", x, x_, showdiff=False)

    # Sequential
    A = A.repeat((L, 1, 1))
    for _ in range(1):
        x_ = variable_unroll_sequential(A, u)
        summarize("variable unroll sequential", x, x_, showdiff=False)
        x_ = variable_unroll_matrix_sequential(A, u)
        summarize("variable matrix sequential", x, x_, showdiff=False)
        x_ = variable_unroll_toeplitz_sequential(A[..., 0], u, pad=True)
        summarize("variable toeplitz sequential", x, x_, showdiff=False)

    # Parallel
    for _ in range(1):
        x_ = variable_unroll(A, u)
        summarize("variable matrix original", x, x_, showdiff=False)
        x_ = variable_unroll_matrix(A, u)
        summarize("variable matrix general", x, x_, showdiff=False)
        x_ = variable_unroll_toeplitz(A[..., 0], u, pad=True, recurse_limit=8)
        summarize("variable toeplitz", x, x_, showdiff=False)

def test_speed(variable=False, it=1):
    print("Testing Speed\n====================")
    N = 256
    L = 1024
    B = 100
    A, u = generate_data(L, N, B)
    As = A.repeat((L, 1, 1))

    u.requires_grad=True
    As.requires_grad=True
    for _ in range(it):
        x = unroll(A, u)
        x = torch.sum(x)
        x.backward()

        x = parallel_unroll_recursive(A, u)
        x = torch.sum(x)
        x.backward()

        # parallel_unroll_recursive_br(A, u)
        # parallel_unroll_iterative(A, u)

    for _ in range(it):
        if variable:
            x = variable_unroll_sequential(As, u, variable=True, recurse_limit=16)
            x = torch.sum(x)
            x.backward()
            x = variable_unroll(As, u, variable=True, recurse_limit=16)
            x = torch.sum(x)
            x.backward()
        else:
            variable_unroll_sequential(A, u, variable=False, recurse_limit=16)
            variable_unroll(A, u, variable=False, recurse_limit=16)

# TODO refactor using benchmark util

if __name__ == '__main__':
    # test_correctness()
    test_stability()
    # test_toeplitz()
    # test_speed(variable=True, it=100)
