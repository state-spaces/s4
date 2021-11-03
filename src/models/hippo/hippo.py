""" Definitions of A and B matrices for various HiPPO operators. """

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import special as ss
from einops import rearrange
from opt_einsum import contract

# TODO take in 'torch' option to return torch instead of numpy, which converts the shape of B from (N, 1) to (N)
# TODO remove tlagt
def transition(measure, N, **measure_args):
    """ A, B transition matrices for different measures

    measure: the type of measure
      legt - Legendre (translated)
      legs - Legendre (scaled)
      glagt - generalized Laguerre (translated)
      lagt, tlagt - previous versions of (tilted) Laguerre with slightly different normalization
    """
    # Laguerre (translated)
    if measure == 'lagt':
        b = measure_args.get('beta', 1.0)
        A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    elif measure == 'tlagt':
        # beta = 1 corresponds to no tilt
        b = measure_args.get('beta', 1.0)
        A = (1.-b)/2 * np.eye(N) - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    # Generalized Laguerre
    # alpha 0, beta small is most stable (limits to the 'lagt' measure)
    # alpha 0, beta 1 has transition matrix A = [lower triangular 1]
    elif measure == 'glagt':
        alpha = measure_args.get('alpha', 0.0)
        beta = measure_args.get('beta', 0.01)
        A = -np.eye(N) * (1 + beta) / 2 - np.tril(np.ones((N, N)), -1)
        B = ss.binom(alpha + np.arange(N), np.arange(N))[:, None]

        L = np.exp(.5 * (ss.gammaln(np.arange(N)+alpha+1) - ss.gammaln(np.arange(N)+1)))
        A = (1./L[:, None]) * A * L[None, :]
        B = (1./L[:, None]) * B * np.exp(-.5 * ss.gammaln(1-alpha)) * beta**((1-alpha)/2)
    # Legendre (translated)
    elif measure == 'legt':
        Q = np.arange(N, dtype=np.float64)
        R = (2*Q + 1) ** .5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.)**(i-j), 1) * R[None, :]
        B = R[:, None]
        A = -A
    # LMU: equivalent to LegT up to normalization
    elif measure == 'lmu':
        Q = np.arange(N, dtype=np.float64)
        R = (2*Q + 1)[:, None] # / theta
        j, i = np.meshgrid(Q, Q)
        A = np.where(i < j, -1, (-1.)**(i-j+1)) * R
        B = (-1.)**Q[:, None] * R
    # Legendre (scaled)
    elif measure == 'legs':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy() # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
    else:
        raise NotImplementedError

    return A, B

def rank_correction(measure, N, rank=1, dtype=torch.float):
    """ Return low-rank matrix L such that A + L is normal """

    if measure == 'legs':
        assert rank >= 1
        p = torch.sqrt(.5+torch.arange(N, dtype=dtype)).unsqueeze(0) # (1 N)
    elif measure == 'legt':
        assert rank >= 2
        p = torch.sqrt(1+2*torch.arange(N, dtype=dtype)) # (N)
        p0 = p.clone()
        p0[0::2] = 0.
        p1 = p.clone()
        p1[1::2] = 0.
        p = torch.stack([p0, p1], dim=0) # (2 N)
    elif measure == 'lagt':
        assert rank >= 1
        p = .5**.5 * torch.ones(1, N, dtype=dtype)
    else: raise NotImplementedError

    d = p.size(0)
    if rank > d:
        p = torch.stack([p, torch.zeros(N, dtype=dtype).repeat(rank-d, d)], dim=0) # (rank N)
    return p

def nplr(measure, N, rank=1, dtype=torch.float):
    """ Return w, p, q, V, B such that
    (w - p q^*, B) is unitarily equivalent to the original HiPPO A, B by the matrix V
    i.e. A = V[w - p q^*]V^*, B = V B
    """
    A, B = transition(measure, N)
    A = torch.as_tensor(A, dtype=dtype) # (N, N)
    B = torch.as_tensor(B, dtype=dtype)[:, 0] # (N,)

    p = rank_correction(measure, N, rank=rank, dtype=dtype)
    Ap = A + torch.sum(p.unsqueeze(-2)*p.unsqueeze(-1), dim=-3)
    w, V = torch.linalg.eig(Ap) # (..., N) (..., N, N)
    # V w V^{-1} = A

    # Only keep one of the conjugate pairs
    w = w[..., 0::2].contiguous()
    V = V[..., 0::2].contiguous()

    V_inv = V.conj().transpose(-1, -2)

    B = contract('ij, j -> i', V_inv, B.to(V)) # V^* B
    p = contract('ij, ...j -> ...i', V_inv, p.to(V)) # V^* p


    return w, p, p, B, V

def test_nplr():
    N = 4
    measure = 'legs'
    w, p, q, B, V = nplr(measure, N)
    w = torch.cat([w, w.conj()], dim=-1)
    V = torch.cat([V, V.conj()], dim=-1)
    B = torch.cat([B, B.conj()], dim=-1)
    p = torch.cat([p, p.conj()], dim=-1)
    q = torch.cat([q, q.conj()], dim=-1)
    A = torch.diag_embed(w) - contract('... r p, ... r q -> ... p q', p, q.conj())

    A = contract('ij, jk, kl -> ... il', V, A, V.conj().transpose(-1,-2)) # Ap^{-1} = V @ w^{-1} @ V^T
    B = contract('ij, ... j -> ... i', V, B)
    print(A.real)
    print(B.real)

if __name__ == '__main__':
    from benchmark import utils

    device = 'cuda' # 'cpu'
    device = torch.device(device)

    # benchmark_krylov(measure='legs', rank=1)
    test_nplr()
