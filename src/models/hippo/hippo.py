""" Definitions of A and B matrices for various HiPPO operators. """

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import special as ss
from einops import rearrange, repeat
from opt_einsum import contract

def embed_c2r(A):
    A = rearrange(A, '... m n -> ... m () n ()')
    A = np.pad(A, ((0, 0), (0, 1), (0, 0), (0, 1))) + \
        np.pad(A, ((0, 0), (1, 0), (0, 0), (1,0)))
    return rearrange(A, 'm x n y -> (m x) (n y)')

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

        # Halve again for timescale correctness
        # A, B = A/2, B/2
        A *= 0.5
        B *= 0.5
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
    elif measure == 'legsd':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy() # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
        A += .5 * B*B[None, :, 0]
        B = B / 2.0
    elif measure == 'fourier_old':
        freqs = np.arange(N//2)
        d = np.stack([freqs, np.zeros(N//2)], axis=-1).reshape(-1)[:-1]
        A = 2*np.pi*(np.diag(d, 1) - np.diag(d, -1))
        A = A - embed_c2r(np.ones((N//2, N//2)))
        B = embed_c2r(np.ones((N//2, 1)))[..., :1]
    elif measure == 'fourier_diag':
        freqs = np.arange(N//2)
        d = np.stack([freqs, np.zeros(N//2)], axis=-1).reshape(-1)[:-1]
        A = 2*np.pi*(-np.diag(d, 1) + np.diag(d, -1))
        # A = A - 0.5*embed_c2r(np.ones((N//2, N//2)))
        A = A - .5 * np.eye(N)
        B = embed_c2r(np.ones((N//2, 1)))[..., :1]
    elif measure == 'fourier':
        freqs = np.arange(N//2)
        d = np.stack([np.zeros(N//2), freqs], axis=-1).reshape(-1)[1:]
        A = np.pi*(-np.diag(d, 1) + np.diag(d, -1))
        B = np.zeros(N)
        B[0::2] = 2**.5
        B[0] = 1

        # Subtract off rank correction - this corresponds to the other endpoint u(t-1) in this case
        A = A - B[:, None] * B[None, :]
        B = B[:, None]
    elif measure == 'fourier_decay':
        freqs = np.arange(N//2)
        d = np.stack([np.zeros(N//2), freqs], axis=-1).reshape(-1)[1:]
        A = np.pi*(-np.diag(d, 1) + np.diag(d, -1))
        B = np.zeros(N)
        B[0::2] = 2**.5
        B[0] = 1

        # Subtract off rank correction - this corresponds to the other endpoint u(t-1) in this case
        A = A - .5 * B[:, None] * B[None, :]
        B = .5 * B[:, None]
    elif measure == 'fourier2': # Double everything: orthonormal on [0, 1]
        freqs = 2*np.arange(N//2)
        d = np.stack([np.zeros(N//2), freqs], axis=-1).reshape(-1)[1:]
        A = np.pi*(-np.diag(d, 1) + np.diag(d, -1))
        B = np.zeros(N)
        B[0::2] = 2**.5
        B[0] = 1

        # Subtract off rank correction - this corresponds to the other endpoint u(t-1) in this case
        A = A - B[:, None] * B[None, :] * 2
        B = B[:, None] * 2

    elif measure == 'random':
        A = np.random.randn(N, N) / N
        B = np.random.randn(N, 1)
    elif measure == 'diagonal':
        A = -np.diag(np.exp(np.random.randn(N)))
        B = np.random.randn(N, 1)
    else:
        raise NotImplementedError

    return A, B

def rank_correction(measure, N, rank=1, dtype=torch.float):
    """ Return low-rank matrix L such that A + L is normal """

    if measure == 'legs':
        assert rank >= 1
        P = torch.sqrt(.5+torch.arange(N, dtype=dtype)).unsqueeze(0) # (1 N)
    elif measure == 'legt':
        assert rank >= 2
        P = torch.sqrt(1+2*torch.arange(N, dtype=dtype)) # (N)
        P0 = P.clone()
        P0[0::2] = 0.
        P1 = P.clone()
        P1[1::2] = 0.
        P = torch.stack([P0, P1], dim=0) # (2 N)
        P *= 2**(-0.5) # Halve the rank correct just like the original matrix was halved
    elif measure == 'lagt':
        assert rank >= 1
        P = .5**.5 * torch.ones(1, N, dtype=dtype)
    elif measure == 'fourier_old':
        P = torch.ones(N, dtype=dtype) # (N)
        P0 = P.clone()
        P0[0::2] = 0.
        P1 = P.clone()
        P1[1::2] = 0.
        P = torch.stack([P0, P1], dim=0) # (2 N)
        P = torch.zeros(1, N, dtype=dtype)
    elif measure == 'fourier':
        P = torch.zeros(N)
        P[0::2] = 2**.5
        P[0] = 1
        P = P.unsqueeze(0)
    elif measure == 'fourier_decay':
        P = torch.zeros(N)
        P[0::2] = 2**.5
        P[0] = 1
        P = P.unsqueeze(0)
        P = P / 2**.5
    elif measure == 'fourier2':
        P = torch.zeros(N)
        P[0::2] = 2**.5
        P[0] = 1
        P = 2**.5 * P.unsqueeze(0)
    elif measure in ['fourier_diag', 'legsd']:
        P = torch.zeros(1, N, dtype=dtype)
    else: raise NotImplementedError

    d = P.size(0)
    if rank > d:
        P = torch.cat([P, torch.zeros(rank-d, N, dtype=dtype)], dim=0) # (rank N)
    return P

def initial_C(measure, N, dtype=torch.float):
    """ Return C that captures the other endpoint in the HiPPO approximation """

    if measure == 'legt':
        C = (torch.arange(N, dtype=dtype)*2+1)**.5 * (-1)**torch.arange(N)
    elif measure == 'fourier_old':
        C = torch.ones(N, dtype=dtype) # (N)
    elif measure == 'fourier':
        C = torch.zeros(N)
        C[0::2] = 2**.5
        C[0] = 1
    else:
        C = torch.zeros(N, dtype=dtype) # (N)

    return C


def nplr(measure, N, rank=1, dtype=torch.float):
    """ Return w, p, q, V, B such that
    (w - p q^*, B) is unitarily equivalent to the original HiPPO A, B by the matrix V
    i.e. A = V[w - p q^*]V^*, B = V B
    """
    assert dtype == torch.float or torch.cfloat

    A, B = transition(measure, N)
    A = torch.as_tensor(A, dtype=dtype) # (N, N)
    B = torch.as_tensor(B, dtype=dtype)[:, 0] # (N,)

    P = rank_correction(measure, N, rank=rank, dtype=dtype) # (r N)
    AP = A + torch.sum(P.unsqueeze(-2)*P.unsqueeze(-1), dim=-3)
    w, V = torch.linalg.eig(AP) # (..., N) (..., N, N)
    # V w V^{-1} = A

    # print("check", V @ torch.diag_embed(w) @ V.conj().transpose(-1, -2))


    # We require AP to be nearly skew-symmetric
    _A = AP + AP.transpose(-1, -2)
    if (err := torch.sum((_A - _A[0,0]*torch.eye(N))**2) / N) > 1e-5: # if not torch.allclose(_A - _A[0,0]*torch.eye(N), torch.zeros(N, N), atol=1e-5):
        print("WARNING: HiPPO matrix not skew symmetric", err)


    # Only keep half of each conjugate pair
    # w = w[..., 0::2].contiguous()
    # V = V[..., 0::2].contiguous()
    _, idx = torch.sort(w.imag)
    w_sorted = w[idx]
    V_sorted = V[:, idx]

    # There is an edge case when eigenvalues can be 0, which requires some machinery to handle
    # We use a huge hack here: Assume only one pair is 0, and that it is the first row/column of A (only happens in Fourier case)
    V = V_sorted[:, :N//2]
    w = w_sorted[:N//2]
    assert w[-2].abs() > 1e-4, "Only 1 zero eigenvalue allowed in diagonal part of A"
    if w[-1].abs() < 1e-4:
        V[:, -1] = 0.
        V[0, -1] = 2**-0.5
        V[1, -1] = 2**-0.5 * 1j

    _AP = V @ torch.diag_embed(w) @ V.conj().transpose(-1, -2)
    # assert torch.allclose(2*_AP.real, AP, atol=1e-5)
    if ((err := torch.sum((2*_AP.real-AP)**2)/N) > 1e-5):
        print("Warning: Diagonalization of A matrix not numerically precise - error", err)
    # print("check", V @ torch.diag_embed(w) @ V.conj().transpose(-1, -2))


    # # Override eigenvectors for 0 eigenvalues, to make them conjugate pairs
    # breakpoint()
    # rotate = torch.tensor([[1, 1], [1j, -1j]]) / 2**.5
    # # rotate = torch.tensor([[1, -1j], [1, 1j]]) / 2**.5
    # V_rot = (V.view(N, N//2, 2) @ rotate).view(N, N) # rotate every pair of eigenvectors
    # V = torch.where(w.repeat(N, 1) == 0, V_rot, V)


    V_inv = V.conj().transpose(-1, -2)

    C = initial_C(measure, N, dtype=dtype)
    B = contract('ij, j -> i', V_inv, B.to(V)) # V^* B
    C = contract('ij, j -> i', V_inv, C.to(V)) # V^* C
    P = contract('ij, ...j -> ...i', V_inv, P.to(V)) # V^* P


    return w, P, B, C, V

def random_dplr(N, rank=1, H=1, dtype=torch.float, real_scale=1.0, imag_scale=1.0, scaling='inverse', random_real=False, random_imag=False, normalize=True):
    assert dtype == torch.float or torch.double
    # batch_shape = (H, N//2) if H is not None else (N//2,)
    dtype = torch.cfloat if dtype == torch.float else torch.cdouble
    # w = -torch.exp(torch.randn(N//2)) + 1j*torch.randn(N//2)
    # w = -torch.exp(torch.randn(N//2)) + 1j*2*torch.tensor(np.pi)*N*torch.rand(N//2) # try larger eigenvalue spread

    pi = torch.tensor(np.pi)
    if random_real:
        real_part = torch.rand(H, N//2)
    else:
        real_part = .5 * torch.ones(H, N//2)
    if random_imag:
        imag_part = N//2 * torch.rand(H, N//2)
    else:
        imag_part = repeat(torch.arange(N//2), 'n -> h n', h=H)

    real_part = real_scale * real_part
    if scaling == 'random':
        imag_part = torch.randn(H, N//2)
    elif scaling == 'linear':
        imag_part = pi * imag_part
    elif scaling == 'inverse': # Based on asymptotics of the default HiPPO matrix
        # intercept = torch.log(N//2)/torch.log(2) * 2./3.
        # log_imag_part = intercept + 2. * torch.atanh((1+imag_part*2)/N*2-1)
        # imag_part = torch.exp(log_imag_part)
        # intercept = torch.log(N//2) - .5
        # imag_part = torch.exp(2. * torch.atanh((1+imag_part*2)/N*2-1))
        imag_part = 1/pi * N * (N/(1+2*imag_part)-1)
    elif scaling == 'inverse2': # Based on asymptotics of the default HiPPO matrix
        # intercept = torch.log(N//2)/torch.log(2) * 2./3.
        # log_imag_part = intercept + 2. * torch.atanh((1+imag_part*2)/N*2-1)
        # imag_part = torch.exp(log_imag_part)
        # intercept = torch.log(N//2) - .5
        # imag_part = torch.exp(2. * torch.atanh((1+imag_part*2)/N*2-1))
        imag_part = 1/pi * N * (N/(1+imag_part)-1)
    elif scaling == 'quadratic':
        imag_part = 1/pi * (1+2*imag_part)**2
    else: raise NotImplementedError
    imag_part = imag_scale * imag_part
    w = -real_part + 1j * imag_part




    # w = -torch.rand(N//2) + 1j*2*torch.tensor(np.pi)*N*torch.rand(N//2) # try larger eigenvalue spread
    # w = -1 + torch.arange(N//2) * 1j * 2 * torch.tensor(np.pi)
    P = torch.randn(rank, H, N//2, dtype=dtype)
    # p = torch.zeros(rank, N//2, dtype=dtype)
    B = torch.randn(H, N//2, dtype=dtype)
    # B = torch.ones(N//2, dtype=dtype)
    C = torch.randn(H, N//2, dtype=dtype)
    V = torch.eye(N, dtype=dtype)[..., :N//2] # Only used in testing

    if normalize: # TODO can normalize the full matrix with rank correction too
        norm = -B/w # (H, N) # Result if you integrate the kernel with constant 1 function
        zeta = 2*torch.sum(torch.abs(norm)**2, dim=-1, keepdim=True) # Variance with a random C vector
        B = B / zeta**.5

    return w, P, B, C, V

def test_nplr():
    N = 4
    measure = 'fourier_decay'
    w, P, B, C, V = nplr(measure, N, rank=1)
    w = torch.cat([w, w.conj()], dim=-1)
    V = torch.cat([V, V.conj()], dim=-1)
    B = torch.cat([B, B.conj()], dim=-1)
    P = torch.cat([P, P.conj()], dim=-1)
    Q = P
    # q = torch.cat([q, q.conj()], dim=-1)
    A = torch.diag_embed(w) - contract('... r p, ... r q -> ... p q', P, Q.conj())

    A = contract('ij, jk, kl -> ... il', V, A, V.conj().transpose(-1,-2)) # Ap^{-1} = V @ w^{-1} @ V^T
    B = contract('ij, ... j -> ... i', V, B)
    print(A.real)
    print(B.real)

if __name__ == '__main__':
    from benchmark import utils

    torch.set_printoptions(precision=3)

    device = 'cuda' # 'cpu'
    device = torch.device(device)

    # benchmark_krylov(measure='legs', rank=1)
    test_nplr()
