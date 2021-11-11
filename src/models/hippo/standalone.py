""" Standalone implementation of HiPPO-LegS and HiPPO-LegT operators.

Contains experiments for the function reconstruction experiment in original HiPPO paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import linalg as la
from scipy import special as ss
import nengo

import src.models.functional.unroll as unroll
from src.models.hippo.hippo import transition


"""
The HiPPO_LegT and HiPPO_LegS modules satisfy the HiPPO interface:

The forward() method takes an input sequence f of length L to an output sequence c of shape (L, N) where N is the order of the HiPPO operator.
c[k] can be thought of as representing all of f[:k] via coefficients of a polynomial approximation.

The reconstruct() method takes the coefficients and turns each coefficient into a reconstruction of the original input.
Note that each coefficient c[k] turns into an approximation of the entire input f, so this reconstruction has shape (L, L),
and the last element of this reconstruction (which has shape (L,)) is the most accurate reconstruction of the original input.

Both of these two methods construct approximations according to different measures, defined in the HiPPO paper.
The first one is the "Translated Legendre" (which is up to scaling equal to the LMU matrix),
and the second one is the "Scaled Legendre".
Each method comprises an exact recurrence c_k = A_k c_{k-1} + B_k f_k, and an exact reconstruction formula based on the corresponding polynomial family.
"""

class HiPPO_LegT(nn.Module):
    def __init__(self, N, dt=1.0, discretization='bilinear'):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        """
        super().__init__()
        self.N = N
        A, B = transition('lmu', N)
        C = np.ones((1, N))
        D = np.zeros((1,))
        # dt, discretization options
        A, B, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)

        B = B.squeeze(-1)

        self.register_buffer('A', torch.Tensor(A)) # (N, N)
        self.register_buffer('B', torch.Tensor(B)) # (N,)

        # vals = np.linspace(0.0, 1.0, 1./dt)
        vals = np.arange(0.0, 1.0, dt)
        self.eval_matrix = torch.Tensor(ss.eval_legendre(np.arange(N)[:, None], 1 - 2 * vals).T)

    def forward(self, inputs):
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """

        inputs = inputs.unsqueeze(-1)
        u = inputs * self.B # (length, ..., N)

        c = torch.zeros(u.shape[1:])
        cs = []
        for f in inputs:
            c = F.linear(c, self.A) + self.B * f
            cs.append(c)
        return torch.stack(cs, dim=0)

    def reconstruct(self, c):
        return (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)



class HiPPO_LegS(nn.Module):
    """ Vanilla HiPPO-LegS model (scale invariant instead of time invariant) """
    def __init__(self, N, max_length=1024, measure='legs', discretization='bilinear'):
        """
        max_length: maximum sequence length
        """
        super().__init__()
        self.N = N
        A, B = transition(measure, N)
        B = B.squeeze(-1)
        A_stacked = np.empty((max_length, N, N), dtype=A.dtype)
        B_stacked = np.empty((max_length, N), dtype=B.dtype)
        for t in range(1, max_length + 1):
            At = A / t
            Bt = B / t
            if discretization == 'forward':
                A_stacked[t - 1] = np.eye(N) + At
                B_stacked[t - 1] = Bt
            elif discretization == 'backward':
                A_stacked[t - 1] = la.solve_triangular(np.eye(N) - At, np.eye(N), lower=True)
                B_stacked[t - 1] = la.solve_triangular(np.eye(N) - At, Bt, lower=True)
            elif discretization == 'bilinear':
                A_stacked[t - 1] = la.solve_triangular(np.eye(N) - At / 2, np.eye(N) + At / 2, lower=True)
                B_stacked[t - 1] = la.solve_triangular(np.eye(N) - At / 2, Bt, lower=True)
            else: # ZOH
                A_stacked[t - 1] = la.expm(A * (math.log(t + 1) - math.log(t)))
                B_stacked[t - 1] = la.solve_triangular(A, A_stacked[t - 1] @ B - B, lower=True)
        self.A_stacked = torch.Tensor(A_stacked) # (max_length, N, N)
        self.B_stacked = torch.Tensor(B_stacked) # (max_length, N)
        # print("B_stacked shape", B_stacked.shape)

        vals = np.linspace(0.0, 1.0, max_length)
        self.eval_matrix = torch.Tensor((B[:, None] * ss.eval_legendre(np.arange(N)[:, None], 2 * vals - 1)).T)

    def forward(self, inputs, fast=False):
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """

        L = inputs.shape[0]

        inputs = inputs.unsqueeze(-1)
        u = torch.transpose(inputs, 0, -2)
        u = u * self.B_stacked[:L]
        u = torch.transpose(u, 0, -2) # (length, ..., N)

        if fast:
            result = unroll.variable_unroll_matrix(self.A_stacked[:L], u)
        else:
            result = unroll.variable_unroll_matrix_sequential(self.A_stacked[:L], u)
        return result

    def reconstruct(self, c):
        a = self.eval_matrix @ c.unsqueeze(-1)
        return a.squeeze(-1)


class FunctionApprox(data.TensorDataset):

    def __init__(self, length, dt, nbatches, freq=10.0, seed=0):
        rng = np.random.RandomState(seed=seed)
        process = nengo.processes.WhiteSignal(length * dt, high=freq, y0=0)
        X = np.empty((nbatches, length, 1))
        for i in range(nbatches):
            X[i, :] = process.run_steps(length, dt=dt, rng=rng)
            # X[i, :] /= np.max(np.abs(X[i, :]))
        X = torch.Tensor(X)
        super().__init__(X, X)


def test():
    N = 256
    L = 128
    hippo = HiPPO_LegT(N, dt=1./L)

    x = torch.randn(L, 1)

    y = hippo(x)
    print(y.shape)
    z = hippo.reconstruct(y)
    print(z.shape)

    # mse = torch.mean((z[-1,0,:L].flip(-1) - x.squeeze(-1))**2)
    mse = torch.mean((z[-1,0,:L] - x.squeeze(-1))**2)
    print(mse)

    # print(y.shape)
    hippo_legs = HiPPO_LegS(N, max_length=L)
    y = hippo_legs(x)
    # print(y.shape)
    z = hippo_legs(x, fast=True)
    print(hippo_legs.reconstruct(z).shape)
    # print(y-z)


def plot():
    T = 10000
    dt = 1e-3
    N = 256
    nbatches = 10
    train = FunctionApprox(T, dt, nbatches, freq=1.0, seed=0)
    test = FunctionApprox(T, dt, nbatches, freq=1.0, seed=1)
    test_loader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False)
    it = iter(test_loader)
    f, _ = next(it)
    f, _ = next(it)
    f = f.squeeze(0).squeeze(-1)

    legt = HiPPO_LegT(N, 1./T)
    f_legt = legt.reconstruct(legt(f))[-1]
    legs = HiPPO_LegS(N, T)
    f_legs = legs.reconstruct(legs(f))[-1]
    print(F.mse_loss(f, f_legt))
    print(F.mse_loss(f, f_legs))

    vals = np.linspace(0.0, 1.0, T)
    plt.figure(figsize=(6, 2))
    plt.plot(vals, f+0.1, 'k', linewidth=1.0)
    plt.plot(vals[:T//1], f_legt[:T//1])
    plt.plot(vals[:T//1], f_legs[:T//1])
    plt.xlabel('Time (normalized)', labelpad=-10)
    plt.xticks([0, 1])
    plt.legend(['f', 'legt', 'legs'])
    plt.savefig(f'function_approx_whitenoise.pdf', bbox_inches='tight')
    # plt.show()
    plt.close()


if __name__ == '__main__':
    plot()
