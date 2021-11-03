""" Core S3 convolution kernel implementing the 'normal plus low-rank' algorithm.

The main module is SSKernelNPLR, which stores parameters A, B, C, dt, and calling it creates the SSM convolution kernel bar{K}.

A much simpler version SSKernelSlow is included for illustration purposes: it has the same output, but uses the naive algorithm which is much slower. This module is meant for testing and exposition, to understand what the State Space Kernel actually does.

HiPPOSSKernel specializes the SSKernels to specific instantiations of HiPPO matrices.
"""

if __name__ == "__main__":
    import sys
    import pathlib

    p = pathlib.Path().absolute()
    print("Adding path: ", p)
    sys.path.append(str(p))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.fft
from einops import rearrange, repeat
from opt_einsum import contract
from omegaconf import DictConfig

import src.models.hippo.hippo as hippo
from src.models.functional.krylov import krylov, power

import src.utils.train

log = src.utils.train.get_logger(__name__)

try:
    from extensions.cauchy.cauchy import cauchy_mult
    has_cauchy_extension = True
except:
    log.warn(
        "CUDA extension for cauchy multiplication not found. Install by going to extensions/cauchy/ and running `python setup.py install`. This should speed up end-to-end training by 10-50%"
    )
    has_cauchy_extension = False

try:
    import src.models.functional.cauchy as cauchy
except ImportError:
    if not has_cauchy_extension:
        log.error(
            "Install at least one of pykeops or cauchy_mult."
        )



_isnan = lambda x: torch.isnan(x).any()
_isinf = lambda x: torch.isinf(x).any()

_conj = lambda x: torch.cat([x, x.conj()], dim=-1)


class OptimModule(nn.Module):
    """ Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters """

    def register(self, name, tensor, trainable=0, lr=None, wd=None, repeat=1):
        """Utility method: register a tensor as a buffer or trainable parameter"""

        if trainable == 0:
            self.register_buffer(name, tensor)
        elif trainable == 1:
            self.register_parameter(name, nn.Parameter(tensor))
        elif trainable == 2:
            tensor = tensor.repeat(repeat, *(1,) * len(tensor.shape))
            self.register_parameter(name, nn.Parameter(tensor))
        else:
            raise NotImplementedError

        optim = {}
        if trainable and lr is not None:
            optim["lr"] = lr
            # setattr(getattr(self, name), '_lr', lr)
        if trainable and wd is not None:
            optim["weight_decay"] = wd
            # setattr(getattr(self, name), '_wd', wd)
        if len(optim) > 0:
            setattr(getattr(self, name), "_optim", optim)


class SSKernelNPLR(OptimModule):
    """Stores a representation of and computes the SSKernel function K_L(A^dt, B^dt, C) corresponding to a discretized state space, where A is Normal + Low Rank (NPLR)

    The class name stands for 'State-Space SSKernel for Normal Plus Low-Rank'.
    The parameters of this function are as follows.

    A: (... N N) the state matrix
    B: (... N) input matrix
    C: (... N) output matrix
    dt: (...) timescales / discretization step size
    p, q: (... P N) low-rank correction to A, such that Ap=A+pq^T is a normal matrix

    The forward pass of this Module returns:
    (... L) that represents represents FFT SSKernel_L(A^dt, B^dt, C)

    """

    @torch.no_grad()
    def _process_C(self, L, double_length=False):
        C = torch.view_as_complex(self.C)
        self._setup(setup_C=False)
        dA = self.dA
        dA_L = power(L, dA)
        # I = torch.eye(dA.size(-1)).to(dA)
        N = C.size(-1)
        # Multiply C by I - dA_L
        C_ = C[..., 0, :]
        C_ = torch.cat([C_, C_.conj()], dim=-1)
        prod = contract("... m n, ... n -> ... m", dA_L.conj().transpose(-1, -2), C_)
        if double_length:  # Multiply by I + dA_L instead
            C_ = C_ + prod
        else:
            C_ = C_ - prod
        C_ = C_[..., :N]
        self.C[..., 0, :, :].copy_(torch.view_as_real(C_))

    def _nodes(self, L, dtype, device):
        # Cache FFT nodes and their "unprocessed" them with the bilinear transform
        # nodes = torch.tensor(np.exp(-2j * np.pi / (L)), dtype=torch.cfloat, device=Ap.device) # \omega_{2L}
        nodes = torch.tensor(
            np.exp(-2j * np.pi / (L)), dtype=dtype, device=device
        )  # \omega_{2L}
        nodes = nodes ** torch.arange(0, L // 2 + 1, device=device)
        z = 2 * (1 - nodes) / (1 + nodes)
        return nodes, z

    def __init__(
        self,
        L,
        w,
        p,
        q,
        B,
        C,
        log_dt,
        trainable=None,
        lr=None,
        setup_C=False,
        keops=False,
    ):
        """Optim arguments into a representation. This occurs after init so that these operations can occur after moving model to device

        L: Maximum length; this module computes SSKernel function of length L
        A: (..., N, N) represented by diag(w) - pq^*
        B: (..., N)
        C: (..., N)
        dt: (...)
        p: (..., N) low-rank correction to A
        q: (..., N)
        """

        super().__init__()
        self.keops = keops

        # Rank of low-rank correction
        assert p.shape[-2] == q.shape[-2]
        self.rank = p.shape[-2]
        self.L = L

        # Augment B and C with low rank correction
        B = B.unsqueeze(-2)  # (..., 1, N)
        C = C.unsqueeze(-2)  # (..., 1, N)
        if len(B.shape) > len(p.shape):
            p = p.repeat(B.shape[:-2] + (1, 1))
        B = torch.cat([B, p], dim=-2)
        if len(C.shape) > len(q.shape):
            q = q.repeat(C.shape[:-2] + (1, 1))
        C = torch.cat([C, q], dim=-2)

        if L is not None:
            nodes, z = self._nodes(L, dtype=w.dtype, device=w.device)
            self.register_buffer("nodes", torch.view_as_real(nodes))
            self.register_buffer("z", torch.view_as_real(z))

        # Register parameters
        if trainable is None:
            trainable = DictConfig({"A": 0, "B": 0, "C": 0, "dt": 0})
        if lr is None:
            lr = DictConfig({"A": None, "B": None, "C": None, "dt": None})
        repeat = C.size(0)
        self.register("log_dt", log_dt, trainable.dt, lr.dt, 0.0)
        self.register("w", torch.view_as_real(w), trainable.A, lr.A, 0.0, repeat=repeat)
        self.register("B", torch.view_as_real(B), trainable.B, lr.B, 0.0, repeat=repeat)
        self.register("C", torch.view_as_real(C), trainable.C, lr.C)

        if setup_C:
            self._process_C(L)

    def forward(self, state=None, rate=1.0, L=None):
        """
        state: (..., s, N) extra tensor that augments B
        rate: sampling rate factor
        """
        # if L is not None: raise NotImplementedError

        # TODO: handle potential length doubling logic so that max_len doesn't need to be passed in
        while rate == 1.0 and L > self.L:
            log.info(f"S3: Doubling length from L = {self.L} to {2*self.L}")
            self.double_length()

        if L is None:
            L = self.L
        if rate == 1.0:
            L = self.L
        else:
            rate = self.L / L
        dt = torch.exp(self.log_dt) * rate
        B = torch.view_as_complex(self.B)
        C = torch.view_as_complex(self.C)
        w = torch.view_as_complex(self.w)  # (..., N)
        # z = torch.view_as_complex(self.z) # (..., L)

        # TODO adjust based on rate times normal max length
        if L == self.L:
            nodes = torch.view_as_complex(self.nodes)
            z = torch.view_as_complex(self.z)  # (..., L)
        else:
            nodes, z = self._nodes(L, dtype=w.dtype, device=w.device)

        # Augment B
        if state is not None:  # TODO have not updated
            # Have to "unbilinear" the state to put it into the same "type" as B
            # Compute (I + dt/2 A) @ state
            s = state.transpose(0, 1)  # (H B N)
            p = B[..., 1:, :]  # (... r N)
            q = C[..., 1:, :]  # (... r N)

            # Calculate contract('... s n, ... r n, ... r m -> ... s m', sV, qV.conj(), pV), but take care of conjugate symmetry
            sA = (
                s * w.unsqueeze(-2)
                - (2 + 0j) * (s @ q.conj().transpose(-1, -2)).real @ p
            )
            s = s / dt.unsqueeze(-1).unsqueeze(-1) + sA / 2

            B = torch.cat([s, B], dim=-2)  # (..., 2+s, N)

        # Incorporate dt into A
        w = w * dt.unsqueeze(-1)  # (... N)

        # Incorporate B and C batch dimensions
        v = B.unsqueeze(-3) * C.unsqueeze(-2).conj()  # (..., 2, 2, N)
        w = w[..., None, None, :]  # (..., 1, 1, N)
        z = z[..., None, None, :]  # (..., 1, 1, L)

        # Calculate resolvent at nodes
        if not self.keops and has_cauchy_extension:
            r = cauchy_mult(v, z, w, symmetric=True)
        else:
            r = cauchy.cauchy_conj(v, z, w)
        r = r * dt[..., None, None, None]  # (..., 1+r, 1+r, L)

        # Low-rank Woodbury correction
        if self.rank == 1:
            k_f = r[..., :-1, :-1, :] - r[..., :-1, -1:, :] * r[..., -1:, :-1, :] / (
                1 + r[..., -1:, -1:, :]
            )
        elif self.rank == 2:
            r00 = r[..., : -self.rank, : -self.rank, :]
            r01 = r[..., : -self.rank, -self.rank :, :]
            r10 = r[..., -self.rank :, : -self.rank, :]
            r11 = r[..., -self.rank :, -self.rank :, :]
            det = (1 + r11[..., :1, :1, :]) * (1 + r11[..., 1:, 1:, :]) - r11[
                ..., :1, 1:, :
            ] * r11[..., 1:, :1, :]
            s = (
                r01[..., :, :1, :] * (1 + r11[..., 1:, 1:, :]) * r10[..., :1, :, :]
                + r01[..., :, 1:, :] * (1 + r11[..., :1, :1, :]) * r10[..., 1:, :, :]
                - r01[..., :, :1, :] * (r11[..., :1, 1:, :]) * r10[..., 1:, :, :]
                - r01[..., :, 1:, :] * (r11[..., 1:, :1, :]) * r10[..., :1, :, :]
            )
            s = s / det
            k_f = r00 - s
        else:
            r00 = r[..., : -self.rank, : -self.rank, :]
            r01 = r[..., : -self.rank, -self.rank :, :]
            r10 = r[..., -self.rank :, : -self.rank, :]
            r11 = r[..., -self.rank :, -self.rank :, :]
            r11 = rearrange(r11, "... a b n -> ... n a b")
            r11 = torch.linalg.inv(torch.eye(self.rank, device=r.device) + r11)
            r11 = rearrange(r11, "... n a b -> ... a b n")
            k_f = r00 - torch.einsum(
                "... i j n, ... j k n, ... k l n -> ... i l n", r01, r11, r10
            )

        # Final correction for the bilinear transform
        k_f = k_f * 2 / (1 + nodes)

        k = torch.fft.irfft(k_f)  # (..., 1, 1+s, L)
        if state is not None:
            k_state = k[..., 0, :-1, :]  # (..., s, L)
            k_state = k_state.transpose(0, 1)
            k_B = k[..., 0, -1, :]  # (..., L)
            return k_B.to(torch.float), k_state.to(torch.float)
        else:
            return k.squeeze(-2).squeeze(-2).to(torch.float)

    @torch.no_grad()
    def double_length(self):
        self._process_C(self.L, double_length=True)

        self.L *= 2
        dtype = torch.view_as_complex(self.w).dtype
        nodes, z = self._nodes(self.L, dtype=dtype, device=self.w.device)
        self.register_buffer("nodes", torch.view_as_real(nodes))
        self.register_buffer("z", torch.view_as_real(z))

    @torch.no_grad()
    def _check(self):
        """Check if A, B, C parameters and vanilla SSKernel construction can be recovered"""

        self._setup(setup_C=True)

        K = krylov(self.L, self.dA, self.dB, self.dC.conj())

        diff = K - self.forward()
        print("checking SSKernel construction", torch.sum(diff ** 2))

    def _setup(self, setup_C=True):
        w = _conj(torch.view_as_complex(self.w))
        B = _conj(torch.view_as_complex(self.B))
        C = _conj(torch.view_as_complex(self.C))
        C = C.conj()
        p = B[..., -1, :]
        q = C[..., -1, :]
        B = B[..., 0, :]
        C = C[..., 0, :]
        dt = torch.exp(self.log_dt)
        d = (2.0 / dt.unsqueeze(-1) - w).reciprocal()  # (H, N)
        r = (1 + contract("... n, ... n, ... n -> ...", q, d, p)).reciprocal()
        # A_f = torch.diag_embed(2./dt[:, None] + w) - contract('... n, ... m -> ... n m', p, q)
        # A_b = torch.diag_embed(d) - contract('... p, ... p, ..., ... q, ... q -> ... p q', d, p, r, q, d)
        # dA = A_b @ A_f

        self.step_params = {
            "d": d,
            "r": r.unsqueeze(-1) * d * q,
            # 'r': r,
            "p": p,
            "q": q,
            "B": B,
            "d1": 2.0 / dt.unsqueeze(-1) + w,
        }
        N = d.size(-1)
        H = dt.size(-1)

        state = torch.eye(N, dtype=w.dtype, device=w.device).unsqueeze(-2)
        u = w.new_zeros(H)
        dA = self.step_state_linear(u, state)
        dA = rearrange(dA, "n h m -> h m n")
        self.dA = dA
        u = w.new_ones(H)
        state = w.new_zeros(N // 2)
        dB = self.step_state_linear(u, state)
        dB = _conj(dB)
        self.dB = dB

        if setup_C:
            dA_L = power(self.L, dA)
            I = torch.eye(dA.size(-1)).to(dA)
            dC = torch.linalg.solve(
                I - dA_L.transpose(-1, -2).conj(), C.conj().unsqueeze(-1)
            ).squeeze(-1)
            self.dC = dC

    def step_state_linear(self, u=None, state=None):
        """Version of the step function that has time O(N) instead of O(N^2) per step. Unfortunately, as currently implemented it's about 2x slower because it calls several sequential operations. Perhaps a fused CUDA kernel implementation would be much faster"""
        N = self.step_params["d"].size(-1)
        H = self.log_dt.size(-1)

        if u is None:
            u = torch.zeros(H, dtype=torch.float, device=self.log_dt.device)
        if state is None:
            state = torch.zeros(H, N, dtype=torch.cfloat, device=self.log_dt.device)

        conj = state.size(-1) != N
        step_params = self.step_params.copy()
        if conj:
            assert state.size(-1) == N // 2
            step_params = {k: v[..., : N // 2] for k, v in step_params.items()}
        d1 = step_params["d1"]  # (H N)
        p = step_params["p"]  # (H N)
        q = step_params["q"]  # (H N)
        B = step_params["B"]  # (H N)
        r = step_params["r"]
        d = step_params["d"]  # (H N)
        # dC = self.step_params['dC'] # (H N)
        state = state.to(d1)

        if conj:
            new_state = (
                2 * p * torch.sum(q * state, dim=-1, keepdim=True).real
            )  # conjugated version
        else:
            new_state = contract("... n, ... m, ... m -> ... n", p, q, state)  # (B H N)
        new_state = d1 * state - new_state
        new_state = new_state + 2.0 * B * u.unsqueeze(-1)  # (B H N)
        if conj:
            A_ = (
                2 * p * torch.sum(r * new_state, dim=-1, keepdim=True).real
            )  # conj version
        else:
            A_ = contract("... p, ... q, ... q -> ... p", p, r, new_state)  # (B H N)
        new_state = d * (new_state - A_)

        return new_state

    def step_state(self, u, state):
        state = state.to(self.dA)
        conj = state.size(-1) != self.dA.size(-1)
        if conj:
            state = _conj(state)
        next_state = contract("h m n, b h n -> b h m", self.dA, state) + contract(
            "h n, b h -> b h n", self.dB, u
        )
        if conj:
            next_state = next_state[..., : state.size(-1) // 2]
        return next_state

    def step(self, u, state, linear=False):

        N = self.step_params["d"].size(-1)
        conj = state.size(-1) != N

        if linear:
            new_state = self.step_state_linear(u, state)
        else:
            new_state = self.step_state(u, state)

        if conj:
            assert state.size(-1) == N // 2
            # dC = self.dC[..., 0::2].conj()
            dC = self.dC[..., : N // 2].conj()
            out = 2 * torch.sum(dC * new_state, dim=-1).real  # conj version
        else:
            out = contract("... n, ... n -> ...", self.dC.conj(), new_state)
        return out.to(torch.float), new_state


class SSKernelSlow(OptimModule):
    """Slow version of SSKernel function for illustration and benchmarking.

    - Caches discretized matrices A^(dt), B^(dt)
    - Computes K_L(A^dt, B^dt, C)

    Usage:
    ```
    krylov = SSKernelSlow(L, A, B, C, log_dt)()
    ```
    Result is expected to be equal to SSKernelNPLR(L, A, B, C, log_dt, p, q)() for p, q such that A+pq^T is normal
    """

    def __init__(self, L, A, B, C, log_dt, trainable=None, lr=None):
        super().__init__()
        self.N = A.shape[-1]
        self.L = L
        dA, dB = SSKernelSlow.bilinear(torch.exp(log_dt), A, B)

        # Register parameters
        if trainable is None:
            trainable = DictConfig({"A": 0, "B": 0, "C": 0, "dt": 0})
        if lr is None:
            lr = DictConfig({"A": None, "B": None, "C": None, "dt": None})
        if trainable is not None and lr is not None:
            repeat = C.size(0)
            self.register("log_dt", log_dt, trainable.dt, lr.dt)
            self.register("dA", dA, trainable.A, lr.A, repeat=repeat)
            self.register("dB", dB, 1, lr.B)
            self.register("C", C, trainable.C, lr.C)

    def forward(self, rate=1.0, L=None, state=None):
        if L is None:
            L = self.L
        if rate is None:
            rate = self.L / L  # TODO this class doesn't actually support rates
        k = krylov(L, self.dA, self.dB, self.C.conj())  # (H L)
        if state is not None:
            if state.size(-1) != self.dA.size(-1):
                state = _conj(state)
            state = state.to(self.dA)
            state = contract("... n m, ... m -> ... n", self.dA, state)
            k_state = krylov(L, self.dA, state, self.C.conj())
            return k.to(torch.float), k_state.to(torch.float)
        return k.to(torch.float)

    @classmethod
    def bilinear(cls, dt, A, B=None, separate=False):
        """
        dt: (...) timescales
        A: (... N N)
        B: (... N)
        """
        N = A.shape[-1]
        I = torch.eye(N).to(A)
        A_backwards = I - dt[:, None, None] / 2 * A
        A_forwards = I + dt[:, None, None] / 2 * A

        if B is None:
            dB = None
        else:
            dB = dt[..., None] * torch.linalg.solve(
                A_backwards, B.unsqueeze(-1)
            ).squeeze(
                -1
            )  # (... N)

        if separate:
            A_b = torch.linalg.solve(A_backwards, I)  # (... N N)
            return A_forwards, A_b, dB
        else:
            dA = torch.linalg.solve(A_backwards, A_forwards)  # (... N N)
            return dA, dB

    def _setup(self, setup_C=True):
        if setup_C:
            self.dC = self.C

    def step(self, u, state):
        state = state.to(self.dA)
        if state.size(-1) != self.dA.size(-1):
            state = _conj(state)
        next_state = contract("h m n, b h n -> b h m", self.dA, state) + contract(
            "h n, b h -> b h n", self.dB, u
        )
        y = contract("... n, ... n -> ...", self.dC.conj(), next_state)
        return y.to(torch.float), next_state


class HippoSSKernel(nn.Module):
    """Wrapper around SSKernelNPLR that generates A, B, C, dt according to HiPPO arguments."""

    def __init__(
        self,
        N,
        H,
        L=None,
        measure="legs",
        rank=1,
        dt_min=0.001,
        dt_max=0.1,
        trainable=None,
        lr=None,
        mode="nplr",  # 'slow' for complex naive version, 'real' for real naive version
        length_correction=False,
        precision=1,
        cache=False,
        resample=False,  # if given inputs of different lengths, adjust the sampling rate
        keops=False,
    ):
        super().__init__()
        self.N = N
        self.H = H
        L = L or 1
        self.precision = precision
        dtype = torch.double if self.precision == 2 else torch.float
        self.rate = None if resample else 1.0

        # Set default trainable and lr parameters
        self.trainable = DictConfig(
            {
                "A": 1,
                "B": 2,
                "C": 1,
                "dt": 1,
            }
        )
        if trainable is not None:
            self.trainable.update(trainable)
        self.lr = DictConfig(
            {
                "A": 1e-3,
                "B": 1e-3,
                "C": None,
                "dt": 1e-3,
            }
        )
        if lr is not None:
            self.lr.update(lr)

        # Generate dt
        self.log_dt = torch.rand(self.H, dtype=dtype) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        # Compute the preprocessed representation
        if mode == "real":  # Testing purposes only
            # Generate A, B
            A, B = hippo.transition(measure, N)
            A = torch.as_tensor(A, dtype=dtype)
            B = torch.as_tensor(B, dtype=dtype)[:, 0]

            # Generate C
            C = torch.randn(self.H, self.N, dtype=dtype)

            self.krylov = SSKernelSlow(
                L, A, B, C, self.log_dt, trainable=self.trainable, lr=self.lr
            )
        else:
            # Generate low rank correction p for the measure
            w, p, q, B, _ = hippo.nplr(measure, N, rank, dtype=dtype)
            cdtype = torch.cfloat if dtype == torch.float else torch.cdouble
            C = torch.randn(self.H, self.N // 2, dtype=cdtype)
            if mode == "nplr":
                self.krylov = SSKernelNPLR(
                    L,
                    w,
                    p,
                    q,
                    B,
                    C,
                    self.log_dt,
                    trainable=self.trainable,
                    lr=self.lr,
                    setup_C=length_correction,
                    keops=keops,
                )
            elif mode == "slow":  # Testing only
                A = torch.diag_embed(_conj(w)) - contract(
                    "... r p, ... r q -> ... p q", _conj(p), _conj(q).conj()
                )
                self.krylov = SSKernelSlow(
                    L,
                    A,
                    _conj(B),
                    _conj(C),
                    self.log_dt,
                    trainable=self.trainable,
                    lr=self.lr,
                )

        # Cached tensors
        self.K = None
        self.cache = cache

    def forward(self, state=None, L=None):
        """
        state: (B, H, N)
        """

        if state is not None:
            k, k_state = self.krylov(
                state=state, rate=self.rate, L=L
            )  # (B, H, L) (B, H, N)
            return k, k_state
        else:
            # Calculate K if needed
            if not self.training and self.K is not None and self.K.size(-1) == L:
                k = self.K
            else:
                k = self.krylov(rate=self.rate, L=L).to(torch.float)

            # Store K if needed
            if self.cache and not self.training:
                self.K = k
            else:  # If training, parameter will change after backprop so make sure to recompute on next pass
                self.K = None
            return k

    @torch.no_grad()
    def next_state(self, state, u):
        """
        state: (..., N)
        u: (..., L)

        Returns: (..., N)
        """

        self.krylov._setup()
        dA, dB = self.krylov.dA, self.krylov.dB

        conj = state.size(-1) != dA.size(-1)
        if conj:
            state = _conj(state)

        v = dB.unsqueeze(-1) * u.flip(-1).unsqueeze(-2)  # (..., N, L)
        AL, v = power(u.size(-1), dA, v)
        next_state = contract("... m n, ... n -> ... m", AL, state)
        next_state = next_state + v

        if conj:
            next_state = next_state[..., : next_state.size(-1) // 2]
        return next_state

    def step(self, u, state):
        return self.krylov.step(u, state)

    def double_length(self):
        self.krylov.double_length()


""" Tests below """


def generate_krylov(H, N, L, measure="legs", rank=1):
    trainable = DictConfig(
        {
            "A": 1,
            "B": 2,
            "C": 1,
            "dt": 1,
        }
    )
    lr = DictConfig(
        {
            "A": 1e-3,
            "B": 1e-3,
            "C": None,
            "dt": 1e-3,
        }
    )

    A, B = hippo.transition(measure, N)
    A = torch.as_tensor(A, dtype=torch.float)
    B = torch.as_tensor(B, dtype=torch.float)[:, 0]
    C = torch.ones(H, N)
    log_dt = torch.log((1 + 10 * torch.arange(H) / H) * 1 / L)
    krylov_real = SSKernelSlow(L, A, B, torch.ones(N), log_dt)

    w, p, q, B, V = hippo.nplr(measure, N)
    C = contract(
        "ij, ... j -> ... i", V.conj().transpose(-1, -2), V.new_ones(H, N)
    )  # V^* B
    A = torch.diag_embed(_conj(w)) - contract(
        "... r p, ... r q -> ... p q", _conj(p), _conj(q).conj()
    )
    krylov_slow = SSKernelSlow(
        L, A, _conj(B), _conj(C), log_dt, trainable=trainable, lr=lr
    )

    print("krylov real vs krylov complex", krylov_real() - krylov_slow())
    krylov = SSKernelNPLR(L, w, p, q, B, C, log_dt, setup_C=True)

    krylov._setup()
    krylov._check()

    print("krylov slow vs krylov fast", krylov_slow() - krylov())

    krylov = SSKernelNPLR(
        L, w, p, q, B, C, log_dt, trainable=trainable, lr=lr, setup_C=True
    )
    krylov_slow = SSKernelSlow(
        L, A, _conj(B), _conj(C), log_dt, trainable=trainable, lr=lr
    )

    return krylov_real.to(device), krylov_slow.to(device), krylov.to(device)


def benchmark_krylov():
    N = 64
    L = 4096
    H = 256

    krylov_real, krylov_slow, krylov = generate_krylov(H, N, L)

    utils.compare_outputs(krylov_slow(), krylov(), full=False, relative=True)

    utils.benchmark_forward(100, krylov_slow, desc="krylov fft manual")
    utils.benchmark_forward(100, krylov, desc="krylov fft rank")
    utils.benchmark_backward(100, krylov_slow, desc="krylov fft manual")
    utils.benchmark_backward(100, krylov, desc="krylov fft rank")

    utils.benchmark_memory(krylov_slow, desc="krylov fft manual")
    utils.benchmark_memory(krylov, desc="krylov fft rank")


def test_step():
    B = 2
    L = 4
    N = 4
    H = 3

    krylov_real, krylov_slow, krylov = generate_krylov(H, N, L)

    print("TESTING SLOW STEP")
    krylov_slow._setup()
    state = torch.zeros(B, H, N).to(device)
    u = torch.ones(B, H, L).to(device)
    for u_ in torch.unbind(u, dim=-1):
        y_, state = krylov_slow.step(u_, state=state)
        print("y", y_, y_.shape)
    print("state", state, state.shape)

    print("TESTING STEP")
    krylov._setup()
    state = torch.zeros(B, H, N).to(device)
    u = torch.ones(B, H, L).to(device)
    for u_ in torch.unbind(u, dim=-1):
        y_, state = krylov.step(u_, state=state, linear=False)
        # y_, state = krylov.step(u_, state=state)
        print("y", y_, y_.shape)
    print("state", state, state.shape)

    print("TESTING LINEAR STEP")
    krylov._setup()
    state = torch.zeros(B, H, N // 2).to(device).to(torch.cfloat)
    u = torch.ones(B, H, L).to(device)
    for u_ in torch.unbind(u, dim=-1):
        y_, state = krylov.step(u_, state=state, linear=True)
        print("y", y_, y_.shape)
    print("state", state, state.shape)


@torch.inference_mode()
def benchmark_step():
    B = 1024
    L = 16
    N = 64
    H = 1024

    _, _, krylov = generate_krylov(H, N, L)
    krylov._setup()

    print("Benchmarking Step")
    state = torch.zeros(B, H, N).to(device)
    u = torch.ones(B, H).to(device)
    utils.benchmark_forward(16, krylov.step, u, state, linear=False, desc="dense step")

    print("Benchmarking Linear Step")
    state = torch.zeros(B, H, N).to(device)  # .to(torch.cfloat)
    u = torch.ones(B, H).to(device)
    utils.benchmark_forward(16, krylov.step, u, state, linear=True, desc="linear step")

    state = torch.zeros(B, H, N // 2).to(device)  # .to(torch.cfloat)
    u = torch.ones(B, H).to(device)
    utils.benchmark_forward(
        16, krylov.step, u, state, linear=True, desc="linear step conj"
    )


def test_double():
    torch.set_printoptions(sci_mode=False, linewidth=160)
    L = 8
    N = 4
    H = 3

    _, krylov_slow, krylov = generate_krylov(H, N, L, "legs", 1)

    k = krylov.forward()
    print(k, k.shape)
    krylov._check()

    krylov.double_length()
    k = krylov.forward()
    print(k, k.shape)


def test_state():
    B = 1
    N = 4
    L = 4
    H = 3
    krylov_real, krylov_slow, krylov = generate_krylov(H, N, L)

    state = torch.ones(B, H, N // 2, device=device, dtype=torch.cfloat)

    k, k_state = krylov_slow.forward(state=state)
    print("k slow", k)
    print("k_state slow", k_state)

    k, k_state = krylov.forward(state=state)
    print("k", k)
    print("k_state", k_state)


if __name__ == "__main__":
    from benchmark import utils

    device = "cuda"  # 'cpu'
    device = torch.device(device)

    # benchmark_krylov()
    # test_double()
    # test_step()
    # benchmark_step()
    test_state()
