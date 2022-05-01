""" Core S4 convolution kernel implementing the 'normal plus low-rank' algorithm.

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
from opt_einsum import contract, contract_expression
from omegaconf import DictConfig

import src.models.hippo.hippo as hippo
from src.models.functional.krylov import krylov, power

import src.utils.train

log = src.utils.train.get_logger(__name__)

try:
    from extensions.cauchy.cauchy import cauchy_mult
    has_cauchy_extension = True
    log.info("CUDA extension for cauchy multiplication found.")
except:
    log.warn(
        "CUDA extension for cauchy multiplication not found. Install by going to extensions/cauchy/ and running `python setup.py install`. This should speed up end-to-end training by 10-50%"
    )
    has_cauchy_extension = False

try:
    import pykeops
    from src.models.functional.cauchy import cauchy_conj
    has_pykeops = True
    log.info("Pykeops installation found.")
except ImportError:
    has_pykeops = False
    from src.models.functional.cauchy import cauchy_slow
    if not has_cauchy_extension:
        log.error(
            "Falling back on slow Cauchy kernel. Install at least one of pykeops or the CUDA extension for efficiency."
        )



_isnan = lambda x: torch.isnan(x).any()
_isinf = lambda x: torch.isinf(x).any()

_conj = lambda x: torch.cat([x, x.conj()], dim=-1)
_c2r = torch.view_as_real
_r2c = torch.view_as_complex

if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 10):
    _resolve_conj = lambda x: x.conj().resolve_conj()
else:
    _resolve_conj = lambda x: x.conj()

def bilinear(dt, A, B=None):
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
        ).squeeze(-1) # (... N)

    dA = torch.linalg.solve(A_backwards, A_forwards)  # (... N N)
    return dA, dB


class OptimModule(nn.Module):
    """ Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters """

    def register(self, name, tensor, trainable=False, lr=None, wd=None):
        """Utility method: register a tensor as a buffer or trainable parameter"""

        if trainable:
            self.register_parameter(name, nn.Parameter(tensor))
        else:
            self.register_buffer(name, tensor)

        optim = {}
        if trainable and lr is not None:
            optim["lr"] = lr
        if trainable and wd is not None:
            optim["weight_decay"] = wd
        if len(optim) > 0:
            setattr(getattr(self, name), "_optim", optim)


class SSKernelNPLR(OptimModule):
    """ Stores a representation of and computes the SSKernel function K_L(A^dt, B^dt, C) corresponding to a discretized state space, where A is Normal + Low Rank (NPLR)

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
    def _setup_C(self, L):
        """ Construct C~ from C

        Two modes are supported: go directly to length L if self.L is 1, or length is doubled
        """

        if self.L.item() == 0:
            if self.verbose: log.info(f"S4: Initializing kernel to length {L}")
            double_length = False
        elif L > self.L.item(): # 2*int(self.L) == L:
            if self.verbose: log.info(f"S4: Doubling length from L = {self.L.item()} to {2*self.L.item()}")
            double_length = True
            L = self.L.item() # Convenience for the math below
        else: return

        C = _r2c(self.C)
        dA, _ = self._setup_state()
        dA_L = power(L, dA)
        # Multiply C by I - dA_L
        C_ = _conj(C)
        prod = contract("h m n, c h n -> c h m", dA_L.transpose(-1, -2), C_)
        if double_length: prod = -prod # Multiply by I + dA_L instead
        C_ = C_ - prod
        C_ = C_[..., :self.N] # Take conjugate pairs again
        self.C.copy_(_c2r(C_))

        self.L = 2*self.L if double_length else self.L+L # Preserve type/device

    def _omega(self, L, dtype, device, cache=True):
        """ Calculate (and cache) FFT nodes and their "unprocessed" version with the bilinear transform
        This should be called everytime the internal length self.L changes """

        # Use cached if available
        if cache and hasattr(self, 'omega') and self.omega.size(-1) == L//2+1:
            return self.omega, self.z

        omega = torch.tensor(
            np.exp(-2j * np.pi / (L)), dtype=dtype, device=device
        )  # \omega_{2L}
        omega = omega ** torch.arange(0, L // 2 + 1, device=device)
        z = 2 * (1 - omega) / (1 + omega)

        # Cache if necessary
        if cache:
            self.omega = omega
            self.z = z
        return omega, z

    def __init__(
        self,
        w, P, B, C, log_dt,
        L=None, # starting/maximum length of kernel
        trainable=None,
        lr=None,
        lr_dt=None,
        verbose=False,
        keops=False,
        fast_gate=False,
        quadrature=None,
    ):
        """
        L: Maximum length; this module computes an SSM kernel of length L
        w: (n_ssm, N)
        p: (r, n_ssm, N) low-rank correction to A
        A represented by diag(w) - pq^*

        B: (n_ssm, N)
        dt: (H) timescale per feature
        C: (C, H, N) system is 1-D to c-D (channels)

        trainable: toggle which of the parameters is trainable
        lr: add hook to set lr of hippo parameters specially (everything besides C)

        Note: tensor shape N here denotes half the true state size, because of conjugate symmetry
        """

        super().__init__()
        self.verbose = verbose
        self.keops = keops
        self.fast_gate = fast_gate

        # Rank of low-rank correction
        self.rank = P.shape[-3]
        assert w.size(-1) == P.size(-1) == B.size(-1) == C.size(-1)
        self.H = log_dt.size(-1)
        self.N = w.size(-1)

        # Check different SSM inits
        assert w.size(-2) == P.size(-2) == B.size(-2) # Number of copies
        assert self.H % w.size(0) == 0
        self.n_ssm = w.size(0)
        self.copies = self.H // w.size(0)
        if lr_dt is None: lr_dt = lr

        # Broadcast everything to correct shapes
        C = C.expand(torch.broadcast_shapes(C.shape, (1, self.H, self.N))) # (H, C, N)
        B = B.unsqueeze(0) # (1, 1, N)

        # Register parameters
        self.C = nn.Parameter(_c2r(_resolve_conj(C)))
        train = False
        if trainable is None: trainable = {}
        if trainable == False: trainable = {}
        if trainable == True: trainable, train = {}, True
        self.register("log_dt", log_dt, trainable.get('dt', train), lr_dt, 0.0)
        self.register("B", _c2r(B), trainable.get('B', train), lr, 0.0)
        self.register("P", _c2r(P), trainable.get('P', train), lr, 0.0)
        log_w_real = torch.log(-w.real + 1e-3) # Some of the HiPPO methods have real part 0
        w_imag = w.imag
        self.register("log_w_real", log_w_real, trainable.get('A', 0), lr, 0.0)
        self.register("w_imag", w_imag, trainable.get('A', train), lr, 0.0)

        self.l_max = L
        self.register_buffer('L', torch.tensor(0)) # Internal length

        self.quadrature = quadrature

    def _w(self):
        # Get the internal w (diagonal) parameter
        w_real = -torch.exp(self.log_w_real)
        w_imag = self.w_imag
        w = w_real + 1j * w_imag
        return w

    def forward(self, state=None, rate=1.0, L=None):
        """
        state: (..., s, N) extra tensor that augments B
        rate: sampling rate factor

        returns: (..., c+s, L)
        """

        # Initialize C~ if necessary (done in forward pass so it's on the correct device)
        if self.L.item() == 0 and self.l_max is not None and self.l_max > 0:
            self._setup_C(self.l_max)

        # Handle sampling rate logic
        # The idea is that this kernel's length (in continuous units) is self.L, while we are asked to provide a kernel of length L at (relative) sampling rate rate
        # If either are not passed in, assume we're not asked to change the scale of our kernel
        assert not (rate is None and L is None)
        if rate is None:
            rate = self.L.item() / L
        if L is None:
            L = round(self.L.item() / rate)

        # Increase the internal length if needed
        continuous_L = round(rate*L)
        while continuous_L > self.L.item():
            self._setup_C(continuous_L)
        discrete_L = round(self.L.item()/rate)

        if self.fast_gate: dt = torch.exp(torch.sinh(self.log_dt)) * rate
        else: dt = torch.exp(self.log_dt) * rate
        B = _r2c(self.B)
        C = _r2c(self.C)
        P = _r2c(self.P)
        Q = P.conj()
        w = self._w()

        # Get FFT nodes of right length
        omega, z = self._omega(discrete_L, dtype=w.dtype, device=w.device, cache=(rate==1.0))

        # Broadcast parameters to same hidden features H
        B = repeat(B, '1 t n -> 1 (v t) n', v=self.copies)
        P = repeat(P, 'r t n -> r (v t) n', v=self.copies)
        Q = repeat(Q, 'r t n -> r (v t) n', v=self.copies)
        w = repeat(w, 't n -> (v t) n', v=self.copies)

        # Augment B
        if state is not None:
            # Have to "unbilinear" the state to put it into the same "type" as B
            # Compute 1/dt * (I + dt/2 A) @ state

            # Can do this without expanding (maybe minor speedup using conj symmetry in theory), but it's easier to read this way
            s = _conj(state) if state.size(-1) == self.N else state # (B H N)
            sA = (
                s * _conj(w) # (B H N)
                - contract('bhm, rhm, rhn -> bhn', s, _conj(Q), _conj(P))
            )
            s = s / dt.unsqueeze(-1) + sA / 2
            s = s[..., :self.N]

            B = torch.cat([s, B], dim=-3)  # (s+1, H, N)

        # Incorporate dt into A
        w = w * dt.unsqueeze(-1)  # (H N)

        # Stack B and p, C and q for convenient batching
        B = torch.cat([B, P], dim=-3) # (s+1+r, H, N)
        C = torch.cat([C, Q], dim=-3) # (c+r, H, N)

        # Incorporate B and C batch dimensions
        v = B.unsqueeze(-3) * C.unsqueeze(-4)  # (s+1+r, c+r, H, N)
        # w = w[None, None, ...]  # (1, 1, H, N)
        # z = z[None, None, None, ...]  # (1, 1, 1, L)

        # Calculate resolvent at omega
        if has_cauchy_extension and z.dtype == torch.cfloat and not self.keops:
            r = cauchy_mult(v, z, w, symmetric=True)
        elif has_pykeops:
            r = cauchy_conj(v, z, w)
        else:
            r = cauchy_slow(v, z, w)
        r = r * dt[None, None, :, None]  # (S+1+R, C+R, H, L)

        # Low-rank Woodbury correction
        if self.rank == 1:
            k_f = r[:-1, :-1, :, :] - r[:-1, -1:, :, :] * r[-1:, :-1, :, :] / (1 + r[-1:, -1:, :, :])
        elif self.rank == 2:
            r00 = r[: -self.rank, : -self.rank, :, :]
            r01 = r[: -self.rank, -self.rank :, :, :]
            r10 = r[-self.rank :, : -self.rank, :, :]
            r11 = r[-self.rank :, -self.rank :, :, :]
            det = (1 + r11[:1, :1, :, :]) * (1 + r11[1:, 1:, :, :]) - r11[:1, 1:, :, :] * r11[1:, :1, :, :]
            s = (
                r01[:, :1, :, :] * (1 + r11[1:, 1:, :, :]) * r10[:1, :, :, :]
                + r01[:, 1:, :, :] * (1 + r11[:1, :1, :, :]) * r10[1:, :, :, :]
                - r01[:, :1, :, :] * (r11[:1, 1:, :, :]) * r10[1:, :, :, :]
                - r01[:, 1:, :, :] * (r11[1:, :1, :, :]) * r10[:1, :, :, :]
            )
            s = s / det
            k_f = r00 - s
        else:
            r00 = r[:-self.rank, :-self.rank, :, :]
            r01 = r[:-self.rank, -self.rank:, :, :]
            r10 = r[-self.rank:, :-self.rank, :, :]
            r11 = r[-self.rank:, -self.rank:, :, :]
            r11 = rearrange(r11, "a b h n -> h n a b")
            r11 = torch.linalg.inv(torch.eye(self.rank, device=r.device) + r11)
            r11 = rearrange(r11, "h n a b -> a b h n")
            k_f = r00 - torch.einsum("i j h n, j k h n, k l h n -> i l h n", r01, r11, r10)

        # Final correction for the bilinear transform
        k_f = k_f * 2 / (1 + omega)

        # Move from frequency to coefficients
        k = torch.fft.irfft(k_f, n=discrete_L)  # (S+1, C, H, L)

        # # Truncate to target length
        k = k[..., :L]

        if state is not None:
            k_state = k[:-1, :, :, :]  # (S, C, H, L)
        else:
            k_state = None
        k_B = k[-1, :, :, :] # (C H L)

        if self.quadrature == 'trapezoid':
            w = torch.ones(*k_B.shape).to(k_B) * dt[None, :, None]
            w[..., 0] /= 2.
            w[..., -1] /= 2
            k_B = k_B * w
        elif self.quadrature == 'simpson':
            w = torch.ones(*k_B.shape).to(k_B) * dt[None, :, None] / 3.
            w[..., 1:-1:2] *= 4
            w[..., 2:-1:2] *= 2
            k_B = k_B * w

        return k_B, k_state

    @torch.no_grad()
    def double_length(self):
        # if self.verbose: log.info(f"S4: Doubling length from L = {self.L} to {2*self.L}")
        self._setup_C(2*self.L)

    @torch.no_grad()
    def _check(self):
        """Check if A, B, C parameters and vanilla SSKernel construction can be recovered"""

        self.setup_step()

        K = krylov(self.L, self.dA, self.dB, self.dC)

        diff = K - self.forward(L=self.L)[0]
        print("checking DPLR Kernel construction", torch.sum(diff ** 2))

    @torch.no_grad()
    def _setup_linear(self):
        """ Create parameters that allow fast linear stepping of state """
        w = self._w()
        B = _r2c(self.B) # (H N)
        P = _r2c(self.P)
        Q = P.conj()

        # Repeat w shape properly
        B = repeat(B, '1 t n -> 1 (v t) n', v=self.copies)
        P = repeat(P, 'r t n -> r (v t) n', v=self.copies)
        Q = repeat(Q, 'r t n -> r (v t) n', v=self.copies)
        w = repeat(w, 't n -> (v t) n', v=self.copies)

        # Prepare Linear stepping
        dt = torch.exp(self.log_dt)
        D = (2.0 / dt.unsqueeze(-1) - w).reciprocal()  # (H, N)
        R = (torch.eye(self.rank, dtype=w.dtype, device=w.device) + 2*contract('r h n, h n, s h n -> h r s', Q, D, P).real) # (H r r)
        Q_D = rearrange(Q*D, 'r h n -> h r n')
        try:
            R = torch.linalg.solve(R.to(Q_D), Q_D) # (H r N)
        except torch._C._LinAlgError:
            R = torch.tensor(np.linalg.solve(R.to(Q_D).cpu(), Q_D.cpu())).to(Q_D)
        R = rearrange(R, 'h r n -> r h n')

        self.step_params = {
            "D": D, # (H N)
            "R": R, # (r H N)
            "P": P, # (r H N)
            "Q": Q, # (r H N)
            "B": B, # (1 H N)
            "E": 2.0 / dt.unsqueeze(-1) + w, # (H N)
        }

    def _step_state_linear(self, u=None, state=None):
        """
        Version of the step function that has time O(N) instead of O(N^2) per step, which takes advantage of the DPLR form and bilinear discretization.

        Unfortunately, as currently implemented it's about 2x slower because it calls several sequential operations. Perhaps a fused CUDA kernel implementation would be much faster

        u: (H) input
        state: (H, N/2) state with conjugate pairs
          Optionally, the state can have last dimension N
        Returns: same shape as state
        """
        C = _r2c(self.C) # View used for dtype/device

        if u is None: # Special case used to find dA
            u = torch.zeros(self.H, dtype=C.dtype, device=C.device)
        if state is None: # Special case used to find dB
            state = torch.zeros(self.H, self.N, dtype=C.dtype, device=C.device)

        step_params = self.step_params.copy()
        if state.size(-1) == self.N: # Only store half of the conjugate pairs; should be true by default
            # There should be a slightly faster way using conjugate symmetry
            contract_fn = lambda p, x, y: contract('r h n, r h m, ... h m -> ... h n', _conj(p), _conj(x), _conj(y))[..., :self.N] # inner outer product
        else:
            assert state.size(-1) == 2*self.N
            step_params = {k: _conj(v) for k, v in step_params.items()}
            # TODO worth setting up a contract_expression in default_state if we want to use this at inference time for stepping
            contract_fn = lambda p, x, y: contract('r h n, r h m, ... h m -> ... h n', p, x, y) # inner outer product
        D = step_params["D"]  # (H N)
        E = step_params["E"]  # (H N)
        R = step_params["R"]  # (r H N)
        P = step_params["P"]  # (r H N)
        Q = step_params["Q"]  # (r H N)
        B = step_params["B"]  # (1 H N)

        new_state = E * state - contract_fn(P, Q, state) # (B H N)
        new_state = new_state + 2.0 * B * u.unsqueeze(-1)  # (B H N)
        new_state = D * (new_state - contract_fn(P, R, new_state))

        return new_state

    def _setup_state(self):
        """ Construct dA and dB for discretized state equation """

        # Construct dA and dB by using the stepping
        self._setup_linear()
        C = _r2c(self.C) # Just returns a view that we use for finding dtype/device

        state = torch.eye(2*self.N, dtype=C.dtype, device=C.device).unsqueeze(-2) # (N 1 N)
        dA = self._step_state_linear(state=state)
        dA = rearrange(dA, "n h m -> h m n")
        # self.dA = dA # (H N N)

        u = C.new_ones(self.H)
        dB = self._step_state_linear(u=u)
        dB = _conj(dB)
        dB = rearrange(dB, '1 h n -> h n') # (H N)
        return dA, dB

    def _step_state(self, u, state):
        """ Must be called after self.default_state() is used to construct an initial state!  """
        next_state = self.state_contraction(self.dA, state) + self.input_contraction(self.dB, u)
        return next_state


    def setup_step(self, mode='dense'):
        """ Set up dA, dB, dC discretized parameters for stepping """
        self.dA, self.dB = self._setup_state()

        # Calculate original C
        dA_L = power(self.L, self.dA)
        I = torch.eye(self.dA.size(-1)).to(dA_L)
        C = _conj(_r2c(self.C)) # (H C N)

        dC = torch.linalg.solve(
            I - dA_L.transpose(-1, -2),
            C.unsqueeze(-1),
        ).squeeze(-1)
        self.dC = dC

        # Do special preprocessing for different step modes

        self._step_mode = mode
        if mode == 'linear':
            # Linear case: special step function for the state, we need to handle output
            # use conjugate symmetry by default, which affects the output projection
            self.dC = 2*self.dC[:, :, :self.N]
        elif mode == 'diagonal':
            # Eigendecomposition of the A matrix
            L, V = torch.linalg.eig(self.dA)
            V_inv = torch.linalg.inv(V)
            # Check that the eigendedecomposition is correct
            if self.verbose:
                print("Diagonalization error:", torch.dist(V @ torch.diag_embed(L) @ V_inv, self.dA))

            # Change the parameterization to diagonalize
            self.dA = L
            self.dB = contract('h n m, h m -> h n', V_inv, self.dB)
            self.dC = contract('h n m, c h n -> c h m', V, self.dC)

        elif mode == 'dense':
            pass
        else: raise NotImplementedError("NPLR Kernel step mode must be {'dense' | 'linear' | 'diagonal'}")


    def default_state(self, *batch_shape):
        C = _r2c(self.C)
        N = C.size(-1)
        H = C.size(-2)

        # Cache the tensor contractions we will later do, for efficiency
        # These are put in this function because they depend on the batch size
        if self._step_mode !='linear':
            N *= 2

            if self._step_mode == 'diagonal':
                self.state_contraction = contract_expression(
                    "h n, ... h n -> ... h n",
                    (H, N),
                    batch_shape + (H, N),
                )
            else:
                # Dense (quadratic) case: expand all terms
                self.state_contraction = contract_expression(
                    "h m n, ... h n -> ... h m",
                    (H, N, N),
                    batch_shape + (H, N),
                )

            self.input_contraction = contract_expression(
                "h n, ... h -> ... h n",
                (H, N), # self.dB.shape
                batch_shape + (H,),
            )

        self.output_contraction = contract_expression(
            "c h n, ... h n -> ... c h",
            (C.shape[0], H, N), # self.dC.shape
            batch_shape + (H, N),
        )

        state = torch.zeros(*batch_shape, H, N, dtype=C.dtype, device=C.device)
        return state

    def step(self, u, state):
        """ Must have called self.setup_step() and created state with self.default_state() before calling this """

        if self._step_mode == 'linear':
            new_state = self._step_state_linear(u, state)
        else:
            new_state = self._step_state(u, state)
        y = self.output_contraction(self.dC, new_state)
        return y, new_state


class SSKernelSlow(OptimModule):
    """Slow version of SSKernel function for illustration and benchmarking.

    - Caches discretized matrices A^(dt), B^(dt)
    - Computes K_L(A^dt, B^dt, C)

    Usage:
    ```
    krylov = SSKernelSlow(L, A, B, C, log_dt)()
    ```
    Result is expected to be equal to SSKernelNPLR(L, w, P, B, C, log_dt, P)() if A = w - PP^*
    """

    def __init__(self, A, B, C, log_dt, L=None, trainable=None, lr=None):
        super().__init__()
        self.L = L
        self.N = A.size(-1)
        self.H = log_dt.size(-1)

        C = C.expand(torch.broadcast_shapes(C.shape, (1, self.H, self.N))) # (C, H, N)

        # Register parameters
        train = False
        if trainable is None: trainable = {}
        if trainable == False: trainable = {}
        if trainable == True: trainable, train = {}, True
        self.register("log_dt", log_dt, trainable.get('dt', train), lr)
        self.register("A", A, trainable.get('A', train), lr)
        self.register("B", B, trainable.get('B', train), lr)
        # NOTE leaving in complex form for convenience, which means it currently won't work with DDP and might have incorrect param count
        # This class shouldn't be used for anything other than testing and simple ablations, so this is fine
        # self.register("C", C.conj().resolve_conj(), True, None, wd=None)
        self.C = nn.Parameter(_resolve_conj(C))

        # Cache if nothing is trained
        self.trainable = trainable.get('dt', train) or trainable.get('A', train) or trainable.get('B', train)
        self.K = None # Compute in forward pass since that ensures correct device

    def forward(self, state=None, rate=1.0, L=None):
        if L is None: L = self.L
        # This class shouldn't support the more advanced sampling and variable length functionalities, since it's just for testing
        # But the code from NPLR could be pasted here if desired
        assert rate == 1.0 and L is not None

        if self.trainable:
            dA, dB = bilinear(torch.exp(self.log_dt), self.A, self.B)
            k = krylov(L, dA, dB, self.C)  # (H L)
        else:
            if self.K is None:
                dA, dB = bilinear(torch.exp(self.log_dt), self.A, self.B)
                self.K = krylov(L, dA, dB) # (H N L)
            k = contract('hnl,chn->chl', self.K[..., :L], self.C)

        if state is not None:
            state = state.to(self.dA)
            state = contract("... n m, ... m -> ... n", self.dA, state)
            k_state = krylov(L, self.dA, state.unsqueeze(-3), self.C)
        else:
            k_state = None
        return k, k_state
        # return k.to(torch.float)

    def default_state(self, *batch_shape):
        state = torch.zeros(*batch_shape, self.H, self.N, dtype=self.C.dtype, device=self.C.device)
        return state

    def _setup_state(self):
        self.dA, self.dB = bilinear(torch.exp(self.log_dt), self.A, self.B)

    def setup_step(self):
        self._setup_state()
        self.dC = self.C

    def step(self, u, state):
        next_state = contract("h m n, b h n -> b h m", self.dA, state) \
                + contract("h n, b h -> b h n", self.dB, u)
        y = contract("c h n, b h n -> b c h", self.dC, next_state)
        return y, next_state


class SSKernelDiag(OptimModule):
    """ Version using (complex) diagonal state matrix. Main difference is this uses the ZOH instead of Bilinear transform. Note that it is slower and less memory efficient than the NPLR kernel because of lack of kernel support.

    """

    def __init__(
        self,
        w, C, log_dt,
        L=None,
        disc='bilinear',
        trainable=None,
        lr=None,
        quadrature=None,
    ):

        super().__init__()
        self.L = L
        self.disc = disc

        # Rank of low-rank correction
        assert w.size(-1) == C.size(-1)
        self.H = log_dt.size(-1)
        self.N = w.size(-1)
        assert self.H % w.size(0) == 0
        self.copies = self.H // w.size(0)

        # Broadcast everything to correct shapes
        C = C.expand(torch.broadcast_shapes(C.shape, (1, self.H, self.N))) # (H, C, N)

        # Register parameters
        # C is a regular parameter, not part of state
        self.C = nn.Parameter(_c2r(_resolve_conj(C)))
        train = False
        if trainable is None: trainable = {}
        if trainable == False: trainable = {}
        if trainable == True: trainable, train = {}, True
        self.register("log_dt", log_dt, trainable.get('dt', train), lr, 0.0)

        if self.disc in ['bilinear', 'zoh', 'foh']:
            log_w_real = torch.log(-w.real + 1e-3) # Some of the HiPPO methods have real part 0
            w_imag = w.imag
            self.register("log_w_real", log_w_real, trainable.get('A', 0), lr, 0.0)
            self.register("w_imag", w_imag, trainable.get('A', train), lr, 0.0)
        elif self.disc == 'dss':
            self.register("w", _c2r(w), trainable.get('A', train), lr, 0.0)
        else: raise NotImplementedError

        self.quadrature = quadrature


    def _w(self):
        # Get the internal w (diagonal) parameter
        if self.disc != 'dss':
            w_real = -torch.exp(self.log_w_real)
            w_imag = self.w_imag
            w = w_real + 1j * w_imag
        else:
            w = _r2c(self.w)  # (..., N)
        w = repeat(w, 't n -> (v t) n', v=self.copies) # (H N)
        return w

    def forward(self, state=None, rate=1.0, L=None):
        """
        state: (..., s, N) extra tensor that augments B
        rate: sampling rate factor

        returns: (..., c+s, L)
        """
        # Handle sampling rate logic
        # The idea is that this kernel's length (in continuous units) is self.L, while we are asked to provide a kernel of length L at (relative) sampling rate rate
        # If either are not passed in, assume we're not asked to change the scale of our kernel
        assert not (rate is None and L is None)
        if rate is None:
            rate = self.L / L
        if L is None:
            L = round(self.L / rate)

        dt = torch.exp(self.log_dt) * rate # (H)
        C = _r2c(self.C) # (C H N)
        w = self._w() # (H N)


        # Incorporate dt into A
        dtA = w * dt.unsqueeze(-1)  # (H N)

        if self.disc == 'zoh':
            # Power up
            K = dtA.unsqueeze(-1) * torch.arange(L, device=w.device) # (H N L)
            C = C * (torch.exp(dtA)-1.) / w
            K = contract('chn, hnl -> chl', C, torch.exp(K))
            K = 2*K.real
        elif self.disc == 'foh':
            # Power up
            K = dtA.unsqueeze(-1) * torch.arange(L, device=w.device) # (H N L)
            K_exp = torch.exp(K)
            C = C / (dt.unsqueeze(-1) * w ** 2)
            exp_dA = torch.exp(dtA)
            C_0 = - (exp_dA - 1. - dtA * exp_dA) * C # kernel for conv with u_k
            C_1 = (exp_dA - 1. - dtA) * C # kernel for conv with u_{k+1}
            K_0 = contract('chn, hnl -> chl', C_0, K_exp)
            K_1 = contract('chn, hnl -> chl', C_1, K_exp)
            K_0 = 2*K_0.real
            K_1 = 2*K_1.real
            K = K_1
            K[..., -1] = 0.
            K[..., 1:] += K_0[..., :-1]
        elif self.disc == 'bilinear':
            dA = (1. + dtA/2) / (1. - dtA/2)
            K = dA.unsqueeze(-1) ** torch.arange(L, device=w.device) # (H N L)
            C = C * (1. - dtA/2).reciprocal() * dt.unsqueeze(-1) # or * dtA / w
            K = contract('chn, hnl -> chl', C, K)
            K = 2*K.real
        else:
            # Implementation from DSS meant for case when real eigenvalues can be positive
            P = dtA.unsqueeze(-1) * torch.arange(L, device=C.device)   # [H N L]
            w_gt_0 = w.real > 0                                    # [N]
            if w_gt_0.any():
                with torch.no_grad():
                    P_max = dtA * (w_gt_0 * (L-1))                # [H N]
                P = P - P_max.unsqueeze(-1)                                  # [H N L]
            S = P.exp()                                                      # [H N L]

            dtA_neg = dtA * (1 - 2*w_gt_0)                  # [H N]
            # S.sum(-1) == den / num
            num = dtA_neg.exp() - 1                                    # [H N]
            den = (dtA_neg * L).exp() - 1                              # [H N]

            # Inline reciprocal function for DSS logic
            x = den * w
            x_conj = _resolve_conj(x)
            r = x_conj / (x*x_conj + 1e-7)

            C = C * num * r             # [C H N]
            K = contract('chn,hnl->chl', C, S).float()


        if self.quadrature == 'trapezoid':
            w = torch.ones(*K.shape).to(K) * dt[None, :, None]
            w[..., 0] /= 2.
            w[..., -1] /= 2
            K = K * w
        elif self.quadrature == 'simpson':
            w = torch.ones(*K.shape).to(K) * dt[None, :, None] / 3.
            w[..., 1:-1:2] *= 4
            w[..., 2:-1:2] *= 2
            K = K * w

        return K, None

    def setup_step(self):
        dt = torch.exp(self.log_dt) # (H)
        C = _r2c(self.C) # (C H N)
        w = self._w() # (H N)

        # Incorporate dt into A
        dtA = w * dt.unsqueeze(-1)  # (H N)
        if self.disc == 'zoh':
            self.dA = torch.exp(dtA) # (H N)
            self.dC = C * (torch.exp(dtA)-1.) / w # (C H N)
        elif self.disc == 'bilinear':
            self.dA = (1. + dtA/2) / (1. - dtA/2)
            self.dC = C * (1. - dtA/2).reciprocal() * dt.unsqueeze(-1) # or * dtA / w
        self.dB = self.dC.new_ones(self.H, self.N) # (H N)

        # if self.disc == 'zoh':
        #     # Power up
        #     K = dtA.unsqueeze(-1) * torch.arange(L, device=w.device) # (H N L)
        #     C = C * (torch.exp(dtA)-1.) / w
        #     K = contract('chn, hnl -> chl', C, torch.exp(K))
        #     K = 2*K.real
        # elif self.disc == 'bilinear':
        #     dA = (1. + dtA/2) / (1. - dtA/2)
        #     K = dA.unsqueeze(-1) ** torch.arange(L, device=w.device) # (H N L)
        #     C = C * (1. - dtA/2).reciprocal() * dt.unsqueeze(-1) # or * dtA / w
        #     K = contract('chn, hnl -> chl', C, K)
        #     K = 2*K.real

    def default_state(self, *batch_shape):
        C = _r2c(self.C)
        state = torch.zeros(*batch_shape, self.H, self.N, dtype=C.dtype, device=C.device)
        return state

    def step(self, u, state):
        next_state = contract("h n, b h n -> b h n", self.dA, state) \
                + contract("h n, b h -> b h n", self.dB, u)
        y = contract("c h n, b h n -> b c h", self.dC, next_state)
        return 2*y.real, next_state


    # @staticmethod
    # def reciprocal(x, epsilon=1e-7, clamp=False):
    #     """ returns 1 / x, with bounded norm """
    #     x_conj = x.conj()
    #     norm_sq = (x*x_conj).real.clamp(epsilon) if clamp else (x*x_conj + epsilon)
    #     return x_conj / norm_sq


class HippoSSKernel(nn.Module):
    """Wrapper around SSKernel that generates A, B, C, dt according to HiPPO arguments.

    The SSKernel is expected to support the interface
    forward()
    default_state()
    setup_step()
    step()
    """

    def __init__(
        self,
        H,
        N=64,
        L=None,
        measure="legs",
        rank=1,
        channels=1, # 1-dim to C-dim map; can think of C as having separate "heads"
        dt_min=0.001,
        dt_max=0.1,
        deterministic=False,
        trainable=None, # Dictionary of options to train various HiPPO parameters
        lr=None, # Hook to set LR of hippo parameters differently
        lr_dt=None,
        mode="nplr",  # 'slow' for complex naive version, 'real' for real naive version
        n_ssm=1, # Copies of the ODE parameters A and B. Must divide H
        precision=1, # 1 (single) or 2 (double) for the kernel
        resample=False,  # If given inputs of different lengths, adjust the sampling rate. Note that L should always be provided in this case, as it assumes that L is the true underlying length of the continuous signal
        verbose=False,
        fast_gate=False,
        diag_tilt=0.0,
        rank_weight=1.0,
        measure_args={},
        **kernel_args,
    ):
        super().__init__()
        self.N = N
        self.H = H
        assert not (resample and L is None), "Cannot have sampling rate adjustment and no base length passed in"
        self.precision = precision
        dtype = torch.double if self.precision == 2 else torch.float
        cdtype = torch.cfloat if dtype == torch.float else torch.cdouble
        self.rate = None if resample else 1.0
        self.channels = channels
        self.n_ssm = n_ssm

        # Generate dt
        log_dt = torch.rand(self.H, dtype=dtype) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        if fast_gate:
            log_dt = torch.asinh(log_dt)

        # Compute the preprocessed representation
        if mode == "real":  # For testing and ablation purposes
            # Generate A, B
            A, B = hippo.transition(measure, self.N)
            A = torch.as_tensor(A, dtype=dtype)
            B = torch.as_tensor(B, dtype=dtype)[:, 0]

            # Generate C
            C = torch.randn(channels, self.H, self.N, dtype=dtype)

            self.kernel = SSKernelSlow(
                A, B, C, log_dt, L=L,
                trainable=trainable, lr=lr,
            )
        else:
            # Generate low rank correction p for the measure
            if measure == "random":
                w, P, B, C, _ = hippo.random_dplr(self.N, rank=rank, H=n_ssm, dtype=dtype, **measure_args)
            elif measure == 'hippo':
                w0, P0, B0, C0, _ = hippo.nplr('legs', self.N, rank, dtype=dtype)
                w1, P1, B1, C1, _ = hippo.nplr('fourier', self.N, rank, dtype=dtype)
                w = torch.stack([w0, w1], dim=0)
                P = torch.stack([P0, P1], dim=1)
                B = torch.stack([B0, B1], dim=0)
                C = torch.stack([C0, C1], dim=0)
            else:
                w, P, B, C, _ = hippo.nplr(measure, self.N, rank, dtype=dtype)
                w = w.unsqueeze(0) # (s N), s is num SSM copies
                P = P.unsqueeze(1) # (r s N)
                B = B.unsqueeze(0) # (s N)
                C = C.unsqueeze(0) # (H N)

            # Handle extra options
            w = w - diag_tilt
            P = P * rank_weight

            # Broadcast C to have H channels
            if deterministic:
                C = repeat(C, 't n -> c (v t) n', c=channels, v=self.H // C.size(0))
            else:
                C = torch.randn(channels, self.H, self.N // 2, dtype=cdtype)

            # Broadcast other parameters to have n_ssm copies
            assert self.n_ssm % B.size(-2) == 0 \
                    and self.n_ssm % P.size(-2) == 0 \
                    and self.n_ssm % w.size(-2) == 0
            # Broadcast tensors to n_ssm copies
            # These will be the parameters, so make sure tensors are materialized and contiguous
            B = repeat(B, 't n -> (v t) n', v=self.n_ssm // B.size(-2)).clone().contiguous()
            P = repeat(P, 'r t n -> r (v t) n', v=self.n_ssm // P.size(-2)).clone().contiguous()
            w = repeat(w, 't n -> (v t) n', v=self.n_ssm // w.size(-2)).clone().contiguous()

            if mode == "nplr":
                self.kernel = SSKernelNPLR(
                    w, P, B, C,
                    log_dt, L=L,
                    trainable=trainable,
                    lr=lr, lr_dt=lr_dt,
                    verbose=verbose,
                    fast_gate=fast_gate,
                    **kernel_args,
                )
            elif mode == "diag":
                C = C * repeat(B, 't n -> (v t) n', v=H//self.n_ssm)
                self.kernel = SSKernelDiag(
                    w, C, log_dt, L=L,
                    trainable=trainable,
                    lr=lr,
                    **kernel_args,
                )
            elif mode == "slow":  # Testing only
                A = torch.diag_embed(_conj(w)) \
                        - contract("... r p, ... r q -> ... p q", _conj(P), _conj(P).conj())
                self.kernel = SSKernelSlow(
                    A, _conj(B), _conj(C), log_dt, L=L,
                    trainable=trainable, lr=lr,
                )
            else: raise NotImplementedError(f"{mode=} is not valid")

    def forward(self, state=None, L=None):
        k, k_state = self.kernel(state=state, rate=self.rate, L=L)
        k_state = None if k_state is None else k_state.float()
        return k.float(), k_state

    @torch.no_grad()
    def forward_state(self, u, state):
        """ Forward the state through a sequence, i.e. computes the state after passing chunk through SSM

        state: (..., H, N)
        u: (..., H, L)

        Returns: (..., H, N)
        """

        self.kernel._setup_state()
        dA, dB = self.kernel.dA, self.kernel.dB # (H N N) (H N)

        conj = state.size(-1) != dA.size(-1)
        if conj: state = _conj(state)

        v = contract('h n, ... h l -> ... h n l', dB, u.flip(-1)) # dB.unsqueeze(-1) * u.flip(-1).unsqueeze(-2)
        AL, v = power(u.size(-1), dA, v)
        next_state = contract("... m n, ... n -> ... m", AL, state)
        next_state = next_state + v

        if conj: next_state = next_state[..., : next_state.size(-1) // 2]
        return next_state

    def setup_step(self):
        self.kernel.setup_step()

    def step(self, u, state, **kwargs):
        u, state = self.kernel.step(u, state, **kwargs)
        return u.float(), state

    def default_state(self, *args, **kwargs):
        return self.kernel.default_state(*args, **kwargs)



""" Tests below """


def generate_kernel(H, N, L, measure="legs", rank=1):
    A, B = hippo.transition(measure, N)
    A = torch.as_tensor(A, dtype=torch.float)
    B = torch.as_tensor(B, dtype=torch.float)[:, 0]
    # _C = torch.ones(1, H, N)
    _C = torch.randn(1, H, N)
    log_dt = torch.log((1 + 10 * torch.arange(H) / H) * 1 / L)

    # kernel slow real
    kernel_real = SSKernelSlow(A, B, _C, log_dt, L=L)
    kernel_real.to(device)
    kernel_real.setup_step()

    # kernel slow complex
    w, p, B, C, V = hippo.nplr(measure, N, rank=rank)
    C = contract(
        "ij, ... j -> ... i", V.conj().transpose(-1, -2), _C.to(V)
    )  # V^* B
    A = torch.diag_embed(_conj(w)) - contract(
        "... r p, ... r q -> ... p q", _conj(p), _conj(p).conj()
    )
    kernel_slow = SSKernelSlow(A, _conj(B), _conj(C), log_dt, L=L)
    kernel_slow.to(device)
    kernel_slow.setup_step()

    print("kernel real vs kernel complex", kernel_real()[0] - kernel_slow()[0])
    kernel = SSKernelNPLR(w.unsqueeze(0), p.unsqueeze(1), B.unsqueeze(0), C.unsqueeze(0).unsqueeze(0), log_dt, L=L, verbose=True)
    kernel.to(device) # TODO need to add this line for custom CUDA kernel
    kernel.setup_step()
    kernel._check()

    print("kernel slow vs kernel fast", kernel_slow(L=L)[0] - kernel(L=L)[0])

    # print(f"dA \nslow:\n{kernel_slow.dA}\nfast:\n{kernel.dA}")
    # print("dC real slow fast:", kernel_real.dC, kernel_slow.dC, kernel.dC)

    return kernel_real.to(device), kernel_slow.to(device), kernel.to(device)


def benchmark_kernel():
    N = 64
    L = 4096
    H = 256

    kernel_real, kernel_slow, kernel = generate_kernel(H, N, L)

    utils.compare_outputs(kernel_slow(), kernel(), full=False, relative=True)

    utils.benchmark_forward(100, kernel_slow, desc="kernel fft manual")
    utils.benchmark_forward(100, kernel, desc="kernel fft rank")
    utils.benchmark_backward(100, kernel_slow, desc="kernel fft manual")
    utils.benchmark_backward(100, kernel, desc="kernel fft rank")

    utils.benchmark_memory(kernel_slow, desc="kernel fft manual")
    utils.benchmark_memory(kernel, desc="kernel fft rank")


def test_step(diagonal=False, **kwargs):
    B = 1
    L = 8
    N = 4
    H = 3

    kernel_real, kernel_slow, kernel = generate_kernel(H, N, L, **kwargs)

    print("=====TESTING SLOW STEP=====")
    kernel_slow.setup_step()
    state = kernel_slow.default_state(B)
    u = torch.ones(B, H, L).to(device)
    ys = []
    for u_ in torch.unbind(u, dim=-1):
        y_, state = kernel_slow.step(u_, state=state)
        ys.append(y_)
    print("state", state, state.shape)
    y = torch.stack(ys, dim=-1)
    print("y", y, y.shape)

    print("=======TESTING STEP=======")
    kernel.setup_step(mode='dense')
    state = kernel.default_state(B)# torch.zeros(B, H, N).to(device).to(torch.cfloat)
    u = torch.ones(B, H, L).to(device)
    ys = []
    for u_ in torch.unbind(u, dim=-1):
        y_, state = kernel.step(u_, state=state)
        ys.append(y_)
    print("state", state, state.shape)
    y = torch.stack(ys, dim=-1)
    print("y", y, y.shape)

    print("=====TESTING LINEAR STEP=====")
    kernel.setup_step(mode='linear')
    state = kernel.default_state(B)
    u = torch.ones(B, H, L).to(device)
    ys = []
    for u_ in torch.unbind(u, dim=-1):
        y_, state = kernel.step(u_, state=state)
        ys.append(y_)
    print("state", state, state.shape)
    y = torch.stack(ys, dim=-1)
    print("y", y, y.shape)

    if diagonal:
        print("=====TESTING DIAGONAL STEP=====")
        kernel.setup_step(mode='diagonal')
        state = kernel.default_state(B)
        u = torch.ones(B, H, L).to(device)
        ys = []
        for u_ in torch.unbind(u, dim=-1):
            y_, state = kernel.step(u_, state=state)
            ys.append(y_)
        print("state", state, state.shape)
        y = torch.stack(ys, dim=-1)
        print("y", y, y.shape)


@torch.inference_mode()
def benchmark_step():
    B = 1024
    L = 16
    N = 64
    H = 1024

    _, _, kernel = generate_kernel(H, N, L)
    kernel.setup_step()

    print("Benchmarking Step")
    state = torch.zeros(B, H, N).to(device)
    u = torch.ones(B, H).to(device)
    utils.benchmark_forward(16, kernel.step, u, state, linear=False, desc="dense step")

    print("Benchmarking Linear Step")
    state = torch.zeros(B, H, N).to(device)  # .to(torch.cfloat)
    u = torch.ones(B, H).to(device)
    utils.benchmark_forward(16, kernel.step, u, state, linear=True, desc="linear step")

    state = torch.zeros(B, H, N // 2).to(device)  # .to(torch.cfloat)
    u = torch.ones(B, H).to(device)
    utils.benchmark_forward(
        16, kernel.step, u, state, linear=True, desc="linear step conj"
    )


def test_double():
    # torch.set_printoptions(sci_mode=False, linewidth=160)
    L = 8
    N = 4
    H = 3

    _, kernel_slow, kernel = generate_kernel(H, N, L, "legs", 1)

    print("Testing Length Doubling")
    print("=======================")
    print("Original:")
    k = kernel.forward()[0]
    # print(k, k.shape)
    kernel._check()

    print("Doubled:")
    kernel.double_length()
    k_ = kernel.forward()[0]
    # print(k, k_.shape)
    print("Doubling error:", torch.sum((k_[..., :k.size(-1)] - k)**2))


def test_state():
    B = 1
    N = 4
    L = 4
    H = 3
    kernel_real, kernel_slow, kernel = generate_kernel(H, N, L)

    state = torch.ones(B, H, N // 2, device=device, dtype=torch.cfloat)

    k, k_state = kernel_slow.forward(state=state)
    print("k slow", k)
    print("k_state slow", k_state)

    k, k_state = kernel.forward(state=state)
    print("k", k)
    print("k_state", k_state)


def test_diag():
    bs = 1
    L = 8
    N = 4
    H = 3

    w, P, B, C, _ = hippo.nplr('legs', N, 1)
    w = w.unsqueeze(0) # (s N), s is num SSM copies
    P = P.unsqueeze(1) # (r s N)
    B = B.unsqueeze(0) # (s N)
    # C = repeat(C, 'n -> 1 h n', h=H)
    C = torch.randn(1, H, N//2, dtype=torch.cfloat)
    # C = C.unsqueeze(0).unsqueeze(0) # (1 H N)
    log_dt = torch.log((1 + 10 * torch.arange(H) / H) * 1 / L)

    kernel = SSKernelDiag(w, C, log_dt, L=L)
    kernel.to(device)
    kernel.setup_step()

    K, _ = kernel(L=8)
    print(torch.cumsum(K, axis=-1))


    print("=====TESTING DIAG STEP=====")
    kernel.setup_step()
    state = kernel.default_state(bs)
    u = torch.ones(bs, H, L).to(device)
    ys = []
    for u_ in torch.unbind(u, dim=-1):
        y_, state = kernel.step(u_, state=state)
        ys.append(y_)
    print("state", state, state.shape)
    y = torch.stack(ys, dim=-1)
    print("y", y, y.shape)


if __name__ == "__main__":
    from benchmark import utils

    device = "cuda"  # 'cpu'
    device = torch.device(device)

    torch.set_printoptions(sci_mode=False, linewidth=160)

    has_cauchy_extension = False # turn off CUDA kernel for ease of testing, don't have to move between devices

    # generate_kernel(3, 4, 8, measure='legt', rank=2)
    # benchmark_kernel()
    # test_double()
    # test_step(diagonal=True, measure='legt', rank=2) # TODO this needs to be fixed
    # benchmark_step()
    # test_state()
    test_diag()
