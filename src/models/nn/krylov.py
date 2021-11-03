""" Original SSM Kernel class, which was called the Krylov function (e.g. in original LSSL paper).

DEPRECATED and moved to src.models.sequence.ss.kernel
"""

if __name__ == '__main__':
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
import src.models.functional.cauchy as cauchy
from src.models.functional.krylov import krylov, power

_isnan = lambda x: torch.isnan(x).any()
_isinf = lambda x: torch.isinf(x).any()
def resolvent(A, B, C, z):
    """
    A: (... N N)
    B: (... N)
    C: (... N)
    z: (... L)
    returns: (... L) represents C (z - A)^{-1} B
    """
    if A.is_complex() or B.is_complex() or C.is_complex() or z.is_complex():
        dtype = torch.cfloat
    else:
        dtype = torch.float
    N = A.shape[-1]
    I = torch.eye(N).to(device=A.device, dtype=dtype)
    A_ = I * z[...,None,None] - A.unsqueeze(-3)
    B = B.to(dtype)[..., None, :, None] # (... L N 1)
    r = torch.linalg.solve(A_, B).squeeze(-1) # (... L N)
    r = torch.sum(r*C.to(dtype).unsqueeze(-2), dim=-1)
    return r

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

    v = B.conj() * C
    r = cauchy.cauchy(v, z, w)
    return r


class PreprocessModule(nn.Module):
    """ Interface for Module that requires an explicit (expensive) preprocessing step during initialization """

    def __init__(self, *args, **kwargs):
        """ Registers buffers that get setup later """
        super().__init__()

        # Register buffers
        for i, p in enumerate(args):
            if isinstance(p, torch.Tensor):
                self.register_buffer('_init_'+str(i), p)
            else:
                setattr(self, '_init_'+str(i), p)
        for name, p in kwargs.items():
            if isinstance(p, torch.Tensor):
                self.register_buffer('_init_'+name, p)
            else:
                setattr(self, '_init_'+name, p)

        self._n_args = len(args)
        self._kwargs = kwargs.keys()

        self._is_setup = False

    def setup(self):
        """ Preprocess arguments into a representation.

        This is meant to be called after moving model to device
        Subclasses of this interface must implement the _setup method
        """


        # Make sure to not duplicate setup
        if self._is_setup: return
        else: self._is_setup = True

        # Setup module using registered buffers which are on proper device
        init_args = [getattr(self, '_init_'+str(i)) for i in range(self._n_args)]
        init_kwargs = {name: getattr(self, '_init_'+name) for name in self._kwargs}

        # Call actual setup function
        self._setup(*init_args, **init_kwargs)

        # Delete registered buffers
        for i in range(len(init_args)):
            delattr(self, '_init_'+str(i))
        for name in init_kwargs:
            delattr(self, '_init_'+name)

    def _setup(self):
        raise NotImplementedError

    def register(self, name, tensor, trainable=0, lr=None, wd=None, repeat=1):
        """ Utility method: register a tensor as a buffer or trainable parameter """

        if trainable == 0:
            self.register_buffer(name, tensor)
        elif trainable == 1:
            self.register_parameter(name, nn.Parameter(tensor))
        elif trainable == 2:
            tensor = tensor.repeat(repeat, *(1,)*len(tensor.shape))
            self.register_parameter(name, nn.Parameter(tensor))
        else: raise NotImplementedError

        optim = {}
        if trainable and lr is not None:
            optim['lr'] = lr
            # setattr(getattr(self, name), '_lr', lr)
        if trainable and wd is not None:
            optim['weight_decay'] = wd
            # setattr(getattr(self, name), '_wd', wd)
        if len(optim) > 0:
            setattr(getattr(self, name), '_optim', optim)


class KrylovNPLR(PreprocessModule):
    """ Stores a representation of and computes the Krylov function K_L(A^dt, B^dt, C) corresponding to a discretized state space, where A is Normal + Low Rank (NPLR)

    The class name stands for 'State-Space Krylov for Normal Plus Low-Rank'.
    The parameters of this function are as follows.

    A: (... N N) the state matrix
    B: (... N) input matrix
    C: (... N) output matrix
    dt: (...) timescales / discretization step size
    p, q: (... P N) low-rank correction to A, such that Ap=A+pq^T is a normal matrix

    The forward pass of this Module returns:
    (... L) that represents represents FFT Krylov_L(A^dt, B^dt, C)

    """

    # def _process_C(self, L, w, V, p, q, B, C):
    #     dA, _ = KrylovNPLR.bilinear(dt, w, V, p, q, B)
    #     # Handle (I - dA^L) term to move coefficients into evaluations
    #     dA_L = power(L, dA)
    #     C_ = torch.einsum('... m n, ... m -> ... n', dA_L, C)
    #     C = C - C_

    def _process_C(self, L):
        # dt = torch.exp(self.log_dt)
        # w = torch.view_as_complex(self.w)
        # V = torch.view_as_complex(self.V)
        # BV = torch.view_as_complex(self.BV)
        # CV = torch.view_as_complex(self.CV)
        # if precision == 2:
        #     dt = torch.exp(self.log_dt).double()
        #     w = w.to(torch.cdouble)
        #     V = V.to(torch.cdouble)
        #     # p = p.to(torch.cdouble)
        #     # q = q.to(torch.cdouble)
        #     # B = B.to(torch.cdouble)
        #     # C = C.to(torch.cdouble)
        #     BV = BV.to(torch.cdouble)
        #     CV = CV.to(torch.cdouble)

        # V_inv = V.transpose(-1, -2).conj()
        # B_ = contract('... r m, ... m n -> ... r n', BV, V_inv)
        # B_ = 2*B_.real # (..., 2, N)
        # B = B_[..., 0, :]
        # p = B_[..., 1:, :]

        # C_ = contract('... r m, ... m n -> ... r n', CV, V_inv)
        # C_ = 2*C_.real # (..., 2, N)
        # # C = CV[..., 0, :]
        # C = C_[..., 0, :]
        # q = C_[..., 1:, :]
        # qV = CV[..., 1:, :]


        # # Find C
        # dA, dB = KrylovNPLR.bilinear(dt, w, V, p, q, B)
        # dA_L = power(self.L, dA)
        # I = torch.eye(dA.size(-1)).to(dA)
        # C = contract('... m, ... m n -> ... n', C, I-dA_L)
        # CV = contract('... n, ... n m -> ... m', C.to(V), V).contiguous()
        # CV = torch.cat([CV.unsqueeze(-2), qV], dim=-2) # (..., 1+r, N)

        # self.bilinear_DNPLR(dt, w, V, p, q, B)
        # self._check_bilinear(dt, w, V, p, q, B)

        # dA = self.bilinear_DNPLR(separate=False)
        # dA_L = power(L, dA)
        # # [21-09-28] hotfix for potential nans
        # if torch.isinf(dA_L).any() or torch.isnan(dA_L).any():
        #     dA_L = dA_L * torch.logical_not(torch.isnan(dA_L))
        #     dA_L = dA_L * torch.logical_not(torch.isinf(dA_L))
        # I = torch.eye(dA.size(-1)).to(dA)
        CV = torch.view_as_complex(self.CV)
        N = CV.size(-1)
        # Multiply CV by I - dA_L
        CV_ = CV[..., 0, :]
        CV_ = torch.cat([CV_, CV_.conj()], dim=-1)
        # CV_ = CV_ - contract('... m, ... m n -> ... n', CV_, dA_L)
        if _isnan(CV_): breakpoint()
        CV_ = CV_[..., :N]
        CV = torch.cat([CV_.unsqueeze(-2), CV[..., 1:, :]], dim=-2)

        return CV

    def _nodes(self, L, dtype, device):
        # Cache FFT nodes and their "unprocessed" them with the bilinear transform
        # nodes = torch.tensor(np.exp(-2j * np.pi / (L)), dtype=torch.cfloat, device=Ap.device) # \omega_{2L}
        nodes = torch.tensor(np.exp(-2j * np.pi / (L)), dtype=dtype, device=device) # \omega_{2L}
        nodes = nodes ** torch.arange(0, L//2+1, device=device)
        z = 2*(1-nodes)/(1+nodes)
        return nodes, z

    def _setup(self, L, A, B, C, log_dt, p, q, trainable=None, lr=None, use_length=True):
        """ Preprocess arguments into a representation. This occurs after init so that these operations can occur after moving model to device

        L: Maximum length; this module computes Krylov function of length L
        A: (..., N, N)
        B: (..., N)
        C: (..., N)
        dt: (...)
        p: (..., N) low-rank correction to A
        q: (..., N)
        """

        # Rank of low-rank correction
        assert p.shape[-2] == q.shape[-2]
        self.rank = p.shape[-2]
        self.L = L
        self.use_length = use_length

        # dt = torch.exp(log_dt)

        # Store diagonalization of A+pp^T
        Ap = A + torch.sum(q.unsqueeze(-2)*p.unsqueeze(-1), dim=-3)
        w, V = torch.linalg.eig(Ap) # (..., N) (..., N, N)
        w = w[..., 0::2].contiguous()
        V = V[..., 0::2].contiguous()

        # Discretize
        # dA, _ = KrylovNPLR.bilinear(dt, w, V, p, q, B)

        # # Handle (I - dA^L) term to move coefficients into evaluations
        # dA_L = power(L, dA)
        # C_ = torch.einsum('... m n, ... m -> ... n', dA_L, C)
        # C = C - C_
        # if use_length:
        #     C = self._process_C(self.L, w, V, p, q, B, C)

        ### Process B, C, p, q, V into V @ [B p], V @ [C q]

        # Augment B and C with low rank correction
        B = B.unsqueeze(-2) # (..., 1, N)
        C = C.unsqueeze(-2) # (..., 1, N)
        if len(B.shape) > len(p.shape):
            p = p.repeat(B.shape[:-2] + (1, 1))
        B = torch.cat([B, p], dim=-2)
        if len(C.shape) > len(q.shape):
            q = q.repeat(C.shape[:-2] + (1, 1))
        C = torch.cat([C, q], dim=-2)

        # Multiply by V
        # BV = (B.to(V).unsqueeze(-2) @ V).squeeze(-2)
        # CV = (C.to(V).unsqueeze(-2) @ V).squeeze(-2)
        BV = contract('... r n, ... n m -> ... r m', B.to(V), V).contiguous()
        CV = contract('... r n, ... n m -> ... r m', C.to(V), V).contiguous()


        if self.use_length:
            assert L is not None
            nodes, z = self._nodes(L, dtype=w.dtype, device=w.device)
            self.register_buffer('nodes', torch.view_as_real(nodes))
            self.register_buffer('z', torch.view_as_real(z))
        # # Cache FFT nodes and their "unprocessed" them with the bilinear transform
        # # nodes = torch.tensor(np.exp(-2j * np.pi / (L)), dtype=torch.cfloat, device=Ap.device) # \omega_{2L}
        # nodes = torch.tensor(np.exp(-2j * np.pi / (L)), dtype=w.dtype, device=w.device) # \omega_{2L}
        # nodes = nodes ** torch.arange(0, L//2+1, device=w.device)
        # z = 2*(1-nodes)/(1+nodes)
        # self.register_buffer('nodes', torch.view_as_real(nodes))
        # self.register_buffer('z', torch.view_as_real(z))
        self.register_buffer('V', torch.view_as_real(V))

        # Register parameters
        if trainable is None: trainable = DictConfig({'A': 0, 'B': 0, 'C': 0, 'dt': 0})
        if lr is None: lr = DictConfig({'A': None, 'B': None, 'C': None, 'dt': None})
        repeat = CV.size(0)
        self.register('log_dt', log_dt, trainable.dt, lr.dt, 0.0)
        self.register('w', torch.view_as_real(w), trainable.A, lr.A, 0.0, repeat=repeat)
        self.register('BV', torch.view_as_real(BV), trainable.B, lr.B, 0.0, repeat=repeat)
        self.register('CV', torch.view_as_real(CV), trainable.C, lr.C)

        if self.use_length:
            # original_CV = CV
            # contract('... m, ... m n -> ... n', original_CV, V.transpose(-1, -2).conj()).real*2 - C
            CV = self._process_C(L)
            delattr(self, 'CV')
            self.register('CV', torch.view_as_real(CV), trainable.C, lr.C)

            # print("checking ABC at init")
            # self._check_ABC()
            # _, _, check_C = self.ABC()
            # _, _, check_CV = self.ABC_DNPLR()
            # V = torch.view_as_complex(self.V)
            # check_C_ = contract('... m, ... m n -> ... n', check_CV, torch.cat([V, V.conj()], dim=-1).transpose(-1, -2).conj())
            # print(torch.sum((check_C-check_C_)**2))
            # breakpoint()


    def forward(self, state=None, rate=1.0, L=None):
        """
        state: (..., s, N) extra tensor that augments B
        rate: sampling rate factor
        """
        if L is not None: raise NotImplementedError

        dt = torch.exp(self.log_dt) * rate
        BV = torch.view_as_complex(self.BV)

        w = torch.view_as_complex(self.w) # (..., N)
        # z = torch.view_as_complex(self.z) # (..., L)

        # TODO adjust based on rate times normal max length

        if not self.use_length:
            assert L is not None
            CV = self._process_C(L)
            nodes, z = self._nodes(L, dtype=w.dtype, device=w.device)
        else:
            CV = torch.view_as_complex(self.CV)
            if rate == 1.0:
                nodes = torch.view_as_complex(self.nodes)
                z = torch.view_as_complex(self.z) # (..., L)
            else:
                L = int(self.L / rate)
                nodes, z = self._nodes(L, dtype=w.dtype, device=w.device)

        # Augment B
        if state is not None:
            V = torch.view_as_complex(self.V)
            # Have to "unbilinear" the state to put it into the same "type" as B
            # Compute (I + dt/2 A) @ state
            def check(): # More complicated version to debug
                A = contract('ij, ... j, kj -> ... ik', V, w, V.conj()) # Ap^{-1} = V @ w^{-1} @ V^T
                A = 2 * A.real
                V_inv = V.transpose(-1, -2).conj()
                B_ = contract('... r m, ... m n -> ... r n', BV, V_inv)
                B_ = 2*B_.real # (..., 2, N)
                B = B_[..., 0, :]
                p = B_[..., 1:, :]

                C_ = contract('... r m, ... m n -> ... r n', CV, V_inv)
                C_ = 2*C_.real # (..., 2, N)
                q = C_[..., 1:, :]
                A = A - p.unsqueeze(-1)*q.unsqueeze(-2)
                assert self.rank == 1
                A = A.squeeze(-3)
                A = A / 2 + torch.eye(A.size(-1)).to(A) / dt.unsqueeze(-1).unsqueeze(-1)
                sA = contract('... m n, ... s n -> ... s m', A, state)
                sV = contract('... s n, ... n m -> ... s m', sA.to(V), V)
                return sV

            sV = contract('... s n, ... n m -> ... s m', state.to(V), V) # (... s N)
            pV = BV[..., 1:, :] # (... r N)
            qV = CV[..., 1:, :] # (... r N)

            # Calculate contract('... s n, ... r n, ... r m -> ... s m', sV, qV.conj(), pV), but take care of conjugate symmetry
            sA = sV * w.conj().unsqueeze(-2) - (2+0j)*(sV@qV.conj().transpose(-1,-2)).real @ pV
            sV = sV / dt.unsqueeze(-1).unsqueeze(-1) + sA/2

            if (l := len(BV.shape)) < len(sV.shape):
                BV = BV.repeat(sV.shape[:-l] + (1,)*l)
            BV = torch.cat([sV, BV], dim=-2) # (..., 2+s, N)

        # Incorporate dt into A
        w = w * dt.unsqueeze(-1) # (... N)

        # Incorporate B and C batch dimensions
        v = BV.unsqueeze(-3).conj() * CV.unsqueeze(-2) # (..., 2, 2, N)
        w = w[..., None, None, :] # (..., 1, 1, N)
        z = z[..., None, None, :] # (..., 1, 1, L)

        # Calculate resolvent at nodes
        # v = v.to(torch.cfloat)
        # z = z.to(torch.cfloat)
        # w = w.to(torch.cfloat)
        # dt = dt.to(torch.cfloat)
        r = cauchy.cauchy_conj(v, z, w)
        r = r * dt[..., None, None, None] # (..., 1+r, 1+r, L)

        # Low-rank Woodbury correction
        if self.rank == 1:
            k_f = r[..., :-1, :-1, :] - r[..., :-1, -1:, :] * r[..., -1:, :-1, :] / (1 + r[..., -1:, -1:, :])
        elif self.rank == 2:
            r00 = r[..., :-self.rank, :-self.rank, :]
            r01 = r[..., :-self.rank, -self.rank:, :]
            r10 = r[..., -self.rank:, :-self.rank, :]
            r11 = r[..., -self.rank:, -self.rank:, :]
            det = (1+r11[..., :1, :1, :])*(1+r11[..., 1:, 1:, :]) - r11[..., :1, 1:, :]*r11[..., 1:, :1, :]
            s = r01[..., :, :1, :] * (1+r11[..., 1:, 1:, :]) * r10[..., :1, :, :] \
                    + r01[..., :, 1:, :] * (1+r11[..., :1, :1, :]) * r10[..., 1:, :, :] \
                    - r01[..., :, :1, :] * (r11[..., :1, 1:, :]) * r10[..., 1:, :, :] \
                    - r01[..., :, 1:, :] * (r11[..., 1:, :1, :]) * r10[..., :1, :, :]
            s = s / det
            k_f = r00 - s
        else:
            r00 = r[..., :-self.rank, :-self.rank, :]
            r01 = r[..., :-self.rank, -self.rank:, :]
            r10 = r[..., -self.rank:, :-self.rank, :]
            r11 = r[..., -self.rank:, -self.rank:, :]
            r11 = rearrange(r11, '... a b n -> ... n a b')
            r11 = torch.linalg.inv(torch.eye(self.rank,device=r.device)+r11)
            r11 = rearrange(r11, '... n a b -> ... a b n')
            k_f = r00 - torch.einsum('... i j n, ... j k n, ... k l n -> ... i l n', r01, r11, r10)

        # Final correction for the bilinear transform
        k_f = k_f * 2 / (1+nodes)

        k = torch.fft.irfft(k_f) # (..., 1, 1+s, L)
        if state is not None:
            k_state = k[..., 0, :-1, :] # (..., s, L)
            k_B = k[..., 0, -1, :] # (..., L)
            return k_B, k_state
        else:
            return k.squeeze(-2).squeeze(-2)

    @staticmethod
    def bilinearNPLR(cls, dt, w, V, p, q, B=None):
        """ [21-09-21] TODO There appears to be potential numerical stability issues with this method (when implementing doubling) """
        """ Bilinear discretization for matrix A = VwV^T - pq^T

        dt: (...)
        w: (..., N/2)
        V: (..., N, N/2) conjugate pairs
        p, q: (..., R, N)
        B: (..., N)
        """
        if p.size(-2) > 1: raise NotImplementedError
        p = p.squeeze(-2)
        q = q.squeeze(-2)

        w = torch.cat([w, w.conj()], dim=-1)
        V = torch.cat([V, V.conj()], dim=-1)
        # Check: A == torch.real(V @ torch.diag_embed(w) @ V.transpose(-1,-2).conj()) - p.unsqueeze(-1)*q
        w_ = 2./dt.unsqueeze(-1) - w # (... N)
        Ap_inv = contract('ij, ... j, kj -> ... ik', V, w_.reciprocal(), V.conj()) # Ap^{-1} = V @ w^{-1} @ V^T
        Ap_inv = Ap_inv.real

        # Calculate (I - dt/2 A)^{-1} using Woodbury
        # Should be equivalent to:
        # A_backwards = torch.linalg.inv(torch.linalg.inv(Ap_inv) + p.unsqueeze(-1)*q.unsqueeze(-2))
        r = torch.reciprocal(1. + contract('... i, ... i j, ... j -> ...', p, Ap_inv, q))
        correction = contract('... i j, ... j, ..., ... k, ... k l -> ... i l', Ap_inv, p, r, q, Ap_inv)
        A_backwards = Ap_inv - correction

        # 2/dt + A
        A_forwards = contract('ij, ... j, kj -> ... ik', V, 2/dt.unsqueeze(-1) + w, V.conj())
        A_forwards = A_forwards.real - p.unsqueeze(-1) * q.unsqueeze(-2)

        dA = A_backwards @ A_forwards
        dB = 2 * (A_backwards @ B.unsqueeze(-1)).squeeze(-1) if B is not None else None
        return dA, dB

    @staticmethod
    def bilinear(dt, w, V, p, q, B=None, separate=False):
        if p.size(-2) > 1: raise NotImplementedError
        # p = p.squeeze(-2)
        # q = q.squeeze(-2)

        # w = torch.cat([w, w.conj()], dim=-1)
        # V = torch.cat([V, V.conj()], dim=-1)
        # Check: A == torch.real(V @ torch.diag_embed(w) @ V.transpose(-1,-2).conj()) - p.unsqueeze(-1)*q
        # w_ = 2./dt.unsqueeze(-1) - w # (... N)
        Ap = contract('ij, ... j, kj -> ... ik', V, w, V.conj()) # Ap^{-1} = V @ w^{-1} @ V^T
        Ap = 2*Ap.real
        A = Ap - torch.sum(q.unsqueeze(-2)*p.unsqueeze(-1), dim=-3)

        return KrylovSlow.bilinear(dt, A, B=B, separate=separate)

    def _check_bilinear(self, dt, w, V, p, q, B):

        A_forwards, A_backwards, dB = self.bilinear(dt, w, V, p, q, B, separate=True)

        # A_f_d, A_b_d = self.bilinear_DNPLR(dt, w, V, p, q, B, separate=True)
        A_f_d, A_b_d = self.bilinear_DNPLR(separate=True)
        V = torch.cat([V, V.conj()], dim=-1)
        A_f = .5*dt[:, None, None] * contract('ij, ... jk, lk -> ... il', V, A_f_d, V.conj())
        A_b = 2./dt[:, None, None] * contract('ij, ... jk, lk -> ... il', V, A_b_d, V.conj())

        print("checking A forwards", torch.sum((A_f-A_forwards)**2))
        print("checking A backwards", torch.sum((A_b-A_backwards)**2))

    # def bilinear_DNPLR(self, dt, w, V, p, q, B, separate=False):
    def bilinear_DNPLR(self, separate=False):
        # Note that this function doesn't use B or C, just p and q
        if self.rank > 1: raise NotImplementedError # Only do this for rank 1 case

        dt = torch.exp(self.log_dt)
        w = torch.view_as_complex(self.w)
        V = torch.view_as_complex(self.V)
        BV = torch.view_as_complex(self.BV)
        CV = torch.view_as_complex(self.CV)

        # TODO because of the low rank and matrix power, it's doable but a bit annoying to handle the conjugate pairs. easier to just materialize them
        w = torch.cat([w, w.conj()], dim=-1)
        V = torch.cat([V, V.conj()], dim=-1)
        BV = torch.cat([BV, BV.conj()], dim=-1)
        CV = torch.cat([CV, CV.conj()], dim=-1)

        # pV = BV[..., 1:, :]
        # qV = CV[..., 1:, :]
        # We only consider rank 1 case so use index -1 instead of 1:
        pV = BV[..., -1, :] # (H, N)
        qV = CV[..., -1, :] # (H, N)

        # Forward A, diagonalized
        A_f_d = torch.diag_embed(2./dt[:, None] + w) - contract('... n, ... m -> ... n m', pV.conj(), qV)

        # Backwards A, matrix inv
        # inv = torch.diag_embed(2./dt[:, None] - w) + contract('... n, ... m -> ... n m', pV.conj(), qV)
        # A_inv = contract('ij, ... jk, lk -> ... il', V, torch.linalg.inv(inv), V.conj())
        # A_inv = 2./dt[:, None, None] * A_inv
        # print(A_inv)

        # Backwards A, woodbury
        d = (2./dt.unsqueeze(-1) - w).reciprocal() # (H, N)
        r = 1 + contract('... n, ... n, ... n -> ...', qV, d, pV.conj())
        r = r.reciprocal()
        A_b_d = torch.diag_embed(d) - contract('... p, ... p, ..., ... q, ... q -> ... p q', d, pV.conj(), r, qV, d)

        if separate:
            return A_f_d, A_b_d
        else:
            return A_b_d @ A_f_d


    def ABC(self):
        # TODO abstract this into a method
        dt = torch.exp(self.log_dt)
        w = torch.view_as_complex(self.w)
        V = torch.view_as_complex(self.V)
        BV = torch.view_as_complex(self.BV)
        CV = torch.view_as_complex(self.CV)
        # if precision == 2:
        #     dt = torch.exp(self.log_dt).double()
        #     w = w.to(torch.cdouble)
        #     V = V.to(torch.cdouble)
        #     # p = p.to(torch.cdouble)
        #     # q = q.to(torch.cdouble)
        #     # B = B.to(torch.cdouble)
        #     # C = C.to(torch.cdouble)
        #     BV = BV.to(torch.cdouble)
        #     CV = CV.to(torch.cdouble)

        V_inv = V.transpose(-1, -2).conj()
        B_ = contract('... r m, ... m n -> ... r n', BV, V_inv)
        B_ = 2*B_.real # (..., 2, N)
        B = B_[..., 0, :]
        p = B_[..., 1:, :]

        C_ = contract('... r m, ... m n -> ... r n', CV, V_inv)
        C_ = 2*C_.real # (..., 2, N)
        # C = CV[..., 0, :]
        C = C_[..., 0, :]
        q = C_[..., 1:, :]


        # Find C
        # dA, dB = KrylovNPLR.bilinear(dt, w, V, p, q, B)
        if self.use_length:
            dA, dB = KrylovNPLR.bilinear(dt, w, V, p, q, B)
            dA_L = power(self.L, dA)
            I = torch.eye(dA.size(-1)).to(dA)
            C = torch.linalg.solve(I - dA_L.transpose(-1,-2), C.unsqueeze(-1)).squeeze(-1)

        # Find A
        w = torch.cat([w, w.conj()], dim=-1)
        V = torch.cat([V, V.conj()], dim=-1)
        # Check: A == torch.real(V @ torch.diag_embed(w) @ V.transpose(-1,-2).conj()) - p.unsqueeze(-1)*q
        # w_ = 2./dt.unsqueeze(-1) - w # (... N)
        Ap = contract('ij, ... j, kj -> ... ik', V, w, V.conj()) # Ap^{-1} = V @ w^{-1} @ V^T
        Ap = Ap.real
        A = Ap - torch.sum(q.unsqueeze(-2)*p.unsqueeze(-1), dim=-3)
        return A, B, C
        # return self.log_dt, w, V, p, q, B, C

    def ABC_DNPLR(self):
        assert self.rank == 1, "Not implemented for rank > 1"
        w = torch.view_as_complex(self.w)
        BpV = torch.view_as_complex(self.BV)
        CqV = torch.view_as_complex(self.CV)
        w = torch.cat([w, w.conj()], dim=-1)
        BpV = torch.cat([BpV, BpV.conj()], dim=-1)
        CqV = torch.cat([CqV, CqV.conj()], dim=-1)

        BV = BpV[..., 0, :]
        pV = BpV[..., 1, :]
        CV = CqV[..., 0, :]
        qV = CqV[..., 1, :]
        if self.use_length:
            dA = self.bilinear_DNPLR()
            dA_L = power(self.L, dA)
            I = torch.eye(dA.size(-1)).to(dA)
            CV = torch.linalg.solve(I - dA_L.transpose(-1,-2), CV.unsqueeze(-1)).squeeze(-1)
        AV = torch.diag_embed(w) - contract('... n, ... m -> ... n m', pV.conj(), qV)
        return AV, BV, CV

    @torch.no_grad()
    def _check_ABC(self):
        """ Check normal and DNPLR implementations of ABC """
        V = torch.view_as_complex(self.V)
        V = torch.cat([V, V.conj()], dim=-1)
        V_inv = V.transpose(-1, -2).conj()
        AV, BV, CV = self.ABC_DNPLR()
        B = contract('... m, ... m n -> ... n', BV, V_inv)
        C = contract('... m, ... m n -> ... n', CV, V_inv)
        # B = B[..., 0, :]
        # C = C[..., 0, :]
        A = contract('ij, ... jk, lk -> ... il', V, AV, V.conj())

        A_slow, B_slow, C_slow = self.ABC()

        # TODO C is not correct; something goes wrong when the conjugation is combined with use_length aka incorporating (I-dA^L)

        print("Checking ABC reconstruction")
        print(torch.sum((A-A_slow)**2))
        print(torch.sum((B-B_slow)**2))
        print(torch.sum((C-C_slow)**2))

    def dAB(self):
        """ Return the discretized dA, dB transition implicitly represented by this class's representation.

        Returns: (..., N, N) (..., N)
        """

        dt = torch.exp(self.log_dt)
        w = torch.view_as_complex(self.w)
        V = torch.view_as_complex(self.V)
        BV = torch.view_as_complex(self.BV)
        CV = torch.view_as_complex(self.CV)

        V_inv = V.transpose(-1, -2).conj()
        B_ = contract('... r m, ... m n -> ... r n', BV, V_inv)
        B_ = 2*B_.real # (..., 2, N)
        B = B_[..., 0, :]
        p = B_[..., 1:, :]

        C_ = contract('... r m, ... m n -> ... r n', CV, V_inv)
        C_ = 2*C_.real # (..., 2, N)
        # C = CV[..., 0, :]
        q = C_[..., 1:, :]


        return KrylovNPLR.bilinear(torch.exp(self.log_dt), w, V, p, q, B)

    def dAB_DNPLR(self):
        """ Compute dA and dB with the diagonalized representation """
        assert self.rank == 1, "Linear step only implemented for rank 1 case"
        w = torch.view_as_complex(self.w)
        V = torch.view_as_complex(self.V)
        BV = torch.view_as_complex(self.BV)
        CV = torch.view_as_complex(self.CV)
        dt = torch.exp(self.log_dt)
        pV = BV[..., -1, :]
        qV = CV[..., -1, :]
        w = torch.cat([w, w.conj()], dim=-1)
        V = torch.cat([V, V.conj()], dim=-1)
        pV = torch.cat([pV, pV.conj()], dim=-1)
        qV = torch.cat([qV, qV.conj()], dim=-1)


        A_ = torch.diag_embed(2./dt[:, None] + w) - contract('... n, ... m -> ... n m', pV.conj(), qV)
        A_ = .5*dt[:, None, None] * contract('ij, ... jk, lk -> ... il', V, A_, V.conj())
        print("A forwards", A_)

        d = (2./dt.unsqueeze(-1) - w).reciprocal() # (H, N)
        r = 1 + contract('... n, ... n, ... n -> ...', qV, d, pV.conj())
        r = r.reciprocal()
        A_ = torch.diag_embed(d) - contract('... p, ... p, ..., ... q, ... q -> ... p q', d, pV.conj(), r, qV, d)
        A_inv = contract('ij, ... jk, lk -> ... il', V, A_, V.conj())
        A_inv = 2./dt[:, None, None] * A_inv
        print("A backwards", A_inv)

    def dABC(self): # TODO fix for use_length case
        # return torch.zeros(self.CV.size(0), self.w.size(-2), self.w.size(-2)), torch.zeros(self.CV.size(0), self.w.size(-2)),torch.zeros(self.CV.size(0), self.w.size(-2))
        dA, dB = self.dAB()
        dA_L = power(self.L, dA)
        I = torch.eye(dA.size(-1)).to(dA)

        CV = torch.view_as_complex(self.CV)
        V = torch.view_as_complex(self.V)
        C = contract('... r m, ... m n -> ... r n', CV, V.transpose(-1, -2).conj())
        C = 2 * C[..., 0, :].real

        dC = torch.linalg.solve(I - dA_L.transpose(-1,-2), C.unsqueeze(-1)).squeeze(-1)
        return dA, dB, dC

    @torch.no_grad()
    def _double_length(self):
        dA, dB = self.dAB()
        dA_L = power(self.L, dA)
        if _isinf(dA_L):
            # mult = lambda A, B: torch.einsum('... m, ... m n -> ... n', A, B)
            # dA2, dB2 = self.dAB(debug=True)
            breakpoint()
        I = torch.eye(dA.size(-1)).to(dA)

        CV = torch.view_as_complex(self.CV)
        V = torch.view_as_complex(self.V)
        C = contract('... r m, ... m n -> ... r n', CV, V.transpose(-1, -2).conj())
        q = C[..., 1:, :]
        C = 2 * C[..., 0, :].real

        C_ = torch.einsum('... m, ... m n -> ... n', C, I + dA_L)
        C_ = C_.unsqueeze(-2)
        C = torch.cat([C_, q], dim=-2)
        CV = contract('... r n, ... n m -> ... r m', C.to(V), V).contiguous()

        self.CV.copy_(torch.view_as_real(CV))

        self.L *= 2
        # Cache FFT nodes and their "unprocessed" them with the bilinear transform
        nodes = torch.tensor(np.exp(-2j * np.pi / (self.L)), dtype=V.dtype, device=dA.device) # \omega_{2L}
        nodes = nodes ** torch.arange(0, self.L//2+1, device=dA.device)
        z = 2*(1-nodes)/(1+nodes)
        self.register_buffer('nodes', torch.view_as_real(nodes))
        self.register_buffer('z', torch.view_as_real(z))
        self.register_buffer('V', torch.view_as_real(V))

    def double_length(self):
        # k = self.forward()
        self._double_length()
        # k_ = self.forward()[..., :self.L//2]

    @torch.no_grad()
    def forward_resolution(self, rate, L): # TODO deprecate?
        """ Compute the filter at a different resolution than the training one """
        # Get dC
        self._check()
        # A, B, C = self.ABC(precision=precision)
        # A, B, C = self.ABC_DNPLR()
        # B = B.conj()
        # Get new dA and dB at different timescale
        dt = torch.exp(self.log_dt) * rate # self.L / L
        dA, dB = KrylovSlow.bilinear(dt, A, B)
        # print("dA, dB, C", dA, dB, C)

        K = krylov(L, dA, dB, C)

        if _isnan(K) or _isinf(K): breakpoint()
        return K

    def setup_step(self):
        self.dA, self.dB, self.dC = self.dABC()
        self.dA, self.dB, self.dC = self.dA.float() if self.dA is not None else None, \
                                    self.dB.float() if self.dB is not None else None, \
                                    self.dC.float() if self.dC is not None else None

    @torch.no_grad()
    def _check(self):
        """ Check if A, B, C parameters and vanilla Krylov construction can be recovered """

        self._check_ABC()
        # A, B, C = self.ABC(precision=2) # should also work
        A, B, C = self.ABC_DNPLR()
        B = B.conj()
        # Get new dA and dB at different timescale
        dt = torch.exp(self.log_dt)
        dA, dB = KrylovSlow.bilinear(dt, A, B)

        L = 100 if self.L is None else self.L
        K = krylov(L, dA, dB, C)

        diff = K - self.forward()[..., :L]
        # print(diff, torch.sum(diff**2))
        print("checking Krylov construction", torch.sum(diff**2))


    def step(self, u, state):
        next_state = contract('h m n, b h n -> b h m', self.dA, state) + contract('h n, b h -> b h n', self.dB, u)
        y = contract('h n, b h n -> b h', self.dC, next_state)
        return y, next_state

    def setup_step_linear(self):
        w = torch.view_as_complex(self.w)
        V = torch.view_as_complex(self.V)
        BV = torch.view_as_complex(self.BV)
        CV = torch.view_as_complex(self.CV)
        w = torch.cat([w, w.conj()], dim=-1)
        V = torch.cat([V, V.conj()], dim=-1)
        BV = torch.cat([BV, BV.conj()], dim=-1)
        CV = torch.cat([CV, CV.conj()], dim=-1)
        pV = BV[..., -1, :]
        qV = CV[..., -1, :]
        BV = BV[..., 0, :]
        CV = CV[..., 0, :]
        dt = torch.exp(self.log_dt)
        d = (2./dt.unsqueeze(-1) - w).reciprocal() # (H, N)
        r = 1 + contract('... n, ... n, ... n -> ...', qV, d, pV.conj())
        r = r.reciprocal()
        A_f = torch.diag_embed(2./dt[:, None] + w) - contract('... n, ... m -> ... n m', pV.conj(), qV)
        A_b = torch.diag_embed(d) - contract('... p, ... p, ..., ... q, ... q -> ... p q', d, pV.conj(), r, qV, d)
        dA = A_b @ A_f
        dA_L = power(self.L, dA)
        I = torch.eye(dA.size(-1)).to(dA)
        dCV = torch.linalg.solve(I - dA_L.transpose(-1, -2), CV.unsqueeze(-1)).squeeze(-1)

        self.step_params = {
            'd': d,
            'r': r.unsqueeze(-1)*d*qV,
            # 'r': r,
            'dCV': dCV,
            'pV': pV,
            'qV': qV,
            'BV': BV,
            'd1': 2./dt.unsqueeze(-1) + w,
        }
        N = w.shape[-1]
        self.step_params = {k: v[...,:N//2] for k, v in self.step_params.items()}

    def step_linear(self, u, state):
        """ Version of the step function that has time O(N) instead of O(N^2) per step. Unfortunately, as currently implemented it's about 2x slower because it calls several sequential operations. Perhaps a fused CUDA kernel implementation would be much faster """
        d1 = self.step_params['d1'] # (H N)
        pV = self.step_params['pV'] # (H N)
        qV = self.step_params['qV'] # (H N)
        BV = self.step_params['BV'] # (H N)
        r = self.step_params['r']
        d = self.step_params['d'] # (H N)
        dCV = self.step_params['dCV'] # (H N)
        # new_state = contract('... n, ... m, ... m -> ... n', pV.conj(), qV, state) # (B H N)
        new_state = 2*pV.conj() * torch.sum(qV*state, dim=-1, keepdim=True).real # conjugated version
        new_state = d1 * state - new_state
        new_state = new_state + 2. * BV.conj() * u.unsqueeze(-1) # (B H N)
        # A_ = contract('... p, ... q, ... q -> ... p', pV.conj(), r, new_state) # (B H N)
        A_ = 2*pV.conj()*torch.sum(r*new_state, dim=-1, keepdim=True).real # conj version
        new_state = d * (new_state - A_)

        out = 2*contract('... n, ... n -> ...', dCV, new_state).real
        return out, new_state



class KrylovSlow(PreprocessModule):
    """ Slow version of Krylov function for illustration and benchmarking.

    - Caches discretized matrices A^(dt), B^(dt)
    - Computes K_L(A^dt, B^dt, C)

    Usage:
    ```
    krylov = KrylovSlow(L, A, B, C, log_dt)()
    ```
    Result is expected to be equal to KrylovNPLR(L, A, B, C, log_dt, p, q)() for p, q such that A+pq^T is normal
    """
    def _setup(self, L, A, B, C, log_dt, trainable=None, lr=None):
        # super().__init__()
        self.N = A.shape[-1]
        self.L = L
        dA, dB = KrylovSlow.bilinear(torch.exp(log_dt), A, B)

        # Register parameters
        if trainable is None: trainable = DictConfig({'A': 0, 'B': 0, 'C': 0, 'dt': 0})
        if lr is None: lr = DictConfig({'A': None, 'B': None, 'C': None, 'dt': None})
        if trainable is not None and lr is not None:
            repeat = C.size(0)
            self.register('log_dt', log_dt, trainable.dt, lr.dt)
            self.register('dA', dA, trainable.A, lr.A, repeat=repeat)
            self.register('dB', dB, trainable.B, lr.B)
            self.register('C', C, trainable.C, lr.C)

    def forward(self, rate=1.0, state=None):
        k = krylov(self.L, self.dA, self.dB, self.C) # (H L)
        if state is not None: # TODO This doesn't seem correct based on s4.test_state()
            k_state = krylov(self.L, self.dA, state, self.C)
            return k, k_state
        return k

    @classmethod
    def bilinear(cls, dt, A, B=None, separate=False):
        """
        dt: (...) timescales
        A: (... N N)
        B: (... N)
        """
        N = A.shape[-1]
        I = torch.eye(N).to(A)
        A_backwards = I - dt[:, None, None]/2 * A
        A_forwards = I + dt[:, None, None]/2 * A

        if B is None:
            dB = None
        else:
            dB = dt[..., None] * torch.linalg.solve(A_backwards, B.unsqueeze(-1)).squeeze(-1) # (... N)

        if separate:
            A_b = torch.linalg.solve(A_backwards, I) # (... N N)
            return A_forwards, A_b, dB
        else:
            dA = torch.linalg.solve(A_backwards, A_forwards) # (... N N)
            return dA, dB


    def dAB(self):
        """ Return the discretized dA, dB transition implicitly represented by this class's representation.

        Returns: (..., N, N) (..., N)
        """

        return self.dA, self.dB

    def dABC(self):
        return self.dA, self.dB, self.C

    def _cache_all(self): pass
    # def step(self): return self.dABC()
    def setup_step(self):
        self.dC = self.C
    def step(self, u, state):
        next_state = contract('h m n, b h n -> b h m', self.dA, state) + contract('h n, b h -> b h n', self.dB, u)
        y = contract('... n, ... n -> ...', self.dC, next_state)
        return y, next_state

class HippoKrylov(nn.Module):
    """ Wrapper around KrylovNPLR that generates A, B, C, dt according to HiPPO arguments. """

    def __init__(self, N, H, L=None, measure='legs', rank=1, dt_min=0.001, dt_max=0.1, w_bias=0.0, trainable=None, lr=None, slow=False, use_length=True, precision=1):
        super().__init__()
        self.N = N
        self.H = H
        self.L = L
        self.precision = precision
        dtype = torch.double if self.precision == 2 else torch.float

        # Set default trainable and lr parameters
        self.trainable = DictConfig({
            'A': 1,
            'B': 2,
            'C': 1,
            'dt': 1,
        })
        if trainable is not None: self.trainable.update(trainable)
        self.lr = DictConfig({
            'A': 1e-3,
            'B': 1e-3,
            'C': None,
            'dt': 1e-3,
        })
        if lr is not None: self.lr.update(lr)

        # Generate A, B
        A, B = hippo.transition(measure, N)
        A = torch.as_tensor(A, dtype=dtype)
        B = torch.as_tensor(B, dtype=dtype)[:, 0]
        A -= torch.eye(N) * w_bias # Identity correction; stabilizes legt

        # Generate C
        C = torch.randn(self.H, self.N, dtype=dtype)

        # Generate dt
        self.log_dt = torch.rand(self.H, dtype=dtype) * (math.log(dt_max)-math.log(dt_min)) + math.log(dt_min)

        # Compute the preprocessed representation
        self.slow = slow
        if self.slow: # Testing purposes only
            self.krylov = KrylovSlow(L, A, B, C, self.log_dt, trainable=self.trainable, lr=self.lr, use_length=use_length)
        else:
            # Generate low rank correction p for the measure
            p = hippo.rank_correction(measure, N, rank, dtype=dtype)
            # if self.precision == 2: p = p.to(torch.double)
            self.krylov = KrylovNPLR(L, A, B, C, self.log_dt, p, p, trainable=self.trainable, lr=self.lr, use_length=use_length)

        # Cached tensors
        self.K = None
        self.dA = None
        self.dB = None

    def forward(self, state=None, L=None):
        """
        state: (B, H, N)
        """

        if state is not None:
            if not self.slow: state=state.transpose(0,1) # TODO the transpose should probably happen inside krylov
            k, k_state = self.krylov(state=state, L=L) # (B, H, L) (B, H, N)
            k = k.to(torch.float)
            k_state = k_state.to(torch.float)
            if not self.slow: k_state=k_state.transpose(0,1)
            return k, k_state
        else:
            if self.K is None: return self.krylov(L=L).to(torch.float)
            else: return self.K

    def forward_resolution(self, rate, L):
        if self.K is None or L != self.K.size(-1):
            self.K = self.krylov.forward_resolution(rate, L).to(torch.float)
            print("Cached Krylov filter of shape", self.K.shape)
        # return self.K

    @torch.no_grad()
    def next_state(self, state, u):
        """
        state: (..., N)
        u: (..., L)

        Returns: (..., N)
        """

        dA, dB = self.krylov.dAB()
        self.dA = dA
        self.dB = dB

        v = dB.unsqueeze(-1) * u.flip(-1).unsqueeze(-2) # (..., N, L)
        AL, v = power(u.size(-1), dA, v)
        next_state = contract('... m n, ... n -> ... m', AL, state)
        # v = contract('... m n, ... n -> ... m', dA, v) # [21-09-19 AG] TODO I think the math says this line is necessary but the tests say you don't want it...
        next_state = next_state + v
        return next_state

    # def step(self):
    #     """ Return transition matrices for 1 timestep. Must be called after self._cache_all() """
    #     assert self.dA is not None and self.dB is not None
    #     return self.dA, self.dB, self.dC
    def step(self, u, state):
        return self.krylov.step(u, state)

    def _cache(self, rate=1.0, L=None):
        if self.K is None or self.K.size(-1) != L:
            self.K = self.krylov(rate=rate, L=L).to(torch.float)
        return self.K

    def _uncache(self):
        self.K = None

    def _cache_all(self):
        self.K = self.krylov()
        print("caching s4", self.H, self.N, self.L)
        self.dA, self.dB, self.dC = self.krylov.dABC()
    def double_length(self):
        self.krylov.double_length()


def generate_krylov(H, N, L, measure='legs', rank=1):
    A, B = hippo.transition(measure, N)
    A = torch.as_tensor(A, dtype=torch.float)
    B = torch.as_tensor(B, dtype=torch.float)[:, 0]
    print('B', B)
    C = torch.ones(N)
    p = hippo.rank_correction(measure, N, rank=rank)
    print('p', p)
    dt = (1+10*torch.arange(H)/H) * 1/L
    A, p, B, C, dt = utils.convert_data(A, p, B, C, dt, device=device)
    log_dt = torch.log(dt)


    krylov_dense = KrylovSlow(L, A, B, C, log_dt)
    krylov = KrylovNPLR(L, A, B, C, log_dt, p, p)
    krylov_dense.to(device).setup()
    krylov.to(device).setup()

    return krylov_dense, krylov

def benchmark_krylov(measure='legs', rank=1):
    # N = 4
    # L = 4
    # H = 1
    N = 64
    L = 4096
    H = 256

    krylov_dense, krylov = generate_krylov(H, N, L, measure, rank)

    utils.compare_outputs(krylov_dense(), krylov(), full=False, relative=True)

    utils.benchmark_forward(100, krylov_dense, desc='krylov fft manual')
    utils.benchmark_forward(100, krylov, desc='krylov fft rank')
    utils.benchmark_backward(100, krylov_dense, desc='krylov fft manual')
    utils.benchmark_backward(100, krylov, desc='krylov fft rank')

    utils.benchmark_memory(krylov_dense, desc='krylov fft manual')
    utils.benchmark_memory(krylov, desc='krylov fft rank')

def test_bilinear():
    L = 8
    N = 4
    H = 3

    krylov_slow, krylov = generate_krylov(H, N, L, 'legs', 1)

    print("slow dA dB dC", krylov_slow.dABC())
    # print("fast dA dB dC", krylov.dABC())

    A, B, C = krylov.ABC()
    print("ABC", A, B, C)

    print("K slow", krylov_slow.forward())
    print("resolution change", krylov.forward_resolution(1.0, 2*L))

def test_step():
    B = 2
    L = 4
    N = 4
    H = 3

    krylov_slow, krylov = generate_krylov(H, N, L, 'legs', 1)

    print("TESTING SLOW STEP")
    krylov_slow.setup_step()
    state = torch.zeros(B, H, N).to(device)
    u = torch.ones(B, H, L).to(device)
    for u_ in torch.unbind(u, dim=-1):
        y_, state = krylov_slow.step(u_, state=state)
        print("y", y_, y_.shape)

    print("TESTING STEP")
    krylov.setup_step()
    state = torch.zeros(B, H, N).to(device)
    u = torch.ones(B, H, L).to(device)
    for u_ in torch.unbind(u, dim=-1):
        y_, state = krylov.step(u_, state=state)
        print("y", y_, y_.shape)
        # print("state", state, state.shape)

    print("TESTING LINEAR STEP")
    krylov.setup_step_linear()
    # state = torch.zeros(B, H, N).to(device).to(torch.cfloat)
    state = torch.zeros(B, H, N//2).to(device).to(torch.cfloat)
    u = torch.ones(B, H, L).to(device)
    for u_ in torch.unbind(u, dim=-1):
        y_, state = krylov.step_linear(u_, state=state)
        print("y", y_, y_.shape)
        # print("state", state, state.shape)

@torch.inference_mode()
def benchmark_step():
    B = 8192
    L = 16
    N = 64
    H = 1024

    _, krylov = generate_krylov(H, N, L, 'legs', 1)

    print("Benchmarking Step")
    krylov.setup_step()
    state = torch.zeros(B, H, N).to(device)
    u = torch.ones(B, H).to(device)
    utils.benchmark_forward(16, krylov.step, u, state, desc='normal step')

    print("Benchmarking Linear Step")
    krylov.setup_step_linear()
    # state = torch.zeros(B, H, N).to(device).to(torch.cfloat)
    state = torch.zeros(B, H, N//2).to(device).to(torch.cfloat)
    u = torch.ones(B, H).to(device)
    utils.benchmark_forward(16, krylov.step_linear, u, state, desc='normal step')


def test_double():
    torch.set_printoptions(sci_mode=False, linewidth=160)
    L = 8
    N = 4
    H = 3

    krylov_slow, krylov = generate_krylov(H, N, L, 'legs', 1)

    k = krylov.forward()
    print(k, k.shape)

    krylov.double_length()
    k = krylov.forward()
    print(k, k.shape)


def test_state():
    import models.functional as F
    B = 1
    N = 4
    L = 4
    H = 1
    krylov_dense, krylov = generate_krylov(H, N, L)

    state = torch.randn(B, H, N).to(device)
    k, k_state = krylov_dense.forward(state=state)
    # k_state = k_state.transpose(0,1)
    print("k slow", k)
    print("k_state slow", k_state)

    k, k_state = krylov.forward(state=state.transpose(0,1))
    k_state = k_state.transpose(0,1)
    print("k", k)
    print("k_state", k_state)


if __name__ == '__main__':
    from benchmark import utils

    device = 'cuda' # 'cpu'
    device = torch.device(device)

    # benchmark_krylov(measure='legs', rank=1)
    test_bilinear()
    # test_step()
    # benchmark_step()
    # test_state()
    # test_double()
