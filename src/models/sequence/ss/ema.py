# Adapted from https://github.com/facebookresearch/mega/blob/ea355255149d38ffe16bf2c176d47c3864e8b05a/fairseq/modules/exponential_moving_average.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadEMA(nn.Module):
    """Exponential Moving Average Layer.

    This class is a verbatim translation of the original code with minor differences that
    do not change the code execution path.

    - A few variable names have been changed to be more consistent with this codebase.

    - State passing is not supported ("incremental_state" in the Mega code),
      as the original module uses a different fairseq interface than this codebase.

    - The only semantic change is removing the final SiLU activation,
      which is handled by the caller module (e.g. src.models.sequence.mega.MegaBlock).
    """

    def __init__(
        self,
        d_model,
        d_state=2,
        bidirectional=False,
        l_max=None,
    ):
        super().__init__()

        self.H = d_model
        self.N = d_state
        self.bidirectional = bidirectional
        self.l_max = l_max
        self.scale = math.sqrt(1.0 / self.N)

        H = 2 * self.H if self.bidirectional else self.H

        # This is a state-space model variant of S4(D) where
        # delta, alpha, beta, gamma, omega directly correspond to
        # the \Delta, A, B, C, D parameters of SSMs
        self.delta = nn.Parameter(torch.Tensor(H, self.N, 1))
        self.alpha = nn.Parameter(torch.Tensor(H, self.N, 1))
        self.beta = nn.Parameter(torch.Tensor(H, self.N))
        self.gamma = nn.Parameter(torch.Tensor(H, self.N))
        self.omega = nn.Parameter(torch.Tensor(self.H))
        self._kernel = None
        self._coeffs = None

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            # delta & alpha (dt and A parameters of SSM)
            nn.init.normal_(self.delta, mean=0.0, std=0.2)
            nn.init.normal_(self.alpha, mean=0.0, std=0.2)
            # Mega: beta [1, -1, 1, -1, ...] seems more stable.
            val = torch.ones(self.N)
            if self.N > 1:
                idx = torch.tensor(list(range(1, self.N, 2)))
                val.index_fill_(0, idx, -1.0)
            self.beta.normal_(mean=0.0, std=0.02).add_(val)
            # gamma & omega (C and D parameters of SSM)
            # should be unit variance, as specified in HTTYH
            nn.init.normal_(self.gamma, mean=0.0, std=1.0)
            nn.init.normal_(self.omega, mean=0.0, std=1.0)

    def _calc_coeffs(self):
        self._coeffs = None
        p = torch.sigmoid(self.delta)  # (H N 1)
        alpha = torch.sigmoid(self.alpha)
        q = 1.0 - p * alpha
        return p, q

    def _compute_kernel(self, L: int):
        self._kernel = None
        # Materialize parameters - analog of SSM discretization
        p, q = self._calc_coeffs()  # (H N 1)

        vander = torch.log(q) * torch.arange(L).to(p).view(1, 1, L)  # (H N L)
        kernel = p[..., 0] * self.beta * self.gamma * self.scale
        return torch.einsum('dn,dnl->dl', kernel, torch.exp(vander))  # (H L)

    def coeffs(self):
        if self.training:
            return self._calc_coeffs()
        else:
            if self._coeffs is None:
                self._coeffs = self._calc_coeffs()
            return self._coeffs

    def kernel(self, L: int):
        L = L if self.l_max is None else min(self.l_max, L)
        if self.training:
            return self._compute_kernel(L)
        else:
            if self._kernel is None or self._kernel.size(-1) < L:
                self._kernel = self._compute_kernel(L)
            return self._kernel[..., :L]

    def forward(
        self,
        u,
        state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        """Input shape: Time x Batch x Channel
        Args:
            padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
        """

        B, L, H = u.size()
        assert H == self.H

        u = u.transpose(-1, -2)  # (B H L)
        if padding_mask is not None:
            u = u * (1.0 - padding_mask.unsqueeze(1).type_as(u))

        # assert not self.bidirectional or state is None, 'Bidirectional EMA does not support incremental state'
        if state is not None:
            raise NotImplementedError(
                "MultiHeadEMA module does not support state passing in this repository."
                "Use S4D for more functionality such as state passing and better performance."
            )
        else:
            k = self.kernel(L)  # (H L)
            l_fft = L
            s = 0
            l_kernel = k.size(1)
            assert l_kernel == L
            u_ = u
            if self.bidirectional:
                # This is twice as inefficient as it could be
                # See S4 FFT conv bidirectional implementation for improvement
                k1, k2 = torch.split(k, [self.H, self.H], dim=0)
                k = F.pad(k1, (l_kernel - 1, 0)) + F.pad(k2.flip(-1), (0, l_kernel - 1))  # (H 2*L-1)
                u_ = F.pad(u, (l_kernel - 1, 0))
                l_fft = l_fft + l_kernel - 1
                s = 2 * l_kernel - 2

            k_f = torch.fft.rfft(k.float(), n=2 * l_fft)
            u_f = torch.fft.rfft(u_.float(), n=2 * l_fft)
            y = torch.fft.irfft(u_f * k_f, n=2 * l_fft)[..., s:s + L]  # (B H L)
            y = y.type_as(u)
            y = y + u * self.omega.unsqueeze(-1)  # (B H L)
            y = y.transpose(-1, -2)

        return y, None  # empty state

    def extra_repr(self) -> str:
        return 'edim={}, N={}, bidirectional={}, trunction={}'.format(self.H, self.N, self.bidirectional, self.l_max)
