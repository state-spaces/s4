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

    This class is a verbatim translation of the original code.
    A few variable names have been changed to be more consistent with this codebase.

    The only semantic change is removing the final SiLU activation,
    which is handled by the caller class (e.g. MegaBlock).
    """

    def __init__(
        self,
        d_model,
        d_state=2,
        bidirectional=False,
        l_max=None,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.bidirectional = bidirectional
        self.l_max = l_max
        self.scale = math.sqrt(1.0 / self.d_state)

        H = 2 * d_model if self.bidirectional else d_model
        # This is a state-space based on S4(D) where
        # delta, alpha, beta, gamma, omega simply correspond to
        # the \Delta, A, B, C, D parameters of SSMs
        self.delta = nn.Parameter(torch.Tensor(H, d_state, 1))
        self.alpha = nn.Parameter(torch.Tensor(H, d_state, 1))
        self.beta = nn.Parameter(torch.Tensor(H, d_state, 1))
        self.gamma = nn.Parameter(torch.Tensor(H, d_state))
        self.omega = nn.Parameter(torch.Tensor(d_model))
        self._kernel = None
        self._coeffs = None

        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def reset_parameters(self):
        with torch.no_grad():
            # delta & alpha
            nn.init.normal_(self.delta, mean=0.0, std=0.2)
            nn.init.normal_(self.alpha, mean=0.0, std=0.2)
            # beta [1, -1, 1, -1, ...] seems more stable.
            val = torch.ones(self.d_state, 1)
            if self.d_state > 1:
                idx = torch.tensor(list(range(1, self.d_state, 2)))
                val.index_fill_(0, idx, -1.0)
            self.beta.normal_(mean=0.0, std=0.02).add_(val)
            # gamma & omega
            nn.init.normal_(self.gamma, mean=0.0, std=1.0)
            nn.init.normal_(self.omega, mean=0.0, std=1.0)

    def _calc_coeffs(self):
        self._coeffs = None
        # D x N x 1
        p = torch.sigmoid(self.delta)
        alpha = torch.sigmoid(self.alpha)
        q = 1.0 - p * alpha
        return p, q

    def _compute_kernel(self, L: int):
        self._kernel = None
        # D x N x 1
        p, q = self._calc_coeffs()
        # D x N x L
        vander = torch.arange(L).to(p).view(1, 1, L) * torch.log(q)
        kernel = (p * self.beta) * torch.exp(vander)
        # D x L
        return torch.einsum('dnl,dn->dl', kernel, self.gamma * self.scale)

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

    def step(self, x, length, hx=None):
        if length == 1:
            return self.one_step(x, hx=hx)

        # D x N x 1
        p, q = self.coeffs()
        # D x N x L+1
        vander = torch.arange(length + 1).to(p).view(1, 1, length + 1) * torch.log(q)
        vander = torch.exp(vander)
        if hx is not None:
            # D x N x L * D x N x 1 -> D x N x L
            k = vander[:, :, 1:] * (self.gamma * self.scale).unsqueeze(-1)
            ox = torch.einsum('bdn,dnl->bdl', hx, k)
            # D x N * B x D x N -> B x D x N
            hh = vander[:, :, -1] * hx
        else:
            ox = None
            hh = None

        # D x N x L
        vander = vander[:, :, :-1]
        kernel = (p * self.beta) * vander
        k = torch.einsum('dnl,dn->dl', kernel, self.gamma * self.scale)

        k_f = torch.fft.rfft(k.float(), n=2 * length)
        x_f = torch.fft.rfft(x.float(), n=2 * length)
        # B x D x L
        out = torch.fft.irfft(x_f * k_f, n=2 * length)[..., 0:length]
        out = out.type_as(x)
        if ox is not None:
            out = out + ox

        h = torch.einsum('bdl,dnl->bdn', x, torch.flip(kernel, dims=[2]))
        if hh is not None:
            h = h + hh
        # L x B x D, B x D x N
        return out.permute(2, 0, 1), h

    def one_step(self, x, hx=None):
        p, q = self.coeffs()
        # (D x N) x (B x D x 1) -> B x D x N
        h = (p * self.beta).squeeze(-1) * x
        if hx is not None:
            h = h + q.squeeze(-1) * hx
        # B x D
        out = torch.einsum('bdn,dn->bd', h, self.gamma * self.scale)
        # 1 x B x D, B x D x N
        return out.unsqueeze(0), h

    def forward(
        self,
        x,
        padding_mask: Optional[torch.Tensor] = None,
        state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
    ) -> Tuple[torch.Tensor, None]:
        """Input shape: Time x Batch x Channel
        Args:
            padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
        """

        bsz, seq_len, d_model = x.size()
        assert d_model == self.d_model

        residual = x * self.omega

        # L x B x D -> B x D x L
        # x = x.permute(1, 2, 0)
        x = x.transpose(-1, -2)  # (B D L)
        if padding_mask is not None:
            x = x * (1.0 - padding_mask.unsqueeze(1).type_as(x))

        assert not self.bidirectional or state is None, 'Bidirectional EMA does not support incremental state'
        if state is not None:
            saved_state = self._get_input_buffer(state)
            if 'prev_state' in saved_state:
                h = saved_state['prev_state']
            else:
                h = None
            out, h = self.step(x, seq_len, hx=h)
            saved_state['prev_state'] = h
            self._set_input_buffer(state, saved_state)
            # B x D -> 1 x B x D
            # out = F.silu(out + residual)
            out = out + residual
        else:
            # D x L
            k = self.kernel(seq_len)
            fft_len = seq_len
            s = 0
            l_kernel = k.size(1)
            assert l_kernel == seq_len
            if self.bidirectional:
                k1, k2 = torch.split(k, [self.d_model, self.d_model], dim=0)
                # D x 2*L-1
                k = F.pad(k1, (l_kernel - 1, 0)) + F.pad(k2.flip(-1), (0, l_kernel - 1))
                x = F.pad(x, (l_kernel - 1, 0))
                fft_len = fft_len + l_kernel - 1
                s = 2 * l_kernel - 2

            k_f = torch.fft.rfft(k.float(), n=2 * fft_len)
            x_f = torch.fft.rfft(x.float(), n=2 * fft_len)
            # B x D x L
            out = torch.fft.irfft(x_f * k_f, n=2 * fft_len)[..., s:s + seq_len]
            out = out.type_as(x)
            # B x D x L -> L x B x D
            # out = F.silu(out.transpose(-1, -2) + residual)
            out = out.transpose(-1, -2) + residual

        return out, None  # empty state

    def _get_input_buffer(self, state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]]) -> Dict[str, Optional[torch.Tensor]]:
        result = self.get_state(state, "ema_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[torch.Tensor]] = {}
            return empty_result

    def _set_input_buffer(self, state: Dict[str, Dict[str, Optional[torch.Tensor]]], buffer: Dict[str, Optional[torch.Tensor]]):
        return self.set_state(state, "ema_state", buffer)

    @torch.jit.export
    def reorder_state(
            self, state: Dict[str, Dict[str, Optional[torch.Tensor]]], new_order: torch.Tensor
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            state = self._set_input_buffer(state, input_buffer)
        return state

    def extra_repr(self) -> str:
        return 'edim={}, d_state={}, bidirectional={}, trunction={}'.format(self.d_model, self.d_state, self.bidirectional, self.l_max)
