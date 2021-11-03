""" Wrapper around nn.MultiheadAttention to adhere to SequenceModule interface. """

import torch
from torch import nn
from models.sequence.base import SequenceModule
import src.models.nn.utils as U

class MultiheadAttention(SequenceModule):
    """ Simple wrapper for MultiheadAttention """
    def __init__(self, d_model, n_heads, *args, causal=True, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_output = d_model
        self.mha = nn.MultiheadAttention(d_model, n_heads, *args, batch_first=True, **kwargs)
        self.causal = causal

    def forward(self, src, attn_mask=None, key_padding_mask=None, state=None, **kwargs):
        """ state should represent a mask and key padding mask """
        if self.causal and attn_mask is None:
            attn_mask = torch.triu(torch.ones(src.size(-2), src.size(-2),
                                              dtype=torch.bool, device=src.device),
                                       diagonal=1)
        # attn_mask, key_padding_mask = state
        # Note that this returns None for the second argument
        y, z = self.mha(src, src, src, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False, **kwargs)
        return y, None

    def step(self, x, state):
        # TODO proper cached inference
        # x: (B, D)
        y, z = self.mha(src, src, src, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False, **kwargs)

MultiheadAttention = U.Transpose(MultiheadAttention)
