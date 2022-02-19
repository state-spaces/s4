import math
from typing import ForwardRef

import torch
from torch import nn
import torch.nn.functional as F

import src.models.nn.utils as U
import src.utils as utils
import src.utils.config


class Encoder(nn.Module):
    """This class doesn't do much but just signals the interface that Encoder are expected to adhere to
    TODO: is there a way to enforce the signature of the forward method?

    NOTE: all encoders return a *tuple* where the first argument is a tensor and the rest are additional parameters to be passed into the model
    """

    def forward(self, x, *args):
        """
        x: input tensor
        *args: additional info from the dataset (e.g. sequence lengths)

        Returns:
        y: output tensor
        *args: other arguments to pass into the model backbone
        """
        return (x,)


# Adapted from https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class PositionalEncoder(Encoder):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoder(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=16384, pe_init=None, causal=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        if pe_init is not None:
            self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))
            nn.init.normal_(self.pe, 0, pe_init)
            # self.pe = pe.unsqueeze(1)
        else:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0.0, max_len).unsqueeze(1)
            div_term = torch.exp(
                -math.log(10000.0) * torch.arange(0.0, d_model, 2.0) / d_model
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            # pe = pe.unsqueeze(1) # Comment this out to handle (B, L, D) instead of (L, B, D)
            self.register_buffer("pe", pe)

        self.attn_mask = None

    def forward(self, x, seq_len=None, *args, **kwargs):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
            lens: actual lengths of sequences
        Shape:
            x: [l_sequence, n_batch, d_model]
            Returns: [l_sequence, n_batch, d_model]
            attn_mask: [l_sequence, l_sequence]
            padding_mask:
        """

        # TODO currently not used, but maybe will be someday
        # e.g. attn_mask is defined directly in each attention layer

        # if self.attn_mask is None or self.attn_mask.shape[-1] != x.size(-2):
        #     # self.attn_mask = TriangularCausalMask(len(src), device=src.device)
        #     self.attn_mask = torch.triu(torch.ones(x.size(-2), x.size(-2),
        #                                           dtype=torch.bool, device=x.device),
        #                                diagonal=1)

        # padding_mask = None
        # if seq_len is not None and seq_len < x.size(-2):
        #     padding_mask = LengthMask(
        #         torch.full(
        #             (x.size(-2),),
        #             seq_len,
        #             device=x.device,
        #             dtype=torch.long,
        #         ),
        #         max_len=x.size(-2),
        #     )
        # else:
        #     padding_mask = None

        x = x + self.pe[: x.size(-2)]
        # return self.dropout(x), self.attn_mask, padding_mask
        return (self.dropout(x),)


class ClassEmbedding(Encoder):
    # Should also be able to define this by subclassing TupleModule(Embedding)
    def __init__(self, n_classes, d_model):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, d_model)

    def forward(self, x, y, *args, **kwargs):
        x = x + self.embedding(y).unsqueeze(-2)  # (B, L, D)
        return (x,)


class Conv1DEncoder(Encoder):
    def __init__(self, d_input, d_model, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_input,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x, *args):
        # BLD -> BLD
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return (x,)


class TimeEncoder(Encoder):
    def __init__(self, n_tokens_time, d_model, timeenc=0):
        super().__init__()

        self.timeenc = timeenc
        if self.timeenc == 0:
            self.encoders = nn.ModuleList(
                [nn.Embedding(v, d_model) for v in n_tokens_time]
            )
        else:
            self.encoders = nn.Linear(len(n_tokens_time), d_model)
        self.mask_embed = nn.Embedding(2, d_model)

    def forward(self, x, mark, mask, *args, **kwargs):
        if self.timeenc == 0:
            assert mark.size(-1) == len(self.encoders)
            embeddings = [
                embed(z) for embed, z in zip(self.encoders, torch.unbind(mark, dim=-1))
            ]
            time_encode = torch.sum(torch.stack(embeddings), dim=0)
        else:
            time_encode = self.encoders(mark)
        mask_encode = self.mask_embed(mask.squeeze(-1))
        return (x + time_encode + mask_encode,)  # (B, L, d_model)


class PackedEncoder(Encoder):
    def forward(self, x, len_batch=None):
        assert len_batch is not None
        x = nn.utils.rnn.pack_padded_sequence(
            x,
            len_batch.cpu(),
            enforce_sorted=False,
            batch_first=True,
        )
        return (x,)


class OneHotEncoder(Encoder):
    def __init__(self, n_tokens, d_model):
        super().__init__()
        assert n_tokens <= d_model
        self.d_model = d_model

    def forward(self, x, *args, **kwargs):
        return (F.one_hot(x.squeeze(-1), self.d_model).float(),)


# For every type of encoder/decoder, specify:
# - constructor class
# - list of attributes to grab from dataset
# - list of attributes to grab from model

registry = {
    "id": U.Identity,
    "embedding": U.Embedding,
    "linear": U.Linear,
    "position": PositionalEncoder,
    "class": ClassEmbedding,
    "pack": PackedEncoder,
    "time": TimeEncoder,
    "onehot": OneHotEncoder,
    "conv1d": Conv1DEncoder,
}
dataset_attrs = {
    "embedding": ["n_tokens"],
    "linear": ["d_input"],  # TODO make this d_data?
    "class": ["n_classes"],
    "time": ["n_tokens_time"],
    "onehot": ["n_tokens"],
    "conv1d": ["d_input"],
}
model_attrs = {
    "embedding": ["d_model"],
    "linear": ["d_model"],
    "position": ["d_model"],
    "class": ["d_model"],
    "time": ["d_model"],
    "onehot": ["d_model"],
    "conv1d": ["d_model"],
}


def _instantiate(encoder, dataset=None, model=None):
    """Instantiate a single encoder"""
    if encoder is None:
        return U.Identity()
    if isinstance(encoder, str):
        name = encoder
    else:
        name = encoder["_name_"]

    # Extract dataset/model arguments from attribute names
    dataset_args = utils.config.extract_attrs_from_obj(
        dataset, *dataset_attrs.get(name, [])
    )
    model_args = utils.config.extract_attrs_from_obj(model, *model_attrs.get(name, []))

    # Instantiate encoder
    obj = utils.instantiate(registry, encoder, *dataset_args, *model_args)
    return obj


def instantiate(encoder, dataset=None, model=None):
    encoder = utils.to_list(encoder)
    return U.TupleSequential(
        *[_instantiate(e, dataset=dataset, model=model) for e in encoder]
    )
