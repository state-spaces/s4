import datetime
import math
from typing import ForwardRef

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

import src.models.nn.utils as U
import src.utils as utils
import src.utils.config
from src.models.sequence.block import SequenceResidualBlock
from src.models.nn.components import Normalization

class Encoder(nn.Module):
    """Encoder abstraction
    Accepts a tensor and optional kwargs. Outside of the main tensor, all other arguments should be kwargs.
    Returns a tensor and optional kwargs.
    Encoders are combined via U.PassthroughSequential which passes these kwargs through in a pipeline. The resulting kwargs are accumulated and passed into the model backbone.

    """

    def forward(self, x, **kwargs):
        """
        x: input tensor
        *args: additional info from the dataset (e.g. sequence lengths)

        Returns:
        y: output tensor
        *args: other arguments to pass into the model backbone
        """
        return x, {}



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

    def __init__(self, d_model, dropout=0.1, max_len=16384, pe_init=None):
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
            self.register_buffer("pe", pe)

        self.attn_mask = None

    def forward(self, x):
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

        x = x + self.pe[: x.size(-2)]
        return self.dropout(x)


class ClassEmbedding(Encoder):
    # Should also be able to define this by subclassing Embedding
    def __init__(self, n_classes, d_model):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, d_model)

    def forward(self, x, y):
        x = x + self.embedding(y).unsqueeze(-2)  # (B, L, D)
        return x


class Conv1DEncoder(Encoder):
    def __init__(self, d_input, d_model, kernel_size=25, stride=1, padding='same'):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_input,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        # BLD -> BLD
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return x

class LayerEncoder(Encoder):
    """Use an arbitary SequenceModule layer"""

    def __init__(self, d_model, prenorm=False, norm='layer', layer=None):
        super().__init__()

        # Simple stack of blocks
        layer["transposed"] = False
        self.layer = SequenceResidualBlock(
            d_input=d_model,
            prenorm=prenorm,
            layer=layer,
            residual='R',
            norm=norm,
            pool=None,
        )

    def forward(self, x):
        x, _ = self.layer(x) # Discard state
        return x


class TimestampEmbeddingEncoder(Encoder):
    """
    General time encoder for Pandas Timestamp objects (encoded as torch tensors).
    See MonashDataset for an example of how to return time features as 'z's.
    """

    cardinalities = {
        'day': (1, 31),
        'hour': (0, 23),
        'minute': (0, 59),
        'second': (0, 59),
        'month': (1, 12),
        'year': (1950, 2010), # (1800, 3000) used to be (1970, datetime.datetime.now().year + 1) but was not enough for all datasets in monash
        'dayofweek': (0, 6),
        'dayofyear': (1, 366),
        'quarter': (1, 4),
        'week': (1, 53),
        'is_month_start': (0, 1),
        'is_month_end': (0, 1),
        'is_quarter_start': (0, 1),
        'is_quarter_end': (0, 1),
        'is_year_start': (0, 1),
        'is_year_end': (0, 1),
        'is_leap_year': (0, 1),
    }

    def __init__(self, d_model, table=False, features=None):
        super().__init__()
        self.table = table
        self.ranges = {k: max_val - min_val + 2 for k, (min_val, max_val) in self.cardinalities.items()} # padding for null included

        if features is None:
            pass
        else:
            self.cardinalities = {k: v for k, v in self.cardinalities.items() if k in features}

        if table:
            self.embedding = nn.ModuleDict({
                attr: nn.Embedding(maxval - minval + 2, d_model, padding_idx=0)
                for attr, (minval, maxval) in self.cardinalities.items()
            })
        else:
            self.embedding = nn.ModuleDict({
                attr: nn.Linear(1, d_model)
                for attr in self.cardinalities
            })



    def forward(self, x, timestamps=None):
        for attr in timestamps:
            mask = timestamps[attr] == -1
            timestamps[attr] = timestamps[attr] - self.cardinalities[attr][0]
            timestamps[attr][mask] = 0
            if self.table:
                x = x + self.embedding[attr](timestamps[attr].to(torch.long))
            else:
                x = x + self.embedding[attr]((2 * timestamps[attr] / self.ranges[attr] - 1).unsqueeze(-1))

            #x = x + self.embedding(timestamps[attr].to(torch.float)).unsqueeze(1)
        return x


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

    def forward(self, x, mark=None, mask=None):
        assert mark is not None and mask is not None, "Extra arguments should be returned by collate function"
        if self.timeenc == 0:
            assert mark.size(-1) == len(self.encoders)
            embeddings = [
                embed(z) for embed, z in zip(self.encoders, torch.unbind(mark, dim=-1))
            ]
            time_encode = torch.sum(torch.stack(embeddings), dim=0)
        else:
            time_encode = self.encoders(mark)
        mask_encode = self.mask_embed(mask.squeeze(-1))
        return x + time_encode + mask_encode  # (B, L, d_model)


class PackedEncoder(Encoder):
    def forward(self, x, len_batch=None):
        assert len_batch is not None
        x = nn.utils.rnn.pack_padded_sequence(
            x, len_batch.cpu(), enforce_sorted=False, batch_first=True,
        )
        return x


class OneHotEncoder(Encoder):
    def __init__(self, n_tokens, d_model):
        super().__init__()
        assert n_tokens <= d_model
        self.d_model = d_model

    def forward(self, x):
        return F.one_hot(x.squeeze(-1), self.d_model).float()


class Conv2DPatchEncoder(Encoder):

    """
    For encoding images into a sequence of patches.
    """

    def __init__(self, d_input, d_model, filter_sizes, flat=False):

        """
        d_input: dim of encoder input (data dimension)
        d_model: dim of encoder output (model dimension)
        filter_sizes: tuple with fh, fw
        flat: if image is flattened from dataloader (like in cifar),
            then we need to reshape back to 2D before conv
        """

        fh, fw = filter_sizes
        self.flat = flat

        super().__init__()
        assert len(filter_sizes) == 2

        self.encoder = nn.Conv2d(d_input, d_model, kernel_size=(fh, fw), stride=(fh, fw))

    def forward(self, x):

        """
        x shape expected = [b, h, w, c]
        returns tuple with x, with new shape = [b, seq_len, c_out]

        """

        x = rearrange(x, 'b h w c -> b c h w')
        x = self.encoder(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x


# For every type of encoder/decoder, specify:
# - constructor class
# - list of attributes to grab from dataset
# - list of attributes to grab from model

registry = {
    "stop": Encoder,
    "id": nn.Identity,
    "embedding": nn.Embedding,
    "linear": nn.Linear,
    "position": PositionalEncoder,
    "class": ClassEmbedding,
    "pack": PackedEncoder,
    "time": TimeEncoder,
    "onehot": OneHotEncoder,
    "conv1d": Conv1DEncoder,
    "patch2d": Conv2DPatchEncoder,
    "timestamp_embedding": TimestampEmbeddingEncoder,
    "layer": LayerEncoder,
}
dataset_attrs = {
    "embedding": ["n_tokens"],
    "linear": ["d_input"],  # TODO make this d_data?
    "class": ["n_classes"],
    "time": ["n_tokens_time"],
    "onehot": ["n_tokens"],
    "conv1d": ["d_input"],
    "patch2d": ["d_input"],
}
model_attrs = {
    "embedding": ["d_model"],
    "linear": ["d_model"],
    "position": ["d_model"],
    "class": ["d_model"],
    "time": ["d_model"],
    "onehot": ["d_model"],
    "conv1d": ["d_model"],
    "patch2d": ["d_model"],
    "timestamp_embedding": ["d_model"],
    "layer": ["d_model"],
}


def _instantiate(encoder, dataset=None, model=None):
    """Instantiate a single encoder"""
    if encoder is None:
        return None
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
    return U.PassthroughSequential(
        *[_instantiate(e, dataset=dataset, model=model) for e in encoder]
    )
