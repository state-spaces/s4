""" Ported implementation of WaveGAN Discriminator https://github.com/chrisdonahue/wavegan

Several modifications have been made to integrate this better with this codebase, and to add extra options.

DEPRECATED as of July 22 (V3 release); this type of generic ConvNet is subsumed by the standard model backbone and Conv1d layer (see config convnet1d.yaml)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from einops import rearrange, reduce, repeat

from src.models.sequence import SequenceModule
from src.models.nn.components import Normalization

class ResidualBlock(nn.Module):
    def __init__(self, d, layer, norm='none', dropout=0.0):
        super().__init__()
        self.d = d
        self.layer = layer
        self.norm = Normalization(d, transposed=True, _name_=norm)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.layer(x)
        y = F.leaky_relu(y, negative_slope=0.2)
        y = self.drop(y)
        y = x+y
        y = self.norm(y)
        return y

class Conv1DBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        stride=4,
        # padding=12,
        # alpha=0.2,
        n_layers=1,
        norm='none',
        dropout=0,
    ):
        super().__init__()
        layers = []
        # Residual convolution layers
        padding = (kernel_size-1)//2
        for _ in range(n_layers-1):
            layers.append(ResidualBlock(
                input_channels,
                nn.Conv1d(input_channels, input_channels, kernel_size, stride=1, padding=padding),
                norm=norm,
                dropout=dropout,
            ))

        # Final non-residual conv layer with channel upsizing
        layers.append(nn.Conv1d(
            input_channels, output_channels, kernel_size, stride=stride, padding=padding,
        ))
        layers.append(
            Normalization(output_channels, True, norm)
            # nn.BatchNorm1d(output_channels)
            # if use_batch_norm
            # else nn.Identity()
        )
        # self.alpha = alpha
        layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class WaveGANDiscriminator(SequenceModule):
    def __init__(
        self,
        d_model=1,
        d_output=35,
        l_output=0, # Unused, absorbs argument from sequence
        model_size=64,
        n_layers=1,
        n_blocks=5,
        kernel_size=25,
        # alpha=0.2,
        norm='none',
        pool=False,
        verbose=False,
        l_max=16384,
        # use_batch_norm=False,
        dropout=0.0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, f"Only odd kernel sizes supported"

        self.d_model = d_model  # c
        self.d_output = d_output
        # assert l_output == 0, "WaveGAN Discriminator should only be used on classification tasks with l_output == 0"
        self.l_output = l_output

        self.l_max = 2 ** math.ceil(math.log2(l_max))

        self.model_size = model_size  # d
        self.pool = pool
        self.verbose = verbose

        conv_layers = [
            Conv1DBlock(
                d_model,
                model_size,
                kernel_size,
                stride=4,
                # padding=12,
                # use_batch_norm=use_batch_norm,
                norm=norm,
                n_layers=n_layers,
                # alpha=alpha,
                dropout=dropout,
            )
        ]
        for _ in range(1, n_blocks):
            conv_layers.append(
            Conv1DBlock(
                model_size,
                2 * model_size,
                kernel_size,
                stride=4,
                # padding=12,
                # use_batch_norm=use_batch_norm,
                norm=norm,
                n_layers=n_layers,
                # alpha=alpha,
                dropout=dropout,
            )
            )
            model_size *= 2
        self.conv_layers = nn.ModuleList(conv_layers)

        if pool:
            self.fc = nn.Linear(model_size, self.d_output)
        else:
            # self.fc_d_input = self.l_max // 64 * model_size
            self.fc_d_input = self.l_max // 4**(n_blocks) * model_size # total length * channels after all conv layers
            self.fc1 = nn.Linear(self.fc_d_input, self.d_output)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x, *args, **kwargs):
        """
        x: (batch, length, channels)
        y: (batch, 1, d_output)
        """
        x = x.permute(0, 2, 1)
        x = F.pad(x, (0, self.l_max-x.shape[-1]))
        for conv in self.conv_layers:
            x = conv(x)
            if self.verbose:
                print(x.shape)
        assert self.l_output == 0
        if self.pool:
            x = reduce(x, 'b c l -> b c', 'mean')
            x = self.fc(x)
        else:
            x = x.reshape(-1, self.fc_d_input)
            x = self.fc1(x)
        return x, None
