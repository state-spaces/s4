""" Ported implementation of WaveGAN Discriminator https://github.com/chrisdonahue/wavegan

Several modifications have been made to integrate this better with this codebase, and to add extra options.
"""

import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.utils.data
# from params import *

from src.models.sequence import SequenceModule
from src.models.nn.components import Normalization

class ResidualBlock(nn.Module):
    def __init__(self, d, layer, norm='none', dropout=0.0):
        super().__init__()
        self.d = d
        self.layer = layer
        self.norm = Normalization(d, transposed=True, _name_=norm)
        self.drop = nn.Dropout2d(dropout)

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
        causal=True,
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
        # padding = (kernel_size-1, 0) if causal else (kernel_size-1)//2
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
        layers.append(nn.Dropout2d(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class WaveGANDiscriminator(SequenceModule):
    def __init__(
        self,
        d_model=1,
        d_output=10,
        l_output=0, # Unused, absorbs argument from sequence
        model_size=64,
        n_layers=1,
        kernel_size=25,
        # alpha=0.2,
        norm='none',
        causal=True, # Currently doesn't work
        verbose=False,
        l_max=16384,
        # use_batch_norm=False,
        dropout=0.0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, f"Only odd kernel sizes supported"
        # assert l_max in [16384, 32768, 65536]  # used to predict longer utterances
        # assert l_max == 16384 # only support up to 16k sequences for now

        self.d_model = d_model  # c
        self.d_output = d_output
        # assert l_output == 0, "WaveGAN Discriminator should only be used on classification tasks with l_output == 0"
        self.l_output = l_output

        self.l_max = 2 ** math.ceil(math.log2(l_max))
        print(self.l_max)

        self.model_size = model_size  # d
        # self.use_batch_norm = use_batch_norm
        # self.alpha = alpha
        self.verbose = verbose

        conv_layers = [
            Conv1DBlock(
                d_model,
                model_size,
                kernel_size,
                stride=4,
                # padding=12,
                # use_batch_norm=use_batch_norm,
                causal=causal,
                norm=norm,
                n_layers=n_layers,
                # alpha=alpha,
                dropout=dropout,
            ),
            Conv1DBlock(
                model_size,
                2 * model_size,
                kernel_size,
                stride=4,
                # padding=12,
                # use_batch_norm=use_batch_norm,
                causal=causal,
                norm=norm,
                n_layers=n_layers,
                # alpha=alpha,
                dropout=dropout,
            ),
            Conv1DBlock(
                2 * model_size,
                4 * model_size,
                kernel_size,
                stride=4,
                # padding=12,
                # use_batch_norm=use_batch_norm,
                causal=causal,
                norm=norm,
                n_layers=n_layers,
                # alpha=alpha,
                dropout=dropout,
            ),
            Conv1DBlock(
                4 * model_size,
                8 * model_size,
                kernel_size,
                stride=4,
                # padding=12,
                # use_batch_norm=use_batch_norm,
                causal=causal,
                norm=norm,
                n_layers=n_layers,
                # alpha=alpha,
                dropout=dropout,
            ),
            Conv1DBlock(
                8 * model_size,
                16 * model_size,
                kernel_size,
                stride=4,
                # padding=12,
                # use_batch_norm=use_batch_norm,
                causal=causal,
                norm=norm,
                n_layers=n_layers,
                # alpha=alpha,
                dropout=dropout,
            ),
        ]
        self.causal = causal
        # self.fc_d_input = 256 * model_size
        if self.causal:
            self.fc_d_input = 16*model_size
        else:
            self.fc_d_input = self.l_max // 64 * model_size

        # Logic for very long sequences from WaveGAN code
        # if l_max == 32768:
        #     conv_layers.append(
        #         Conv1D(
        #             16 * model_size,
        #             32 * model_size,
        #             kernel_size,
        #             stride=2,
        #             padding=12,
        #             use_batch_norm=use_batch_norm,
        #             alpha=alpha,
        #         )
        #     )
        #     self.fc_d_input = 480 * model_size
        # elif l_max == 65536:
        #     conv_layers.append(
        #         Conv1D(
        #             16 * model_size,
        #             32 * model_size,
        #             kernel_size,
        #             stride=4,
        #             padding=12,
        #             use_batch_norm=use_batch_norm,
        #             alpha=alpha,
        #         )
        #     )
        #     self.fc_d_input = 512 * model_size

        self.conv_layers = nn.ModuleList(conv_layers)

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
        if self.causal:
            x = self.fc1(x.transpose(-1, -2)) # (B, L, output)

            if self.l_output == 0:
                return x[:, -1, :], None
            else:
                return x[:, -self.l_output:, :], None
        else:
            assert self.l_output == 0
            x = x.reshape(-1, self.fc_d_input)
            if self.verbose:
                print(x.shape)
            return self.fc1(x), None


if __name__ == "__main__":
    # from torch.autograd import Variable

    channels = 3
    classes = 10
    for l_max in [1024, 4096, 16000]:

        D = WaveGANDiscriminator(
            d_model=channels,
            d_output=10,
            verbose=True,
            # use_batch_norm=True,
            norm='batch',
            causal=False,
            n_layers=2,
            dropout=0.1,
            l_max=l_max,
        )
        out2 = D(torch.randn(10, l_max, channels))
        print(out2.shape)
        assert out2.shape == (10, classes)
        print("==========================")
