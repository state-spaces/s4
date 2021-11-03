""" Implementations of several types of Discrete Sin/Cosine Transforms with various reductions to FFT. """

if __name__ == '__main__':
    import sys
    import pathlib
    p = pathlib.Path().absolute()
    print("Adding path: ", p)
    sys.path.append(str(p))

import torch
import torch.nn as nn
import numpy as np
import scipy.fft
from einops import rearrange, repeat

class DCT(nn.Module):
    """ Reductions adapted from https://dsp.stackexchange.com/questions/2807/fast-cosine-transform-via-fft """

    def __init__(self, N, norm='backward'):
        super().__init__()

        self.N = N

        # Materialize DCT matrix
        P = scipy.fft.dct(np.eye(N), norm=norm, type=2).T
        P = torch.tensor(P, dtype=torch.float)
        self.register_buffer('P', P)

        # TODO take care of normalization
        Q = np.exp(-1j * np.pi / (2 * self.N) * np.arange(self.N)) 
        Q = torch.tensor(Q, dtype=torch.cfloat)
        self.register_buffer('Q', Q) # half shift

    def forward(self, x, mode=2):
        if mode == 0:
            return self.forward_dense(x)
        elif mode == 1:
            return self.forward_n(x)
        elif mode == 2:
            return self.forward_2n(x)
        elif mode == 4:
            return self.forward_4n(x)

    def forward_dense(self, x):
        """ Baseline DCT type II - matmul by DCT matrix """
        y = (self.P.to(x) @ x.unsqueeze(-1)).squeeze(-1)
        return y

    def forward_4n(self, x):
        """ DCT type II - reduction to FFT size 4N """
        assert self.N == x.shape[-1]
        x = torch.cat([x, x.flip(-1)], dim=-1)
        z = torch.zeros_like(x)
        x = torch.stack([z, x], dim=-1)
        x = x.view(x.shape[:-2] + (-1,))
        y = torch.fft.fft(x)
        y = y[..., :self.N]
        if torch.is_complex(x):
            return y
        else:
            return torch.real(y)

    def forward_2n(self, x):
        """ DCT type II - reduction to FFT size 2N mirrored

        The reduction from the DSP forum is not quite correct in the complex input case.
        halfshift(FFT[a, b, c, d, d, c, b, a]) -> [A, B, C, D, 0, -D, -C, -B]
        In the case of real input, the intermediate step after FFT has form [A, B, C, D, 0, D*, C*, B*]
        """
        assert self.N == x.shape[-1]
        x = torch.cat([x, x.flip(-1)], dim=-1)
        y = torch.fft.fft(x)[..., :self.N]
        y = y * self.Q
        if torch.is_complex(x):
            return y
        else:
            return torch.real(y)

    def forward_n(self, x):
        """ DCT type II - reduction to size N """
        assert self.N == x.shape[-1]
        x = torch.cat([x[..., 0::2], x[..., 1::2].flip(-1)], dim=-1)
        y = torch.fft.fft(x)
        y = y * 2 * self.Q
        if torch.is_complex(x):
            y = torch.cat([y[..., :1], (y[..., 1:] + 1j * y[..., 1:].flip(-1)) / 2], dim=-1) # TODO in-place sum
        else:
            y = torch.real(y)
        return y

class IDCT(nn.Module):
    def __init__(self, N, norm='backward'):
        super().__init__()

        self.N = N

        # Materialize DCT matrix
        P = np.linalg.inv(scipy.fft.dct(np.eye(N), norm=norm, type=2).T)
        P = torch.tensor(P, dtype=torch.float)
        self.register_buffer('P', P)

        # TODO take care of normalization
        Q = np.exp(-1j * np.pi / (2 * self.N) * np.arange(2*self.N))
        Q = torch.tensor(Q, dtype=torch.cfloat)
        self.register_buffer('Q', Q) # half shift

    def forward(self, x, mode=2):
        if mode == 0:
            return self.forward_dense(x)
        elif mode == 1:
            return self.forward_n(x)
        elif mode == 2:
            return self.forward_2n(x)
        elif mode == 4:
            return self.forward_4n(x)

    def forward_dense(self, x):
        """ Baseline DCT type II - matmul by DCT matrix """
        y = (self.P.to(x) @ x.unsqueeze(-1)).squeeze(-1)
        return y

    def forward_4n(self, x):
        """ DCT type II - reduction to FFT size 4N """
        assert self.N == x.shape[-1]
        z = x.new_zeros(x.shape[:-1] + (1,))
        x = torch.cat([x, z, -x.flip(-1), -x[..., 1:], z, x[..., 1:].flip(-1)], dim=-1)
        y = torch.fft.ifft(x)
        y = y[..., 1:2*self.N:2]
        if torch.is_complex(x):
            return y
        else:
            return torch.real(y)

    def forward_2n(self, x):
        """ DCT type II - reduction to FFT size 2N mirrored """
        assert self.N == x.shape[-1]
        z = x.new_zeros(x.shape[:-1] + (1,))
        x = torch.cat([x, z, -x[..., 1:].flip(-1)], dim=-1)
        x = x / self.Q
        y = torch.fft.ifft(x)[..., :self.N]
        if torch.is_complex(x):
            return y
        else:
            return torch.real(y)

    def forward_n(self, x):
        """ DCT type II - reduction to size N """
        assert self.N == x.shape[-1]
        raise NotImplementedError # Straightforward by inverting operations of DCT-II reduction

def test_dct_ii():
    N = 8
    dct = DCT(N)

    baseline = dct.forward_dense
    methods = [dct.forward_4n, dct.forward_2n, dct.forward_n]

    # Real case
    print("DCT-II Real input")
    x = torch.randn(1, N)
    y = baseline(x)
    print(y)
    for fn in methods:
        y_ = fn(x)
        print("err", torch.norm(y-y_))

    # Complex case
    print("DCT-II Complex input")
    x = torch.randn(N) + 1j * torch.randn(N)
    y = baseline(x)
    print(y)
    for fn in methods:
        y_ = fn(x)
        print("err", torch.norm(y-y_))

def test_dct_iii():
    N = 8
    dct = IDCT(N)

    baseline = dct.forward_dense
    methods = [dct.forward_4n, dct.forward_2n]

    # Real case
    print("DCT-III Real input")
    x = torch.randn(1, N)
    y = baseline(x)
    print(y)
    for fn in methods:
        y_ = fn(x)
        print("err", torch.norm(y-y_))

    # Complex case
    print("DCT-III Complex input")
    # x = torch.randn(N) + 1j * torch.randn(N)
    x = 1j * torch.ones(N)
    y = baseline(x)
    print(y)
    for fn in methods:
        y_ = fn(x)
        print("err", torch.norm(y-y_))

def benchmark_dct():
    T = 100
    B = 4 * 512
    N = 512
    dct2 = DCT(N)
    dct3 = IDCT(N)
    x = torch.randn(B, N) + 1j * torch.randn(B, N)
    dct2 = dct2.to(device)
    dct3 = dct3.to(device)
    x = x.to(device)

    methods = [
        ("DCT-II dense", dct2.forward_dense),
        ("DCT-II 4N", dct2.forward_4n),
        ("DCT-II 2N", dct2.forward_2n),
        ("DCT-II N", dct2.forward_n),

        ("DCT-II dense", dct3.forward_dense),
        ("DCT-II 4N", dct3.forward_4n),
        ("DCT-II 2N", dct3.forward_2n),
    ]
    for name, fn in methods:
        utils.benchmark_forward(T, fn, x, desc=name)

"""
Benchmarking results:
Complex (Dense / 4N / 2N / N)

CPU (2 threads):
    B, N = 1, 64    :  30 /  50 / 36 /  86  us
    B, N = 1, 256   :  84 / 105 / 99 / 137 us
    B, N = 1, 1024  : 311 / 129 / 64 / 106 us

T4:
    B, N = 1, 256     :   230 /  203 /  241 /  5090 us
    B, N = 1*256, 256 :   163 /  150 /  241 /  7210 us
    B, N = 4*256, 256 :   285 /  526 /  319 /  5000 us
    B, N = 1, 512     :    55 /   87 /   60 /  4630 us
    B, N = 1*512, 512 :   424 /  730 /  140 /  5060 us
    B, N = 4*512, 512 :  1220 / 2020 /  743 /  7240 us
    B, N = 1, 1024    :   211 /  105 /   55 /  1900 us
    B, N = 4096, 1024 : 10410 / 8690 / 2870 / 10540 us

    DCT-III
    B, N = 4*512, 512 :  1.49 /  5.01 / 4.17 ms
    B, N = 4096, 1024 : 12.02 / 11.32 / 8.18 ms
"""

if __name__ == '__main__':
    from benchmark import utils

    device = 'cuda' # 'cpu'
    device = torch.device(device)

    test_dct_ii()
    test_dct_iii()
    # benchmark_dct()
