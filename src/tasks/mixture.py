import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
from einops import rearrange, repeat

# Implementation from https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py
def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    # TF ordering
    # axis = len(x.size()) - 1
    # m, _ = torch.max(x, dim=axis, keepdim=True)
    # return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))
    m, _ = torch.max(x, dim=-1, keepdim=True)
    x = x-m
    return x - torch.logsumexp(x, dim=-1, keepdim=True)

def discretized_mix_logistic_loss_3d(x, l):
    """
    log-likelihood for mixture of discretized logistics, specially for the 3 channel case
    assumes the data has been rescaled to [-1,1] interval
    """
    # Pytorch ordering
    # x = x.permute(0, 2, 3, 1)
    # l = l.permute(0, 2, 3, 1)
    xs = x.shape # [int(y) for y in x.size()]
    ls = l.shape # [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10)
    logit_probs = l[..., :nr_mix]
    l = l[..., nr_mix:].contiguous().view(xs + (3*nr_mix,)) # 3 for mean, scale, coef
    means = l[..., :, :nr_mix]
    # log_scales = torch.max(l[..., :, nr_mix:2 * nr_mix], -7.)
    log_scales = torch.clamp(l[..., :, nr_mix:2*nr_mix], min=-7.)

    coeffs = torch.tanh(l[..., :, 2*nr_mix:3*nr_mix])
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    # x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix]).cuda(), requires_grad=False)
    x = x.unsqueeze(-1)
    m2 = (means[..., 1, :] + coeffs[..., 0, :] * x[..., 0, :]).view(xs[:-1] + (1, nr_mix))

    m3 = (means[..., 2, :] + coeffs[..., 1, :] * x[..., 0, :] +
            coeffs[..., 2, :] * x[..., 1, :]).view(xs[:-1] + (1, nr_mix))

    means = torch.cat((means[..., 0, :].unsqueeze(-2), m2, m3), dim=-2)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal
    # case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation
    # based on the assumption that the log-density is constant in the bin of
    # the observed sub-pixel value

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out  = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
    inner_cond       = (x > 0.999).float()
    inner_out        = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond             = (x < -0.999).float()
    log_probs        = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs        = torch.sum(log_probs, dim=-2) + log_prob_from_logits(logit_probs)

    # return -torch.sum(log_sum_exp(log_probs))
    return -torch.mean(torch.logsumexp(log_probs, dim=-1)) / 3.0

def discretized_mix_logistic_loss_1d(x, l):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Pytorch ordering
    # x = x.permute(0, 2, 3, 1)
    # l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = ls[-1] // 3 # k
    logit_probs = l[..., :nr_mix] # (b, h, w, k)
    l = l[..., nr_mix:].contiguous().view(xs + [nr_mix * 2]) # 2 for mean, scale
    means = l[..., :, :nr_mix] # (b, h, w, 1, k)
    log_scales = torch.clamp(l[..., :, nr_mix:2 * nr_mix], min=-7.) # (b, h, w, 1, k)
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous() # (b, h, w, 1)
    # x = x.unsqueeze(-1) + nn.Variable(torch.zeros(xs + [nr_mix]).cuda(), requires_grad=False)
    x = x.unsqueeze(-1)

    # means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means # (b, h, w, 1, k)
    inv_stdv = torch.exp(-log_scales) # (b, h, w, 1, k)
    plus_in = inv_stdv * (centered_x + 1. / 255.) # (b, h, w, 1, k)
    cdf_plus = torch.sigmoid(plus_in) # (b, h, w, 1, k)
    min_in = inv_stdv * (centered_x - 1. / 255.) # (b, h, w, 1, k)
    cdf_min = torch.sigmoid(min_in) # (b, h, w, 1, k)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out  = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
    inner_cond       = (x > 0.999).float() # (b, h, w, 1, 1)
    inner_out        = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond             = (x < -0.999).float() # (b, h, w, 1, 1)
    log_probs        = cond * log_cdf_plus + (1. - cond) * inner_out
    # log_probs        = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)
    assert log_probs.size(-2) == 1
    log_probs        = torch.squeeze(log_probs, dim=-2) + log_prob_from_logits(logit_probs) # (b, h, w, k)

    # return -torch.sum(log_sum_exp(log_probs))
    return -torch.mean(torch.logsumexp(log_probs, dim=-1))


""" My implementations of the mixture loss functions. Simpler and faster. """

def mixture_loss(outs, y, c=256, cdf_fn=torch.sigmoid, reduce='mean', scale=2.0):
    """
    outs: (..., 3*k)
    y: (...) int between 0 to c-1 (inclusive)
    c: number of classes

    scale: hyperparameter that increases the size of the buckets, i.e. increases entropy of distribution (less confident)
    """
    assert outs.shape[-1] % 3 == 0
    k = outs.shape[-1] // 3

    # Transform targets
    y = y.unsqueeze(-1) # (..., 1)
    # y_normalized = (2*y - (c-1)) / (c-1)
    # y_normalized = (y - (c-1)/2) / ((c-1)/2)
    # Buckets are slightly offset from normal implementation, to match Mixture() class below
    y_normalized = (y - (c-1)/2) / ((c-2)/2)

    # bin_max = y_normalized + 1./(c-1) # (..., 1)
    # bin_min = y_normalized - 1./(c-1) # (..., 1)
    bin_max = y_normalized + 1./(c-2) # (..., 1)
    bin_min = y_normalized - 1./(c-2) # (..., 1)

    bin_max = bin_max * scale
    bin_min = bin_min * scale

    # Unpack outputs
    mixture_logits = outs[..., :k] # (..., k)
    means = outs[..., k:2*k] # (..., k)
    scales = outs[..., 2*k:3*k].clamp(min=-7.) # (..., k)

    # Transform bins by mean and scale

    # equivalent to dividing by exp(scales) or negating scales; marginally easier for me to reason about multiply
    bin_min = (bin_min - means) * torch.exp(scales) # (..., k)
    bin_max = (bin_max - means) * torch.exp(scales) # (..., k)

    # Calculate probabilities
    cdf_max = cdf_fn(bin_max)
    cdf_min = cdf_fn(bin_min)

    # Edge cases for endpoints
    z = torch.zeros_like(y, dtype=torch.float) # torch.where doesn't support float32 scalar...
    tail_min = torch.where(y == 0, cdf_min, z)
    tail_max = torch.where(y == c-1, 1.-cdf_max, z)

    probs = cdf_max - cdf_min + tail_min + tail_max + 1e-8 # pad for numerical stability

    # Finish calculation in logit space; I doubt its more stable but previous implementations do this
    # Equivalent to working in probability space:
    #   probs = torch.sum(torch.softmax(mixture_logits, dim=-1) * probs, dim=-1)
    #   log_probs = torch.log(probs)
    log_probs = torch.log(probs)
    log_probs = torch.logsumexp(log_probs + log_prob_from_logits(mixture_logits), dim=-1)


    if reduce == 'mean':
        return -log_probs.mean()
    elif reduce == 'none':
        return -log_probs
    else: raise NotImplementedError

def mixture_loss_kd(outs, y, c=256, cdf_fn=torch.sigmoid, reduce='mean'):
    """ Mixture loss for outputting multiple distributions at once, where later predictions can depend linearly on previous ones.

    outs: (..., 3*k)
    y: (..., d) int between 0 to c-1 (inclusive)
    c: number of classes
    """
    d = y.shape[-1]
    factor = 1 + 2*d + d*(d-1)//2
    assert outs.shape[-1] % factor == 0
    k = outs.shape[-1] // factor

    # Transform targets
    y = y.unsqueeze(-1)
    y_normalized = (y - (c-1)/2) / ((c-1)/2)
    bin_max = y_normalized + 1./(c-1) # (..., d)
    bin_min = y_normalized - 1./(c-1) # (..., d)
    # y_normalized = (y - (c-1)/2) / ((c-2)/2)
    # bin_max = y_normalized + 1./(c-2) # (..., d)
    # bin_min = y_normalized - 1./(c-2) # (..., d)

    bin_max = bin_max * 1.
    bin_min = bin_min * 1.

    # Unpack outputs
    outs = rearrange(outs, '... (d k) -> ... d k', k=k)
    mixture_logits = outs[..., 0, :] # (..., k)
    means = outs[..., 1:1+d, :] # (..., d*k)
    scales = outs[..., 1+d:1+2*d, :] # (..., d*k)
    coeffs = torch.tanh(outs[..., 1+2*d:, :])

    # Transform means with linear combinations
    # means = rearrange(means, '... (d k) -> ... d k', k=k)
    # scales = rearrange(scales, '... (d k) -> ... d k', k=k)
    idx = 0
    for i in range(1,d):
        means[..., i, :] += torch.sum(coeffs[..., idx:idx+i, :] * y_normalized[..., :i, :], dim=-2) # (..., k)
        idx += i

    # Transform bins by mean and scale

    # equivalent to dividing by exp(scales) or negating scales; marginally easier for me to reason about multiply
    bin_min = (bin_min - means) * torch.exp(scales) # (..., d, k)
    bin_max = (bin_max - means) * torch.exp(scales) # (..., d, k)

    # Calculate probabilities
    cdf_max = cdf_fn(bin_max)
    cdf_min = cdf_fn(bin_min)

    # Edge cases for endpoints
    z = torch.zeros_like(y, dtype=torch.float) # torch.where doesn't support float32 scalar...
    tail_min = torch.where(y == 0, cdf_min, z)
    tail_max = torch.where(y == c-1, 1.-cdf_max, z)

    probs = cdf_max - cdf_min + tail_min + tail_max + 1e-8 # pad for numerical stability

    # Finish calculation in logit space; I doubt its more stable but previous implementations do this
    # Equivalent to working in probability space:
    #   probs = torch.sum(torch.softmax(mixture_logits, dim=-1) * probs, dim=-1)
    #   log_probs = torch.log(probs)
    log_probs = torch.log(probs) # (..., d, k)
    log_probs = torch.sum(log_probs, dim=-2) # (..., k)
    log_probs = torch.logsumexp(log_probs + log_prob_from_logits(mixture_logits), dim=-1) # (...)


    if reduce == 'mean':
        return -log_probs.mean() / 3.0
    elif reduce == 'none':
        return -log_probs
    else: raise NotImplementedError

def mixture_sample(x):
    """ x: (..., 3*k) mixture params """
    # Pytorch ordering
    assert x.shape[-1] % 3 == 0
    k = x.shape[-1] // 3

    # Unpack outputs
    mixture_logits = x[..., :k] # (..., k)
    means = x[..., k:2*k] # (..., k)
    scales = x[..., 2*k:3*k].clamp(min=-7.) # (..., k)

    # sample mixture indicator from softmax
    eps = 1e-8
    temp = torch.rand_like(means) * (1-2*eps) + eps
    temp = mixture_logits - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=-1, keepdim=True) # (..., 1)

    means = torch.gather(means, -1, argmax).squeeze(-1)
    scales = torch.gather(scales, -1, argmax).squeeze(-1)

    u = torch.rand_like(means) * (1-2*eps) + eps
    x = means + (torch.log(u) - torch.log(1. - u)) / torch.exp(scales)  # (...)
    return x



def piecewise_cdf(x):
    """ Piecewise linear function with nodes at (-1, 0) and (1, 1) """
    x = F.relu(1+x)-1
    x = 1-F.relu(1-x)
    x = (x+1)/2
    return x

def pdf(m, s, buckets, cdf_fn):
    """
    m: (...) mean
    s: (...) scale
    buckets: (..., n-1)

    returns: (..., n)
    """
    samples = s.unsqueeze(-1) * (buckets - m.unsqueeze(-1))
    # samples = (buckets - m.unsqueeze(-1)) / s.unsqueeze(-1)
    # samples = s.unsqueeze(-1) * buckets + m.unsqueeze(-1)
    c = cdf_fn(samples) # (..., b) between 0, 1
    p0 = c[..., :1] # (..., 1)
    pn = 1. - c[..., -1:] # (..., 1)
    p = c[..., 1:] - c[..., :-1] # (..., b-2)
    probs = torch.cat([p0, p, pn], dim=-1) # (..., b)
    return probs

class Mixture(nn.Module):
    def __init__(self, b, a, cdf='piecewise'):
        super().__init__()
        self.b = b
        self.a = a
        self.cdf_fn = {
            'piecewise': piecewise_cdf,
            'sigmoid': F.sigmoid,
        }[cdf]

        assert b % 2 == 0
        buckets = torch.linspace(-1.0, 1.0, b-1) * a
        # buckets = torch.linspace(-1.0+1/(b-1), 1.0-1/(b-1), b-1) * a
        self.register_buffer('buckets', buckets)

    def forward(self, x):
        """
        x: (..., 3*k)
        """
        l, m, s = torch.unbind(rearrange(x, '... (z a) -> ... z a', z=3), dim=-2)
        p = pdf(m, torch.exp(s), self.buckets, self.cdf_fn) # (..., k, b)
        weights = F.softmax(l, dim=-1) # (..., k)
        probs = torch.sum(weights.unsqueeze(-1) * p, dim=-2) # (..., b)
        logits = torch.log(probs + 1e-8)
        return logits

def test_mixture_loss():
    logits = torch.FloatTensor(5, 1024, 30).normal_()
    y = torch.randint(0, 256, (5, 1024, 1))

    ans = []
    for target in range(256):
        y = torch.ones(5, 1024, dtype=torch.long) * target
        loss = mixture_loss(logits, y, reduce='none')
        ans.append(torch.exp(-loss))
    total_prob = sum(ans)
    print(torch.max(total_prob))
    print(torch.min(total_prob))

def test_mixture_function():
    m = torch.tensor([0.0])
    s = torch.tensor([1.0])
    buckets = torch.tensor([-1.0, 0.0, 1.0])

    p = pdf(m, s, buckets, piecewise_cdf)
    print(p)

    mixture = Mixture(4, 1.0, 'piecewise')

    s = torch.tensor([0.0])
    l = torch.tensor([0.0])
    p = mixture(torch.cat([m, s, l], dim=-1))
    print(p)

def test_pixelcnn_mixture():
    # x = torch.FloatTensor(5, 1024, 1).uniform_(-1., 1.)
    y = torch.randint(0, 256, (5, 1024, 1))
    x = (y - 255/2) / (255/2)
    logits = torch.FloatTensor(5, 1024, 30).normal_()
    loss = discretized_mix_logistic_loss_1d(x, logits)
    print(loss)
    loss = mixture_loss(logits, y.squeeze(-1))
    print(loss)

    mixture = Mixture(256, 2.0, 'sigmoid')
    loss = F.cross_entropy(mixture(logits).reshape(-1,256), y.view(-1))
    print(loss)

    y = torch.randint(0, 256, (5, 32, 32, 3))
    x = (y - 255/2) / (255/2)
    # x = torch.FloatTensor(5, 32, 32, 3).uniform_(-1., 1.)
    logits = torch.FloatTensor(5, 32, 32, 30).normal_()
    loss = discretized_mix_logistic_loss_3d(x, logits)
    print(loss)
    loss = mixture_loss_kd(logits, y)
    print(loss)

def test_mixture_sample():
    B = 8
    k = 5
    # x = torch.rand(B, 3*k)
    means = torch.linspace(-1.0, 1.0, k)
    scales = torch.full((B, k), 5.0) # Higher scale means more confident
    logits = torch.zeros(B, k)
    x = torch.cat([logits, means.repeat(B, 1), scales], dim=-1)
    samples = mixture_sample(x)
    print(samples.shape, samples) # Should see values close to -1, -.5, 0, .5, 1

if __name__ == '__main__':
    # test_mixture_function()
    # test_mixture_loss()
    # test_pixelcnn_mixture()
    test_mixture_sample()
