import math
import torch
import torch.nn.functional as F
from src.tasks.mixture import mixture_loss, mixture_loss_kd
from sklearn.metrics import f1_score, roc_auc_score
from functools import partial

def binary_cross_entropy(logits, y):
    # BCE loss requires squeezing last dimension of logits so it has the same shape as y
    # requires y to be float, since it's overloaded to represent a probability
    return F.binary_cross_entropy_with_logits(logits.squeeze(-1), y.float())


def binary_accuracy(logits, y):
    return torch.eq(logits.squeeze(-1) >= 0, y).float().mean()


def cross_entropy(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    return F.cross_entropy(logits, y)


def accuracy(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    if y.numel() > logits.shape[0]:
        # Mixup leads to this case: use argmax class
        y = y.argmax(dim=-1)
    y = y.view(-1)
    return torch.eq(torch.argmax(logits, dim=-1), y).float().mean()


def accuracy_at_k(logits, y, k=1):
    logits = logits.view(-1, logits.shape[-1])
    if y.numel() > logits.shape[0]:
        # Mixup leads to this case: use argmax class
        y = y.argmax(dim=-1)
    y = y.view(-1)
    return torch.topk(logits, k, dim=-1)[1].eq(y.unsqueeze(-1)).any(dim=-1).float().mean()

def f1_binary(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    y_hat = torch.argmax(logits, dim=-1)
    return f1_score(y.cpu().numpy(), y_hat.cpu().numpy(), average="binary")


def f1_macro(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    y_hat = torch.argmax(logits, dim=-1)
    return f1_score(y.cpu().numpy(), y_hat.cpu().numpy(), average="macro")


def f1_micro(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    y_hat = torch.argmax(logits, dim=-1)
    return f1_score(y.cpu().numpy(), y_hat.cpu().numpy(), average="micro")


def roc_auc_macro(logits, y):
    logits = logits.view(
        -1, logits.shape[-1]
    ).detach()  # KS: had to add detach to eval while training
    y = y.view(-1)
    return roc_auc_score(
        y.cpu().numpy(), F.softmax(logits, dim=-1).cpu().numpy()[:, 1], average="macro"
    )


def roc_auc_micro(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    return roc_auc_score(
        y.cpu().numpy(), F.softmax(logits, dim=-1).cpu().numpy()[:, 1], average="micro"
    )


def mse(outs, y, len_batch=None):
    # assert outs.shape[:-1] == y.shape and outs.shape[-1] == 1
    # outs = outs.squeeze(-1)
    if len(y.shape) < len(outs.shape):
        assert outs.shape[-1] == 1
        outs = outs.squeeze(-1)
    if len_batch is None:
        return F.mse_loss(outs, y)
    else:
        # Computes the loss of the first `lens` items in the batches
        mask = torch.zeros_like(outs, dtype=torch.bool)
        for i, l in enumerate(len_batch):
            mask[i, :l, :] = 1
        outs_masked = torch.masked_select(outs, mask)
        y_masked = torch.masked_select(y, mask)
        return F.mse_loss(outs_masked, y_masked)


def mae(outs, y, len_batch=None):
    # assert outs.shape[:-1] == y.shape and outs.shape[-1] == 1
    # outs = outs.squeeze(-1)
    if len(y.shape) < len(outs.shape):
        assert outs.shape[-1] == 1
        outs = outs.squeeze(-1)
    if len_batch is None:
        return F.l1_loss(outs, y)
    else:
        # Computes the loss of the first `lens` items in the batches
        mask = torch.zeros_like(outs, dtype=torch.bool)
        for i, l in enumerate(len_batch):
            mask[i, :l, :] = 1
        outs_masked = torch.masked_select(outs, mask)
        y_masked = torch.masked_select(y, mask)
        return F.l1_loss(outs_masked, y_masked)


# Metrics that can depend on the loss
def loss(x, y, loss_fn):
    """ This metric may be useful because the training loss may add extra regularization (e.g. weight decay implemented as L2 penalty), while adding this as a metric skips the additional losses """
    return loss_fn(x, y)


def rmse(x, y, loss_fn):
    return loss_fn(x, y) ** 0.5  # NOTE this isn't exactly correct


def bpb(x, y, loss_fn):
    """ bits per byte (image density estimation, speech generation, char LM) """
    return loss_fn(x, y) / math.log(2)


def ppl(x, y, loss_fn):
    return torch.exp(loss_fn(x, y))


# should have a better way to do this
output_metric_fns = {
    "binary_cross_entropy": binary_cross_entropy,
    "cross_entropy": cross_entropy,
    "binary_accuracy": binary_accuracy,
    "accuracy": accuracy,
    'accuracy@3': partial(accuracy_at_k, k=3),
    'accuracy@5': partial(accuracy_at_k, k=5),
    'accuracy@10': partial(accuracy_at_k, k=10),
    "eval_loss": loss,
    "mixture": mixture_loss,
    "mixture_kd": mixture_loss_kd,
    "mse": mse,
    "mae": mae,
    "f1_binary": f1_binary,
    "f1_macro": f1_macro,
    "f1_micro": f1_micro,
    "roc_auc_macro": roc_auc_macro,
    "roc_auc_micro": roc_auc_micro,
}
loss_metric_fns = {
    "loss": loss,
    "bpb": bpb,
    "ppl": ppl,
}
metric_fns = {**output_metric_fns, **loss_metric_fns}  # TODO py3.9
