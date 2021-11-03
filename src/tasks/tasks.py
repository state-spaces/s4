from typing import Optional, List, Tuple
import math
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import ListConfig

from src.models.nn.adaptive_softmax import AdaptiveEmbedding, ProjectedAdaptiveLogSoftmax
from src.models.nn.initialization import weights_init_embedding
import src.tasks.metrics as M
import src.models.nn.utils as U
import torchmetrics as tm
from src.utils.config import to_list, instantiate_partial, instantiate


class BaseTask:
    """ Abstract class that takes care of:
    - loss function
    - arbitrary metrics
    - (optional) encoder module that interfaces with dataset (inputs) and model
    - (optional) decoder module that interfaces with dataset (targets) and model
    """
    encoder = None
    decoder = None

    def __init__(self, dataset=None, model=None, loss=None, metrics=None, torchmetrics=None):
        """ This class is allowed to grab attributes directly off a constructed dataset and model object """
        self.dataset = dataset
        self.model = model
        if metrics is None: metrics = []
        self.metric_names = to_list(metrics)

        if torchmetrics is None: torchmetrics = []
        self.torchmetric_names = to_list(torchmetrics)
        self._tracked_torchmetrics = {}

        # Create loss function
        self.loss = instantiate(M.output_metric_fns, loss, partial=True)

    def _init_torchmetrics(self, prefix):
        """
        Instantiate torchmetrics.
        """
        self._tracked_torchmetrics[prefix] = {}
        for name in self.torchmetric_names:
            if name in ['AUROC', 'StatScores', 'Precision', 'Recall', 'F1']:
                self._tracked_torchmetrics[prefix][name] = getattr(tm, name)(average='macro', num_classes=self.dataset.d_output, compute_on_step=False).to('cuda')
            elif '@' in name:
                k = int(name.split('@')[1])
                mname = name.split('@')[0]
                self._tracked_torchmetrics[prefix][name] = getattr(tm, mname)(average='macro', num_classes=self.dataset.d_output, compute_on_step=False, top_k=k).to('cuda')
            else:
                self._tracked_torchmetrics[prefix][name] = getattr(tm, name)(compute_on_step=False).to('cuda')

    def _reset_torchmetrics(self, prefix=None):
        """
        Reset torchmetrics for a prefix
        associated with a particular dataloader (e.g. train, val, test).

        Generally do this at the start of an epoch.
        """
        all_prefixes = [prefix] if prefix is not None else self._tracked_torchmetrics
        for prefix in all_prefixes:
            for name in self.torchmetric_names:
                try:
                    self._tracked_torchmetrics[prefix][name].reset()
                except KeyError:  # metrics don't exist yet
                    pass

    def get_torchmetrics(self, prefix):
        """
        Compute torchmetrics for a prefix associated with
        a particular dataloader (e.g. train, val, test).

        Generally do this at the end of an epoch.
        """
        return {name: self._tracked_torchmetrics[prefix][name].compute() for name in self.torchmetric_names}

    def torchmetrics(self, x, y, prefix):
        """
        Update torchmetrics with new x, y .
        Prefix corresponds to a particular dataloader (e.g. train, val, test).

        Generally call this every batch.
        """
        if prefix not in self._tracked_torchmetrics:
            self._init_torchmetrics(prefix)

        for name in self.torchmetric_names:
            self._tracked_torchmetrics[prefix][name].update(x, y)

    def metrics(self, x, y):
        """
        Metrics are just functions
        output metrics are a function of output and target
        loss metrics are a function of loss (e.g. perplexity)
        """
        output_metrics = {
            name: M.output_metric_fns[name](x, y)
            for name in self.metric_names if name in M.output_metric_fns
        }
        loss_metrics = {
            name: M.loss_metric_fns[name](x, y, self.loss)
            for name in self.metric_names if name in M.loss_metric_fns
        }
        return {**output_metrics, **loss_metrics}


class Scalar(nn.Module):
    def __init__(self, c=1):
        super().__init__()
        self.c = c
    def forward(self, x):
        return x * self.c

class LMTask(BaseTask):
    def __init__(self, tied=False, rescale=True, init=None, **kwargs):
        super().__init__(loss='cross_entropy', **kwargs)
        n_tokens = self.dataset.n_tokens
        d_model = self.model.d_model
        d_output = self.model.d_output

        if rescale:
            scale = U.TupleModule(Scalar)(math.sqrt(d_model))
        else:
            scale = U.Identity()

        embedding = U.Embedding(n_tokens, d_model)
        nn.init.normal_(embedding.weight, mean=0, std=d_model**-.5)
        encoder = nn.Sequential(
            embedding,
            scale,
        )
        self.encoder = encoder
        decoder = U.TupleModule(nn.Linear)(d_output, n_tokens)
        self.decoder = decoder

        if tied:
            assert d_model == d_output
            self.decoder.weight = self.encoder[0].weight

        if init is not None:
            self.encoder.apply(functools.partial(weights_init_embedding, init_cfg=init))

class ForecastingTask(BaseTask):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    
class AdaptiveLMTask(BaseTask):
    def __init__(
        self,
        div_val,
        cutoffs : List[int],
        tie_weights : bool,
        tie_projs : List[bool],
        init_scale=1.0,
        bias_scale=0.0,
        dropemb=0.0,
        dropsoft=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        n_tokens = self.dataset.n_tokens
        d_model = self.model.d_model
        d_output = self.model.d_output

        encoder = U.TupleModule(AdaptiveEmbedding)(
            n_tokens,
            d_model,
            d_model,
            cutoffs=cutoffs,
            div_val=div_val,
            init_scale=init_scale,
            dropout=dropemb,
        )

        if tie_weights:
            assert d_model == d_output
            emb_layers = [i.weight for i in encoder.emb_layers]
        else:
            emb_layers = None

        # Construct decoder/loss
        emb_projs = encoder.emb_projs
        loss = ProjectedAdaptiveLogSoftmax(
            n_tokens, d_output, d_output,
            cutoffs, div_val=div_val,
            tie_projs=tie_projs,
            out_projs=emb_projs,
            out_layers_weights=emb_layers,
            bias_scale=bias_scale,
            dropout=dropsoft,
        )

        self.encoder = U.TupleSequential(encoder, self.encoder)
        self.loss = loss


registry = {
    'base': BaseTask,
    'lm': LMTask,
    'adaptivelm': AdaptiveLMTask,
}
