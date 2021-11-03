from typing import List, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
import hydra
from omegaconf import OmegaConf, DictConfig

import src.utils as utils
import src.utils.train
from src.utils.optim.ema import build_ema_optimizer
from src.utils import registry
from src.tasks import encoders, decoders, tasks
import src.models.nn.utils as U
from src.dataloaders import SequenceDataset  # TODO make registry
from tqdm.auto import tqdm

log = src.utils.train.get_logger(__name__)


class SequenceLightningModule(pl.LightningModule):
    def __init__(self, config):
        # Disable profiling executor. This reduces memory and increases speed.
        try:
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_set_profiling_mode(False)
        except AttributeError:
            pass

        super().__init__()
        # Passing in config expands it one level, so can access by self.hparams.train instead of self.hparams.config.train
        self.save_hyperparameters(config, logger=False)

        self.dataset = SequenceDataset.registry[self.hparams.dataset._name_](
            **self.hparams.dataset
        )

        # Check hparams
        self._check_config()

        # PL has some bugs, so add hooks and make sure they're only called once
        self._has_setup = False
        self._has_on_post_move_to_device = False

    def setup(self, stage=None):
        self.dataset.setup()

        # We need to set up the model in setup() because for some reason when training with DDP, one GPU uses much more memory than the others
        # In order to not overwrite the model multiple times during different stages, we need this hack
        # TODO PL 1.5 seems to have an option to skip hooks to avoid this
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/5410#issuecomment-762257024
        if self._has_setup:
            return
        else:
            self._has_setup = True

        # Convenience feature: if model specifies encoder, combine it with main encoder
        encoder_cfg = utils.to_list(self.hparams.encoder) + utils.to_list(
            self.hparams.model.pop("encoder", None)
        )
        decoder_cfg = utils.to_list(self.hparams.model.pop("decoder", None)) + utils.to_list(self.hparams.decoder)

        # Instantiate model
        self.model = utils.instantiate(registry.model, self.hparams.model)

        # Instantiate the task
        if "task" not in self.hparams:  # TODO maybe don't need this?
            self.hparams.task = self.dataset.default_task
        self.task = task = utils.instantiate(
            tasks.registry, self.hparams.task, dataset=self.dataset, model=self.model
        )

        # Create encoders and decoders
        encoder = encoders.instantiate(
            encoder_cfg, dataset=self.dataset, model=self.model
        )
        decoder = decoders.instantiate(
            self.hparams.decoder, model=self.model, dataset=self.dataset
        )

        # Extract the modules so they show up in the top level parameter count
        self.encoder = U.TupleSequential(task.encoder, encoder)
        self.decoder = U.TupleSequential(decoder, task.decoder)
        self.loss = task.loss
        self.metrics = task.metrics

        # Handle state logic
        self._initialize_state()

    def _check_config(self):
        assert self.hparams.train.state.mode in [None, "none", "null", "reset", "bptt"]
        assert (
            (n := self.hparams.train.state.n_context) is None
            or isinstance(n, int)
            and n >= 0
        )
        assert (
            (n := self.hparams.train.state.n_context_eval) is None
            or isinstance(n, int)
            and n >= 0
        )

    def _initialize_state(self):
        self._state = None
        self._memory_chunks = []

    def _reset_state(self, batch, device=None):
        device = device or batch[0].device
        self._state = self.model.default_state(*batch[0].shape[:1], device=device)

    def _process_state(self, batch, batch_idx, train=True):
        """Handle logic for state context. This is unused for all current S3 experiments"""

        # Number of context steps
        key = "n_context" if train else "n_context_eval"
        n_context = self.hparams.train.state.get(key)

        # Don't need to do anything if 0 context steps
        if n_context == 0:
            return

        # Reset state if needed
        if self.hparams.train.state.mode == "reset":
            if batch_idx % (n_context + 1) == 0:
                self._reset_state(batch)

        # Pass through memory chunks
        elif self.hparams.train.state.mode == "bptt":
            self._reset_state(batch)
            with torch.no_grad():  # should be unnecessary because individual modules should handle this
                for _batch in self._memory_chunks:
                    self.forward(_batch)
            # Prepare for next step
            self._memory_chunks.append(batch)
            self._memory_chunks = self._memory_chunks[-n_context:]

    def on_epoch_start(self):
        self._initialize_state()

    def forward(self, batch):
        """Passes a batch through the encoder, backbone, and decoder"""
        # z holds arguments such as sequence length
        x, y, *z = batch
        # w can model-specific constructions such as key_padding_mask for transformers or state for RNNs
        x, *w = self.encoder(x, *z)
        x, state = self.model(x, *w, state=self._state)
        self._state = state
        x, *w = self.decoder(x, state, *z)
        return x, y, *w

    @torch.inference_mode()
    def forward_recurrence(self, batch, k=1):
        """This is a bit hacky; not part of the main train loop, only used to benchmark speed of recurrent view"""
        x, y, *z = batch
        T = x.shape[1]

        if k > 1:
            x = torch.cat([x] * k, dim=0)

        self._state = self.model.default_state(*x.shape[:1], device="cuda")

        x_all = []
        w_all = []
        for t in tqdm(range(T)):

            x_t = x[:, t]
            x_t = x_t.to("cuda")

            x_t, *w_t = self.encoder(x_t)
            x_t, state = self.model.step(x_t, state=self._state)
            self._state = state
            x_t, *w_t = self.decoder(x_t, state)

            x_all.append(x_t)
            w_all.append(w_t)
        return torch.stack(x_all), y, *[torch.stack(w_) for w_ in zip(*w_all)]

    def _shared_step(self, batch, batch_idx, prefix="train"):

        self._process_state(batch, batch_idx, train=(prefix == "train"))

        x, y, *w = self.forward(batch)

        # Loss
        loss = self.loss(x, y, *w)

        # Metrics
        metrics = self.metrics(x, y)
        metrics["loss"] = loss
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        # Calculate torchmetrics: these are accumulated and logged at the end of epochs
        self.task.torchmetrics(x, y, prefix)

        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        return loss

    def on_train_epoch_start(self):
        # Reset training torchmetrics
        self.task._reset_torchmetrics("train")

    def training_epoch_end(self, outputs):
        # Log training torchmetrics
        super().training_epoch_end(outputs)
        self.log_dict(
            {f"train/{k}": v for k, v in self.task.get_torchmetrics("train").items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )

    def on_validation_epoch_start(self):
        # Reset all validation torchmetrics
        for name in self.val_loader_names:
            self.task._reset_torchmetrics(name)

    def validation_epoch_end(self, outputs):
        # Log all validation torchmetrics
        super().validation_epoch_end(outputs)
        for name in self.val_loader_names:
            self.log_dict(
                {f"{name}/{k}": v for k, v in self.task.get_torchmetrics(name).items()},
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
                sync_dist=True,
            )

    def on_test_epoch_start(self):
        # Reset all test torchmetrics
        for name in self.test_loader_names:
            self.task._reset_torchmetrics(name)

    def test_epoch_end(self, outputs):
        # Log all test torchmetrics
        super().test_epoch_end(outputs)
        for name in self.test_loader_names:
            self.log_dict(
                {f"{name}/{k}": v for k, v in self.task.get_torchmetrics(name).items()},
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
                sync_dist=True,
            )

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, prefix="train")

        # Log the loss explicitly so it shows up in WandB
        # Note that this currently runs into a bug in the progress bar with ddp (as of 1.4.6)
        # https://github.com/PyTorchLightning/pytorch-lightning/pull/9142
        # We additionally log the epochs under 'trainer' to get a consistent prefix with 'global_step'
        loss_epoch = {"trainer/loss": loss, "trainer/epoch": self.current_epoch}
        self.log_dict(
            loss_epoch,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        # Log any extra info that the models want to expose (e.g. output norms)
        metrics = {}
        for module in list(self.modules())[1:]:
            if hasattr(module, "metrics"):
                metrics.update(module.metrics)

        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        ema = (
            self.val_loader_names[dataloader_idx].endswith("/ema")
            and self.optimizers().optimizer.stepped
        )  # There's a bit of an annoying edge case with the first (0-th) epoch; it has to be excluded due to the initial sanity check
        if ema:
            self.optimizers().swap_ema()
        loss = self._shared_step(
            batch, batch_idx, prefix=self.val_loader_names[dataloader_idx]
        )
        if ema:
            self.optimizers().swap_ema()

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_step(
            batch, batch_idx, prefix=self.test_loader_names[dataloader_idx]
        )

    def configure_optimizers(self):

        # Normal parameters
        all_params = list(self.parameters())
        params = [p for p in all_params if not hasattr(p, "_optim")]

        # Construct optimizer, add EMA if necessary
        if self.hparams.train.ema > 0.0:
            optimizer = utils.instantiate(
                registry.optimizer,
                self.hparams.optimizer,
                params,
                wrap=build_ema_optimizer,
                polyak=self.hparams.train.ema,
            )
        else:
            optimizer = utils.instantiate(
                registry.optimizer, self.hparams.optimizer, params
            )

        del self.hparams.optimizer._name_

        # Add parameters with special hyperparameters
        hps = [getattr(p, "_optim") for p in all_params if hasattr(p, "_optim")]
        hps = [
            dict(s) for s in set(frozenset(hp.items()) for hp in hps)
        ]  # Unique dicts
        for hp in hps:
            params = [p for p in all_params if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group(
                {"params": params, **self.hparams.optimizer, **hp}
            )

        # Print optimizer info for debugging
        keys = set(
            [k for hp in hps for k in hp.keys()]
        )  # Get the set of special hparams
        utils.train.log_optimizer(log, optimizer, keys)

        # Configure scheduler
        if "scheduler" not in self.hparams:
            return optimizer
        lr_scheduler = utils.instantiate(
            registry.scheduler, self.hparams.scheduler, optimizer
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": self.hparams.train.interval,  # 'epoch' or 'step'
            "monitor": self.hparams.train.monitor,
            "name": "trainer/lr",  # default is e.g. 'lr-AdamW'
        }
        # See documentation for how to configure the return
        # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.dataset.train_dataloader(**self.hparams.loader)

    def _eval_dataloaders_names(self, loaders, prefix):
        """Process loaders into a list of names and loaders"""
        if utils.is_dict(loaders):
            return [
                f"{prefix}/{k}" if k is not None else prefix for k in loaders.keys()
            ], list(loaders.values())
        elif utils.is_list(loaders):
            return [f"{prefix}/{i}" for i in range(len(loaders))], loaders
        else:
            return [prefix], [loaders]

    def _eval_dataloaders(self):
        # Return all val + test loaders
        val_loaders = self.dataset.val_dataloader(**self.hparams.loader)
        test_loaders = self.dataset.test_dataloader(**self.hparams.loader)
        val_loader_names, val_loaders = self._eval_dataloaders_names(val_loaders, "val")
        test_loader_names, test_loaders = self._eval_dataloaders_names(
            test_loaders, "test"
        )

        # Duplicate datasets for ema
        if self.hparams.train.ema > 0.0:
            val_loader_names += [name + "/ema" for name in val_loader_names]
            val_loaders = val_loaders + val_loaders
            test_loader_names += [name + "/ema" for name in test_loader_names]
            test_loaders = test_loaders + test_loaders

        return val_loader_names + test_loader_names, val_loaders + test_loaders

    def val_dataloader(self):
        val_loader_names, val_loaders = self._eval_dataloaders()
        self.val_loader_names = val_loader_names
        return val_loaders

    def test_dataloader(self):
        test_loader_names, test_loaders = self._eval_dataloaders()
        self.test_loader_names = ["final/" + name for name in test_loader_names]
        return test_loaders


### pytorch-lightning utils and entrypoint


def create_trainer(config, **kwargs):
    callbacks: List[pl.Callback] = []
    logger = None

    # WandB Logging
    if config.get("wandb") is not None:
        # Pass in wandb.init(config=) argument to get the nice 'x.y.0.z' hparams logged
        # Can pass in config_exclude_keys='wandb' to remove certain groups
        import wandb

        logger = WandbLogger(
            config=utils.to_dict(config, recursive=True),
            settings=wandb.Settings(start_method="fork"),
            **config.wandb,
        )

    # Lightning callbacks
    if "callbacks" in config:
        for _name_, callback in config.callbacks.items():
            if config.get("wandb") is None and _name_ in ["learning_rate_monitor"]:
                continue
            log.info(f"Instantiating callback <{registry.callbacks[_name_]}>")
            callback._name_ = _name_
            callbacks.append(utils.instantiate(registry.callbacks, callback))

    # Configure ddp automatically
    if config.trainer.gpus > 1:
        kwargs["plugins"] = [
            pl.plugins.DDPPlugin(
                find_unused_parameters=True,
                gradient_as_bucket_view=False,  # https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#ddp-optimizations
            )
        ]
        kwargs["accelerator"] = "ddp"

    kwargs.update(config.trainer)
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        **kwargs,
    )
    return trainer


def train(config):
    if config.train.seed is not None:
        pl.seed_everything(config.train.seed, workers=True)
    trainer = create_trainer(config)
    model = SequenceLightningModule(config)
    trainer.fit(model)
    if config.train.test:
        trainer.test(model)


def benchmark_step(config):
    """Utility function to benchmark speed of 'stepping', i.e. recurrent view. Unused for main train logic"""
    pl.seed_everything(config.train.seed, workers=True)

    model = SequenceLightningModule(config)
    model.setup()
    model.to("cuda")
    print("Num Parameters: ", sum(p.numel() for p in model.parameters()))
    print(
        "Num Trainable Parameters: ",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    model._on_post_move_to_device()

    for module in model.modules():
        if hasattr(module, "setup_step"):
            module.setup_step()
    model.eval()

    val_dataloaders = model.val_dataloader()
    dl = val_dataloaders[0] if utils.is_list(val_dataloaders) else val_dataloaders

    import benchmark

    for batch in dl:
        benchmark.utils.benchmark(
            model.forward_recurrence,
            batch,
            config.train.benchmark_step_k,
            T=config.train.benchmark_step_T,
        )
        break


@hydra.main(config_path="configs", config_name="config.yaml")
def main(config: OmegaConf):

    # Process config:
    # - register evaluation resolver
    # - filter out keys used only for interpolation
    # - optional hooks, including disabling python warnings or debug friendly configuration
    config = utils.train.process_config(config)

    # Pretty print config using Rich library
    utils.train.print_config(config, resolve=True)

    if config.train.benchmark_step:
        benchmark_step(config)
        exit()

    train(config)


if __name__ == "__main__":
    main()
