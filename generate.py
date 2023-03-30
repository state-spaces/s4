import argparse
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchaudio

import hydra
from omegaconf import OmegaConf
from torch.distributions import Categorical
from tqdm.auto import tqdm

from src import utils
from src.dataloaders.audio import mu_law_decode
from src.models.baselines.wavenet import WaveNetModel
from train import SequenceLightningModule

def test_step(model):
    B, L = 2, 64
    x = torch.ones(B, L, dtype=torch.long).to('cuda')

    # Forward
    batch = (x, None)
    y, _, _ = model(batch) # Forward pass expects a batch which has both x and y (inputs and targets)

    # Step
    model._reset_state(batch, device='cuda')
    ys = []
    for x_ in torch.unbind(x, dim=-1):
        y_ = model.step(x_)
        ys.append(y_)
    ys = torch.stack(ys, dim=1)

    print(torch.norm(y-ys))

    breakpoint()

@torch.inference_mode()
def generate(
    model,
    batch,
    tau=1.0,
    l_prefix=0,
    T=None,
    debug=False,
    top_p=1.0,
    benchmark=False,
    return_logprobs=False,
):

    x, _, *_ = batch # (B, L)
    x = x.to('cuda')
    T = x.shape[1] if T is None else T

    # Special logic for WaveNet
    if isinstance(model.model, WaveNetModel) and not benchmark:
        l_prefix += model.model.receptive_field
        T += model.model.receptive_field
        x = F.pad(x, (model.model.receptive_field, 0), value=128)

    # Set up the initial state
    model._reset_state(batch, device='cuda')

    # First sample
    x_t = x[:, 0]
    y_all = []
    logprobs = np.zeros(x.shape[0])
    entropy = np.zeros(x.shape[0])

    if debug:
        y_raw = []

    # Generation loop
    for t in tqdm(range(T)):

        # Step through the model with the current sample
        y_t = model.step(x_t)

        # Handle special loss functions such as ProjectedAdaptiveSoftmax
        if hasattr(model.loss, "compute_logits"): y_t = model.loss.compute_logits(y_t)

        if debug:
            y_raw.append(y_t.detach().cpu())

        # Output distribution
        probs = F.softmax(y_t, dim=-1)

        # Optional: nucleus sampling
        if top_p < 1.0:
            sorted_probs = probs.sort(dim=-1, descending=True)
            csum_probs = sorted_probs.values.cumsum(dim=-1) > top_p
            csum_probs[..., 1:] = csum_probs[..., :-1].clone()
            csum_probs[..., 0] = 0
            indices_to_remove = torch.zeros_like(csum_probs)
            indices_to_remove[torch.arange(sorted_probs.indices.shape[0])[:, None].repeat(1, sorted_probs.indices.shape[1]).flatten(), sorted_probs.indices.flatten()] = csum_probs.flatten()
            y_t = y_t + indices_to_remove.int() * (-1e20)

        # Sample from the distribution
        y_t = Categorical(logits=y_t/tau).sample()

        # Feed back to the model
        if t < l_prefix-1:
            x_t = x[:, t+1]
        else:
            x_t = y_t

            # Calculate the log-likelihood
            if return_logprobs:
                probs = probs.squeeze(1)
                if len(y_t.shape) > 1:
                    logprobs += torch.log(probs[torch.arange(probs.shape[0]), y_t.squeeze(1)]).cpu().numpy()
                else:
                    logprobs += torch.log(probs[torch.arange(probs.shape[0]), y_t]).cpu().numpy()
                entropy += -(probs * (probs + 1e-6).log()).sum(dim=-1).cpu().numpy()

        y_all.append(x_t.cpu())
        # y_all.append(y_t.cpu())

    y_all = torch.stack(y_all, dim=1) # (batch, length)

    if isinstance(model.model, WaveNetModel) and not benchmark:
        y_all = y_all[:, model.model.receptive_field:]


    if not return_logprobs:
        if debug:
            y_raw = torch.stack(y_raw)
            return y_all, y_raw
        return y_all
    else:
        assert not debug
        return y_all, logprobs, entropy


@hydra.main(config_path="configs", config_name="generate.yaml")
def main(config: OmegaConf):
    ### See configs/generate.yaml for descriptions of generation flags ###

    # Load train config from existing Hydra experiment
    if config.experiment_path is not None:
        config.experiment_path = hydra.utils.to_absolute_path(config.experiment_path)
        experiment_config = OmegaConf.load(os.path.join(config.experiment_path, '.hydra', 'config.yaml'))
        # config = OmegaConf.merge(config, experiment_config)
        config.model = experiment_config.model
        config.task = experiment_config.task
        config.encoder = experiment_config.encoder
        config.decoder = experiment_config.decoder
        config.dataset = experiment_config.dataset
        config.loader = experiment_config.loader

    # Special override flags
    if not config.load_data:
        OmegaConf.update(config, "train.disable_dataset", True)

    if config.n_batch is None:
        config.n_batch = config.n_samples
    OmegaConf.update(config, "loader.batch_size", config.n_batch)

    # Create the Lightning Module - same as train.py

    config = utils.train.process_config(config)
    utils.train.print_config(config, resolve=True)

    print("Loading model...")
    assert torch.cuda.is_available(), 'Use a GPU for generation.'

    if config.train.seed is not None:
        pl.seed_everything(config.train.seed, workers=True)

    # Define checkpoint path smartly
    if not config.experiment_path:
        ckpt_path = hydra.utils.to_absolute_path(config.checkpoint_path)
    else:
        ckpt_path = os.path.join(config.experiment_path, config.checkpoint_path)
    print("Full checkpoint path:", ckpt_path)

    # Load model
    if ckpt_path.endswith('.ckpt'):
        model = SequenceLightningModule.load_from_checkpoint(ckpt_path, config=config)
        model.to('cuda')
    elif ckpt_path.endswith('.pt'):
        model = SequenceLightningModule(config)
        model.to('cuda')

        # Load checkpoint
        state_dict = torch.load(ckpt_path, map_location='cuda')
        model.load_state_dict(state_dict)

    # Setup: required for S4 modules in SaShiMi
    for module in model.modules():
        if hasattr(module, '_setup_step'): module._setup_step()
    model.eval()

    if config.load_data:
        # Get the eval dataloaders
        eval_dataloaders = model.val_dataloader()
        dl = eval_dataloaders[0] if config.split == 'val' else eval_dataloaders[1]
    else:
        assert config.l_prefix == 0, 'Only unconditional generation when data is not loaded.'

    # Handle save directory intelligently
    if config.save_dir:
        save_dir = hydra.utils.to_absolute_path(config.save_dir)
    else:
        save_dir = os.path.join(os.getcwd(), "samples/")
    os.makedirs(save_dir, exist_ok=True)

    # Test
    if config.test_model:
        test_step(model)


    # Generate
    assert config.n_samples % config.n_batch == 0, "For convenience, n_samples should be a multiple of n_batch"
    y = []
    logprobs =  []
    for _ in range(config.n_samples // config.n_batch):
        # Construct a batch
        if config.load_data:
            x, _, *_ = next(iter(dl))
            batch = (x.repeat(config.n_reps, 1), None, None)
        else:
            batch = (torch.zeros(config.n_batch * config.n_reps, 1).to(torch.long) + 128, None, None)

        _y, _logprobs, _ = generate(
            model, # lightning module (SequenceLightningModule from `train.py`)
            batch, # pass data to condition the generation
            l_prefix=config.l_prefix, # length of conditioning prefix
            T=config.l_sample, # length of generated sequence
            top_p=config.top_p, # nucleus sampling: always set to 1.0 for SaShiMi experiments
            tau=config.temp, # temperature: always set to 1.0 for SaShiMi experiments
            return_logprobs=True, # calc exact likelihoods
        )
        y.append(_y)
        logprobs.append(_logprobs)

    # Sort based on likelihoods and save
    y = torch.cat(y, dim=0)
    logprobs = np.concatenate(logprobs, axis=0)
    y = y[np.argsort(logprobs.flatten())]

    # Decode quantization
    if config.decode == 'audio':
        print("Saving samples into:", save_dir)
        y = mu_law_decode(y)
        for i, d in enumerate(y):
            filename = f'{save_dir}/unconditional_{config.dataset._name_}_{config.model._name_}_len_{config.l_sample/16000.:.2f}s_gen_{i+1}.wav'
            torchaudio.save(filename, d.unsqueeze(0), 16000)
        np.save(f'{save_dir}/unconditional_{config.dataset._name_}_{config.model._name_}_len_{config.l_sample/16000.:.2f}s_logprobs.npy', logprobs)
    elif config.decode == 'text':
        y = [model.dataset.vocab.get_symbols(_y) for _y in y]
        breakpoint() # Inspect output manually for now
    else: pass


if __name__ == "__main__":
    main()
