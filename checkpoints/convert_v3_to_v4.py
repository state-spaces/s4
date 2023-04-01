"""Convert a V3 model to V4. See checkpoints/README.md for usage."""

from tqdm.auto import tqdm
import hydra
import torch
import numpy as np
from pathlib import Path
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.nn.modules import module
import torch.nn.functional as F
from torch.distributions import Categorical
from src import utils
from einops import rearrange, repeat, reduce

from train import SequenceLightningModule
from omegaconf import OmegaConf



def convert_dt(state_dict):
    """Unsqueeze log_dt shape to match new shape."""
    new_state_dict = {}

    for k, v in state_dict.items():
        # Unsqueeze log_dt shape [D] -> [D, 1]
        if "log_dt" in k:
            v = v.unsqueeze(dim=-1)
        new_key = k.replace('log_dt', 'inv_dt')
        new_state_dict[new_key] = v
    return new_state_dict

def convert_a(state_dict):
    """Convert names of A_real and A_imag inside kernel."""
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace('inv_w_real', 'A_real')
        k = k.replace('log_w_real', 'A_real')
        k = k.replace('w_imag', 'A_imag')
        new_state_dict[k] = v

    # Negative A imaginary part
    for k, v in new_state_dict.items():
        if k.endswith('A_imag'):
            new_state_dict[k] = -v
    return new_state_dict

def convert_kernel(state_dict):
    """Replace nested kernel with flat kernel and replace L param."""
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace('kernel.kernel.L', 'kernel.kernel.l_kernel')
        k = k.replace('kernel.kernel', 'kernel')
        new_state_dict[k] = v
    return new_state_dict

def convert_conv(state_dict):
    """Move FFTConv parameters a layer deeper."""
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace('layer.kernel', 'layer.layer.kernel')
        k = k.replace('layer.D', 'layer.layer.D')
        new_state_dict[k] = v
    return new_state_dict

def convert_checkpoint(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cuda')

    if ckpt_path.endswith('.ckpt'):
        state_dict = checkpoint['state_dict']
    elif ckpt_path.endswith('.pt'):
        state_dict = checkpoint
    else:
        raise NotImplementedError

    new_state_dict = convert_dt(state_dict)
    new_state_dict = convert_a(new_state_dict)
    new_state_dict = convert_kernel(new_state_dict)
    new_state_dict = convert_conv(new_state_dict)

    if ckpt_path.endswith('.ckpt'):
        checkpoint['state_dict'] = new_state_dict
    else:
        checkpoint = new_state_dict


    return checkpoint


@hydra.main(config_path="../configs", config_name="generate.yaml")
def main(config: OmegaConf):

    # Load train config from existing Hydra experiment
    if config.experiment_path is not None:
        config.experiment_path = hydra.utils.to_absolute_path(config.experiment_path)
        experiment_config = OmegaConf.load(os.path.join(config.experiment_path, '.hydra', 'config.yaml'))
        config.model = experiment_config.model
        config.task = experiment_config.task
        config.encoder = experiment_config.encoder
        config.decoder = experiment_config.decoder
        config.dataset = experiment_config.dataset
        config.loader = experiment_config.loader

    config = utils.train.process_config(config)
    utils.train.print_config(config, resolve=True)

    if config.train.seed is not None:
        pl.seed_everything(config.train.seed, workers=True)

    # Define checkpoint path smartly
    if not config.experiment_path:
        ckpt_path = hydra.utils.to_absolute_path(config.checkpoint_path)
    else:
        ckpt_path = os.path.join(config.experiment_path, config.checkpoint_path)
    print("Full checkpoint path:", ckpt_path)

    # Port checkpoint
    checkpoint = convert_checkpoint(ckpt_path)

    print("Finished converting checkpoint.")

    # Test single batch
    if config.test_model:
        # Load checkpoint
        model = SequenceLightningModule(config)
        model.to('cuda')
        if ckpt_path.endswith('.ckpt'):
            model.load_state_dict(checkpoint['state_dict'])
        elif ckpt_path.endswith('.pt'):
            model.load_state_dict(checkpoint)

        # Dataloader
        val_dataloaders = model.val_dataloader()
        loader = val_dataloaders[0] if isinstance(val_dataloaders, list) else val_dataloaders

        model = model.to('cuda')
        model.eval()
        batch = next(iter(loader))
        batch = (batch[0].cuda(), batch[1].cuda(), batch[2])
        with torch.no_grad():
            x, y, w = model.forward(batch)
            loss = model.loss_val(x, y, **w)
            print("Single batch loss:", loss)

        ## Use PL test to calculate final metrics
        from train import create_trainer
        trainer = create_trainer(config)
        trainer.test(model)

    path = Path(ckpt_path).absolute()
    filename_new = path.stem + "_v4" + path.suffix
    print("Saving to", filename_new)
    torch.save(checkpoint, path.parent / filename_new)

if __name__ == '__main__':
    main()
