import argparse
import os 

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchaudio

from hydra import compose, initialize
from torch.distributions import Categorical
from tqdm.auto import tqdm

from src import utils
from src.dataloaders.audio import mu_law_decode
from src.models.baselines.samplernn import SampleRNN
from src.models.baselines.wavenet import WaveNetModel
from train import SequenceLightningModule

@torch.inference_mode()
def generate(
    model, 
    batch, 
    tau=1.0, 
    prefix=0, 
    T=None, 
    debug=False, 
    top_p=1.0, 
    benchmark=False, 
    calc_logprobs=False,
):
    x, _, *_ = batch
    T = x.shape[1] if T is None else T
    
    # Set up the initial state
    if isinstance(model.model, SampleRNN):
        model._state = None
    elif isinstance(model.model, WaveNetModel):
        model._state = None
        if not benchmark:
            prefix += model.model.receptive_field
            T += model.model.receptive_field
            if x.shape[1] == 1:
                x = x.repeat(1, prefix + 1)
    else:
        model._state = model.model.default_state(*x.shape[:1], device='cuda')

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
        x_t = x_t.to('cuda')
        y_t, *_ = model.encoder(x_t)
        y_t, state = model.model.step(y_t, state=model._state)
        model._state = state
        y_t, *_ = model.decoder(y_t, state)

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
        if t < prefix:
            x_t = x[:, t+1]
        else:
            x_t = y_t

            # Calculate the log-likelihood
            if calc_logprobs:
                probs = probs.squeeze(1)
                if len(y_t.shape) > 1:
                    logprobs += torch.log(probs[torch.arange(probs.shape[0]), y_t.squeeze(1)]).cpu().numpy()
                else:
                    logprobs += torch.log(probs[torch.arange(probs.shape[0]), y_t]).cpu().numpy()
                entropy += -(probs * (probs + 1e-6).log()).sum(dim=-1).cpu().numpy()

        y_all.append(x_t.cpu().squeeze())
    
    y_all = torch.stack(y_all).squeeze()

    if isinstance(model.model, WaveNetModel) and not benchmark:
        y_all = y_all[model.model.receptive_field:]

    if not calc_logprobs:
        if debug:
            y_raw = torch.stack(y_raw)
            return y_all, y_raw
        return y_all
    else:
        assert not debug
        return y_all, logprobs, entropy


def main(args):
    # Initialize hydra
    initialize(config_path="../configs")

    print("Loading model...")

    overrides = []
    if not args.load_data:
        overrides.append('train.disable_dataset=true')

    overrides.append(f'loader.batch_size={args.n_samples}')
    overrides.append(f'experiment={args.model}-{args.dataset}')
    
    # Load in the model config
    if args.model == 'sashimi':
        if args.dataset == 'sc09':
            config = compose(config_name="config.yaml", overrides=overrides + ['model.layer.hurwitz=false', 'decoder.mode=last'])
        else:
            config = compose(config_name="config.yaml", overrides=overrides + ['model.layer.hurwitz=false', 'model.layer.postact=null', 'decoder.mode=last'])
            
    elif args.model == 'wavenet':
        config = compose(config_name="config.yaml", overrides=overrides)
        
    elif args.model == 'samplernn':
        config = compose(config_name="config.yaml", overrides=overrides)

    config = utils.train.process_config(config)
    utils.train.print_config(config, resolve=True)

    # Seed 
    pl.seed_everything(config.train.seed, workers=True)

    assert torch.cuda.is_available(), 'Use a GPU for generation.'

    # Create the Lightning Module
    model = SequenceLightningModule(config)
    model.setup()
    model.to('cuda')

    # Load checkpoint
    state_dict = torch.load(f'sashimi/checkpoints/{args.model}_{args.dataset}.pt', map_location='cuda')
    model.load_state_dict(state_dict)

    # Setup: required for S4 modules in SaShiMi
    for module in model.modules():
        if hasattr(module, 'setup_step'): module.setup_step(mode='dense')
    model.eval()

    if args.load_data:
        # Get the eval dataloaders
        eval_dataloaders = model.val_dataloader()
        dl = eval_dataloaders[0] if args.split == 'val' else eval_dataloaders[1]

        # Construct a batch
        x, _, *_ = next(iter(dl))
        batch = (x.repeat(args.n_reps, 1), None, None)
    else:
        assert args.prefix == 0, 'Only unconditional generation when data is not loaded.'
        batch = (torch.zeros(args.n_samples * args.n_reps, 1).to(torch.long) + 128, None, None)
    
    # Generate
    y, logprobs, _ = generate(
        model, # lightning module (SequenceLightningModule from `train.py`)
        batch, # pass data to condition the generation
        prefix=args.prefix, # length of conditioning prefix
        T=args.sample_len, # length of generated sequence
        top_p=args.top_p, # nucleus sampling: always set to 1.0 for SaShiMi experiments
        tau=args.temp, # temperature: always set to 1.0 for SaShiMi experiments
        calc_logprobs=True, # calc exact likelihoods
    )

    # Decode quantization
    y = mu_law_decode(y.T)

    # Sort based on likelihoods and save
    y = y[np.argsort(logprobs.flatten())]
    for i, d in enumerate(y): 
        torchaudio.save(f'{args.save_dir}/unconditional_{args.dataset}_{args.model}_len_{args.sample_len/16000.:.2f}s_gen_{i+1}.wav', d.unsqueeze(0), 16000)
    np.save(f'{args.save_dir}/unconditional_{args.dataset}_{args.model}_len_{args.sample_len/16000.:.2f}s_logprobs.npy', logprobs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio generation with SaShiMi.')

    parser.add_argument('--model', type=str, default='sashimi', help='Model name', choices=['sashimi', 'wavenet', 'samplernn'])
    parser.add_argument('--dataset', type=str, default='youtubemix', help='Dataset name', choices=['youtubemix', 'sc09'])
    parser.add_argument('--checkpoint_path', type=str, help='Path to checkpoint')
    parser.add_argument('--sample_len', type=int, help='Sample length', default=16000)
    parser.add_argument('--n_samples', type=int, help='Number of samples', default=32)
    parser.add_argument('--n_reps', type=int, help='Number of times to replicate each sample', default=1)
    parser.add_argument('--prefix', type=int, help='Steps to use for conditioning', default=0)
    parser.add_argument('--top_p', type=float, help='Nucleus sampling', default=1.)
    parser.add_argument('--temp', type=float, help='Temperature', default=1.)
    parser.add_argument('--split', type=str, help='If conditioning, which split of the data to use', default='val', choices=['val', 'test'])
    parser.add_argument('--save_dir', type=str, help='Save directory', default='sashimi/samples')
    parser.add_argument('--load_data', help='Load the dataset', action='store_true')
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
