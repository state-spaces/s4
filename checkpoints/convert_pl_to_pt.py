import argparse
import torch
from pathlib import Path

from train import SequenceLightningModule


parser = argparse.ArgumentParser()

parser.add_argument("ckpt_path", type=str)
args = parser.parse_args()

ckpt = torch.load(args.ckpt_path, map_location='cuda')
state_dict = ckpt['state_dict']

torch.save(state_dict, Path(args.ckpt_path).with_suffix(".pt"))
