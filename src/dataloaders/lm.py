# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import re
import subprocess
from pathlib import Path

from typing import Optional, List, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule


from src.utils import distributed
import src.utils.train
log = src.utils.train.get_logger(__name__)


from src.dataloaders.datasets import SequenceDataset, default_data_path
from src.dataloaders.vocabulary import OpenAIVocab, Vocab
import src.utils as utils
# from tasks.legacy.tasks import LMPerplexity, LMBPC

# TODO: create a package so we don't have to mess with sys.path?
project_root = Path(__file__).parent.parent.absolute()
data_path = Path(__file__).absolute().parent / 'data'

import sys

sys.path.insert(0, str(project_root))

class LMOrderedIterator:
    def __init__(
        self,
        data,
        batch_size,
        l_max,
        batch_first=True,
        # device="cpu",
        # mem_len=None,
        # ext_len=None,
        # warmup=True,
        n_context=1,
        n_epoch_double=0,
        pad_last=False,
        roll_seed=None, # roll data based on seed
        limit_tokens=1.0, # reduce tokens; useful for debugging last batch edge cases
    ):
        """
        data -- LongTensor -- the LongTensor is strictly ordered
        pad_last: whether to pad the last sequence in the batch so that all sequences
            have the same length (l_max).
        """
        self.raw_data = data
        self.batch_size = batch_size
        self.l_max = l_max
        self.batch_first = batch_first
        # self.ext_len = ext_len if ext_len is not None else 0
        # self.mem_len = mem_len
        # self.warmup = warmup
        self.pad_last = pad_last
        self.roll_seed = roll_seed
        self.n_context = n_context
        self.n_epoch_double = n_epoch_double

        # self.device = device
        # self.last_iter = None # AG: this isn't in original repo and doesn't appear to be used
        self.epoch = -1

        # DDP
        self.world_size = distributed.get_world_size()
        self.rank = distributed.get_rank()

        if limit_tokens is not None and 0.0 < limit_tokens < 1.0:
            l_data = int(math.floor(data.size(-1) * limit_tokens))
            self.raw_data = self.raw_data[:l_data]

        self.process()

    def process(self):
        """ Process the data. All logic involving sequence length and batch size should go here """
        assert self.l_max % self.n_context == 0
        self.l_inc = self.l_max // self.n_context

        global_batch_size = self.world_size * self.batch_size

        # Work out how cleanly we can divide the dataset into batch_size parts.
        n_step = self.raw_data.size(-1) // global_batch_size

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        self.data = self.raw_data[: n_step * global_batch_size]

        # Evenly divide the data across the batches.
        self.data = self.data.view(global_batch_size, -1).contiguous().pin_memory() # (global_batch_size, length)

        # Partition data for DistributedDataParallel
        self.data = self.data.chunk(self.world_size, dim=0)[self.rank]

        # Number of mini-batches
        # Need to subtract 1 because target is data shifted by 1
        self.n_batch = (self.data.size(-1) - 1 + self.l_inc - 1) // self.l_inc

    def roll(self, seed):
        rng = torch.Generator()
        rng.manual_seed(seed)
        for i in range(self.data.size(0)):
            row = self.data[i, :]
            shift = torch.randint(0, self.data.size(-1), (1,), generator=rng)
            row = torch.cat((row[shift:], row[:shift]))
            self.data[i, :] = row

    def get_batch(self, i, l_max=None):
        """ Get batch starting at token index i """
        # if l_max is None: l_max = self.l_max
        # seq_len = min(l_max, self.data.size(0) - 1 - i)

        end_idx = min(i + self.l_inc, self.data.size(-1)-1)
        # beg_idx = max(0, i - self.ext_len)
        beg_idx = max(0, end_idx - self.l_max)
        seq_len = end_idx - i

        data = self.data[..., beg_idx:end_idx] # .to(self.device, non_blocking=True)
        target = self.data[..., i+1 : end_idx+1] # .to( self.device, non_blocking=True)

        if self.pad_last and seq_len < self.l_inc:
            data = F.pad(data, (0, self.l_inc - seq_len)) # (batch_size, l_inc)
            target = F.pad(target, (0, self.l_inc - seq_len))
            seq_len = self.l_inc

        if not self.batch_first:
            data = data.transpose(0, 1).contiguous() # (n_batch, l_sequence)
            target = target.transpose(0, 1).contiguous()

        # [21-09-19] Unsqueeze the last dimension so that shape is always (n_batch, l_seq, d_input)
        data = data
        target = target
        return data, target, seq_len

    def get_fixlen_iter(self, start=0): # AG: Don't see start ever used?
        if start != 0:
            start += self.l_max
        for i in range(start, self.data.size(-1) - 1, self.l_inc):
            self.last_iter = i
            yield self.get_batch(i)

    def get_varlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
        l_max = self.l_max + max_deviation * std
        i = start
        while True:
            l_max = self.l_max if np.random.random() < 0.95 else self.l_max / 2.0
            l_max = min(l_max, max(min_len, int(np.random.normal(l_max, std))))
            data, target, seq_len = self.get_batch(i, l_max) # AG: this doesn't appear to work...
            i += seq_len
            yield data, target, seq_len
            if i >= self.data.size(-1) - 2:
                break

    def __iter__(self):
        self.epoch += 1
        if (n := self.n_epoch_double) > 0 and self.epoch > 0 and self.epoch % n == 0:
            if self.batch_size > 1:
                log.info(f"LM Iterator doubling length from {self.l_max} to {self.l_max*2}")
                self.l_max *= 2
                self.batch_size //= 2
                self.process()

        if self.roll_seed is not None:
            self.roll(self.roll_seed + self.epoch)
        return self.get_fixlen_iter()

    def __len__(self):
        return self.n_batch


class LMShuffledIterator(object):
    def __init__(
        self, data, batch_size, l_max, device="cpu", ext_len=None, shuffle=False
    ):
        """
        data -- list[LongTensor] -- there is no order among the LongTensors
        """
        self.data = data

        self.batch_size = batch_size
        self.l_max = l_max
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.shuffle = shuffle

    def get_sent_stream(self):
        # index iterator
        epoch_indices = (
            np.random.permutation(len(self.data))
            if self.shuffle
            else np.array(range(len(self.data)))
        )

        # sentence iterator
        for idx in epoch_indices:
            yield self.data[idx]

    def stream_iterator(self, sent_stream):
        # streams for each data in the batch
        streams = [None] * self.batch_size

        data = torch.LongTensor(self.l_max, self.batch_size)
        target = torch.LongTensor(self.l_max, self.batch_size)

        n_retain = 0

        while True:
            # data   : [n_retain+l_max x batch_size]
            # target : [l_max x batch_size]
            data[n_retain:].fill_(-1)
            target.fill_(-1)

            valid_batch = True

            for i in range(self.batch_size):
                n_filled = 0
                try:
                    while n_filled < self.l_max:
                        if streams[i] is None or len(streams[i]) <= 1:
                            streams[i] = next(sent_stream)
                        # number of new tokens to fill in
                        n_new = min(len(streams[i]) - 1, self.l_max - n_filled)
                        # first n_retain tokens are retained from last batch
                        data[
                            n_retain + n_filled : n_retain + n_filled + n_new,
                            i,
                        ] = streams[i][:n_new]
                        target[n_filled : n_filled + n_new, i] = streams[i][
                            1 : n_new + 1
                        ]
                        streams[i] = streams[i][n_new:]
                        n_filled += n_new
                except StopIteration:
                    valid_batch = False
                    break

            if not valid_batch:
                return

            data = data.to(self.device)
            target = target.to(self.device)

            yield data, target, self.l_max

            n_retain = min(data.size(0), self.ext_len)
            if n_retain > 0:
                data[:n_retain] = data[-n_retain:]
            data.resize_(n_retain + self.l_max, data.size(1))

    def __iter__(self):
        # sent_stream is an iterator
        sent_stream = self.get_sent_stream()

        for batch in self.stream_iterator(sent_stream):
            yield batch


class LMMultiFileIterator(LMShuffledIterator):
    def __init__(
        self,
        paths,
        vocab,
        batch_size,
        l_max,
        device="cpu",
        ext_len=None,
        shuffle=False,
    ):

        self.paths = paths
        self.vocab = vocab

        self.batch_size = batch_size
        self.l_max = l_max
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.shuffle = shuffle

    def get_sent_stream(self, path):
        sents = self.vocab.encode_file(path, add_double_eos=True)
        if self.shuffle:
            np.random.shuffle(sents)
        sent_stream = iter(sents)

        return sent_stream

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.paths)

        for path in self.paths:
            # sent_stream is an iterator
            sent_stream = self.get_sent_stream(path)
            for batch in self.stream_iterator(sent_stream):
                yield batch


# class WikiText2(LightningDataModule):
class WikiText2(SequenceDataset):
    _name_ = "wt2"

    # Vocab arguments
    vocab_kwargs = {"special": ["<eos>"], "lower_case": False}
    encode_kwargs = {"ordered": True}

    # Embedding arguments (adaptive softmax / word embeddings)
    # default_task = {
    #     'adaptive': False,
    #     'div_val': 1,
    #     'cutoffs': [],
    #     'tie_weights': False,
    #     'tie_projs': [False],
    # }
    @property
    def default_task(self):
        return {
            '_target_': 'tasks.tasks.LMTask',
            'tied': False,
            'rescale': True,
            # init_cfg,
            'metrics': ['ppl'],
            'init_cfg': {
                'init': 'normal',  # Parameter initializer to use
                'init_range': 0.01,  # Parameters initialized by U(-init_range, init_range)
                'init_std': 0.02,  # Parameters initialized by N(0, init_std)
                'proj_init_std': 0.01, # Separate std for projection params
            }
        }

    # Task class / constructor
    # task_cls = LMPerplexity

    # @property
    # def l_output(self):
    #     return self.l_max

    init_defaults = {
        # Dataset arguments
        'l_max': 512,
        'bpe': False,
        'roll_seed': 42,
        'test_split': True,
        # Task / Embedding arguments
        # 'task': None,
    }

    @property
    def n_tokens(self):
        return len(self.vocab)

    # def __init__(
    #     self,
    #     data_dir,
    #     d_embed, init_cfg, task=None, # Task / Embedding arguments
    #     bpe=False,
    #     l_max=None,
    #     pad_last=False,
    #     roll_seed=42,
    #     eval={
    #         'l_max': None,
    #         'pad_last': False,
    #         'roll_seed': None,
    #     },
    #     **kwargs,
    #     # TODO kwargs is here to absorb things like 'num_workers' and 'pin_memory' which should really be part of every dataset
    # ):
    #     super().__init__()
        # if data_dir is None: self.data_dir = Path(data_dir) / self._name_
        # # self.d_embed = d_embed
        # # self.init_cfg = init_cfg
        # if bpe:
        #     self.vocab = OpenAIVocab()
        # else:
        #     self.vocab = Vocab(**self.vocab_kwargs)

        # # Loader arguments
        # assert l_max is not None
        # self.l_max = l_max
        # self.pad_last = pad_last
        # self.roll_seed = roll_seed

        # self.eval = DictConfig(eval)
        # if self.eval.l_max is None: self.eval.l_max = self.l_max

        # if task is not None:
        #     self.task.update(task)

    def prepare_data(self):
        # [21-09-23] probably broken
        if not self.data_dir.exists():
            subprocess.run(
                [
                    str(project_root / "data" / "getdata.sh"),
                    self._name_,
                    str(self.data_dir.parent.absolute()),
                ],
                check=True,
            )

    def setup(self, stage=None): # [21-09-10 AG]: TODO shouldn't this tokenization happen in the prepare_data? since we're caching it it doesn't really matter, but still
        if self.data_dir is None: self.data_dir = default_data_path / self._name_
        # self.d_embed = d_embed
        # self.init_cfg = init_cfg
        if self.bpe:
            self.vocab = OpenAIVocab()
        else:
            self.vocab = Vocab(**self.vocab_kwargs)

        # Loader arguments
        if not self._load_from_cache():
            logging.info(f"Producing dataset {self._name_}...")
            self._vocab_count()
            self.vocab.build_vocab()
            self.train = self.vocab.encode_file(
                str(self.data_dir / "train.txt"), **self.encode_kwargs
            )
            self.valid = self.vocab.encode_file(
                str(self.data_dir / "valid.txt"), **self.encode_kwargs
            )
            self.test = self.vocab.encode_file(
                str(self.data_dir / "test.txt"), **self.encode_kwargs
            )
            self._save_to_cache()

        # No test set if specified
        if not self.test_split:
            self.test = None

        # Define task
        print("Vocab size:", len(self.vocab))
        # self.task = self.task_cls(len(self.vocab), self.d_embed, init_cfg=self.init_cfg, **self.task_args)
        # self.d_input = self.d_output = self.d_embed

    def _vocab_count(self):
        self.vocab.count_file(self.data_dir / "train.txt")
        self.vocab.count_file(self.data_dir / "valid.txt")
        self.vocab.count_file(self.data_dir / "test.txt")

    def _save_to_cache(self):
        cache_path = self.data_dir / f"cache.pt" # TODO name could include vocab_kwargs to disambiguate
        with distributed.sync_workers() as rank:
            if rank == 0:
                try:
                    torch.save(
                        (self.vocab, self.train, self.valid, self.test),
                        cache_path,
                    )
                    logging.info(f"Saved dataset to {cache_path}...")
                except:
                    pass

    def _load_from_cache(self):
        cache_path = self.data_dir / f"cache.pt"
        if cache_path.exists():
            logging.info("Loading cached dataset...")
            self.vocab, self.train, self.valid, self.test = torch.load(
                cache_path
            )
            return True
        else:
            return False

    def train_dataloader(self, eval=None, **kwargs):
        # TODO kwargs absorbs num_workers
        return LMOrderedIterator(
            self.train,
            roll_seed=self.roll_seed,
            **kwargs,
        )

    # def val_dataloader(self, batch_size, **kwargs):
    def _eval_dataloader(self, dataset, eval=None, **loader_args):
        if dataset is None: return None
        # Make eval a list of dictionaries
        if eval is None: eval = {}
        if not utils.is_list(eval):
            eval = [eval]
        # Each eval setting overrides the train setting
        for eval_args in eval:
            for k in loader_args:
                if eval_args.get(k, None) is None:
                    eval_args[k] = loader_args[k]
            print("eval loader:", eval_args)
        loaders = [LMOrderedIterator(dataset, **eval_args) for eval_args in eval]
        if len(loaders) == 1: return loaders[0]
        return loaders

    def val_dataloader(self, **kwargs):
        return self._eval_dataloader(self.valid, **kwargs)
        # return LMOrderedIterator(
        #     self.valid,
        #     batch_size,
        #     **self.eval,
        # )
        # for k in train_args:
        #     if eval_args.get(k, None) is None:
        #         eval_args[k] = v
        # return LMOrderedIterator(self.valid, **eval_args)

    def test_dataloader(self, **kwargs):
        return self._eval_dataloader(self.test, **kwargs)


class WikiText103(WikiText2):
    _name_ = "wt103"

    @property
    def default_task(self):
        return {
            # 'adaptive': True,
            '_target_': 'tasks.tasks.AdaptiveLMTask',
            'div_val': 1,
            'cutoffs': [19997, 39997, 199997],
            'tie_weights': True,
            'tie_projs': [False] + [True, True, True], # * len(cutoffs),
            'init_cfg': {
                'init': 'normal',  # Parameter initializer to use
                'init_range': 0.01,  # Parameters initialized by U(-init_range, init_range)
                'init_std': 0.02,  # Parameters initialized by N(0, init_std)
                'proj_init_std': 0.01, # Separate std for projection params
            }
        }

    def _vocab_count(self):
        print(self.data_dir)
        self.vocab.count_file(self.data_dir / "train.txt")


class PennTreeBank(WikiText2):

    _name_ = "ptb"
    vocab_kwargs = {"special": ["<eos>"], "lower_case": True}

    # task_cls = LMBPC

class EnWik8(WikiText2):
    _name_ = "enwik8"

    vocab_kwargs = {}
    encode_kwargs = {"ordered": True, "add_eos": False}

    # task_cls = LMBPC
    @property
    def default_task(self):
        return {
            '_target_': 'tasks.tasks.LMTask',
            'tied': False,
            'rescale': True,
            # init_cfg,
            'metrics': ['ppl'],
            # 'init_cfg': {
            #     'init': 'normal',  # Parameter initializer to use
            #     'init_range': 0.01,  # Parameters initialized by U(-init_range, init_range)
            #     'init_std': 0.02,  # Parameters initialized by N(0, init_std)
            #     'proj_init_std': 0.01, # Separate std for projection params
            # }
        }

class Text8(EnWik8):

    _name_ = "text8"
    # task_cls = LMBPC


class LM1B(WikiText2):
    # [21-09-08 AG]: this looks very out of date, the __init__ function should be inherited

    _name_ = "lm1b"
    vocab_kwargs = {"special": [], "lower_case": False}
    cutoffs = [59997, 99997, 639997]
    tie_projs = [False] + [False] * len(cutoffs)

    def __init__(self, data_dir, bpe=False, *args, **kwargs):
        LightningDataModule.__init__(self)
        self.data_dir = Path(data_dir)
        # self.vocab_type = vocab
        if bpe:
            self.vocab = OpenAIVocab()
        else:
            self.vocab = Vocab(
                vocab_file=self.data_dir / "1b_word_vocab.txt",
                **self.vocab_kwargs,
            )

    def setup(self, stage=None):
        if not self._load_from_cache():
            logging.info(f"Producing dataset {self._name_}...")
            # the vocab will load from file when build_vocab() is called
            self.vocab.build_vocab()
            train_paths = list(
                (
                    self.data_dir
                    / "1-billion-word-language-modeling-benchmark-r13output"
                    / "training-monolingual.tokenized.shuffled"
                ).glob("news.en-*")
            )
            self.train = train_paths
            self.valid = self.vocab.encode_file(
                str(self.data_dir / "valid.txt"),
                ordered=False,
                add_double_eos=True,
            )
            self.test = self.vocab.encode_file(
                str(self.data_dir / "test.txt"),
                ordered=False,
                add_double_eos=True,
            )
            self._save_to_cache()

    def train_dataloader(self, *args, **kwargs):
        kwargs["shuffle"] = True
        return LMMultiFileIterator(self.train, self.vocab, *args, **kwargs)

    def val_dataloader(self, *args, **kwargs):
        return LMShuffledIterator(self.valid, *args, **kwargs)

    def test_dataloader(self, *args, **kwargs):
        return LMShuffledIterator(self.test, *args, **kwargs)



class Corpus(object):
    # AG: only used in get_lm_corpus which is only called in the unit test
    def __init__(self, path, dataset, vocab, *args, **kwargs):
        self.dataset = dataset
        if vocab == "word":
            self.vocab = Vocab(*args, **kwargs)
        elif vocab == "bpe":
            self.vocab = OpenAIVocab()
        else:
            raise RuntimeError("Unsupported vocab")

        if self.dataset in ["ptb", "wt2", "enwik8", "text8"]:
            self.vocab.count_file(os.path.join(path, "train.txt"))
            self.vocab.count_file(os.path.join(path, "valid.txt"))
            self.vocab.count_file(os.path.join(path, "test.txt"))
        elif self.dataset == "wt103":
            self.vocab.count_file(os.path.join(path, "train.txt"))
        elif self.dataset == "lm1b":
            train_path_pattern = os.path.join(
                path,
                "1-billion-word-language-modeling-benchmark-r13output",
                "training-monolingual.tokenized.shuffled",
                "news.en-*",
            )
            train_paths = glob.glob(train_path_pattern)
            # the vocab will load from file when build_vocab() is called

        self.vocab.build_vocab()

        if self.dataset in ["ptb", "wt2", "wt103"]:
            self.train = self.vocab.encode_file(
                os.path.join(path, "train.txt"), ordered=True
            )
            self.valid = self.vocab.encode_file(
                os.path.join(path, "valid.txt"), ordered=True
            )
            self.test = self.vocab.encode_file(
                os.path.join(path, "test.txt"), ordered=True
            )
        elif self.dataset in ["enwik8", "text8"]:
            self.train = self.vocab.encode_file(
                os.path.join(path, "train.txt"), ordered=True, add_eos=False
            )
            self.valid = self.vocab.encode_file(
                os.path.join(path, "valid.txt"), ordered=True, add_eos=False
            )
            self.test = self.vocab.encode_file(
                os.path.join(path, "test.txt"), ordered=True, add_eos=False
            )
        elif self.dataset == "lm1b":
            self.train = train_paths
            self.valid = self.vocab.encode_file(
                os.path.join(path, "valid.txt"),
                ordered=False,
                add_double_eos=True,
            )
            self.test = self.vocab.encode_file(
                os.path.join(path, "test.txt"),
                ordered=False,
                add_double_eos=True,
            )

    def get_iterator(self, split, *args, **kwargs):
        if split == "train":
            if self.dataset in ["ptb", "wt2", "wt103", "enwik8", "text8"]:
                data_iter = LMOrderedIterator(self.train, *args, **kwargs)
            elif self.dataset == "lm1b":
                kwargs["shuffle"] = True
                data_iter = LMMultiFileIterator(
                    self.train, self.vocab, *args, **kwargs
                )
        elif split in ["valid", "test"]:
            data = self.valid if split == "valid" else self.test
            if self.dataset in ["ptb", "wt2", "wt103", "enwik8", "text8"]:
                data_iter = LMOrderedIterator(data, *args, **kwargs)
            elif self.dataset == "lm1b":
                data_iter = LMShuffledIterator(data, *args, **kwargs)

        return data_iter

def get_lm_corpus(data_dir, name, vocab):
    if vocab == "word":
        fn = os.path.join(data_dir, "cache.pt")
    elif vocab == "bpe":
        fn = os.path.join(data_dir, "cache.pt.bpe")
    else:
        raise RuntimeError("Unsupported vocab")

    if os.path.exists(fn):
        logging.info("Loading cached dataset...")
        corpus = torch.load(fn)
    else:
        logging.info("Producing dataset {}...".format(name))
        kwargs = {}
        if name in ["wt103", "wt2"]:
            kwargs["special"] = ["<eos>"]
            kwargs["lower_case"] = False
        elif name == "ptb":
            kwargs["special"] = ["<eos>"]
            kwargs["lower_case"] = True
        elif name == "lm1b":
            kwargs["special"] = []
            kwargs["lower_case"] = False
            kwargs["vocab_file"] = os.path.join(data_dir, "1b_word_vocab.txt")
        elif name in ["enwik8", "text8"]:
            pass

        corpus = Corpus(data_dir, name, vocab, **kwargs)
        # with distributed.sync_workers() as rank:
        #     if rank == 0:
        #         torch.save(corpus, fn)

    return corpus


def tokenize_raw(text, lang="en"):
    # AG: Not used?
    import sacremoses

    mt = sacremoses.MosesTokenizer(lang)
    text = mt.tokenize(text, return_str=True)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"&apos;", "'", text)
    text = re.sub(r"(\d)\.(\d)", r"\1 @.@ \2", text)
    text = re.sub(r"(\d),(\d)", r"\1 @,@ \2", text)
    text = re.sub(r"(\w)-(\w)", r"\1 @-@ \2", text)
    return text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="unit test")
    parser.add_argument(
        "--datadir",
        type=str,
        default="../data/text8",
        help="location of the data corpus",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="text8",
        choices=["ptb", "wt2", "wt103", "lm1b", "enwik8", "text8"],
        help="dataset name",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    corpus = get_lm_corpus(args.datadir, args.dataset, vocab="word")
    logging.info("Vocab size : {}".format(len(corpus.vocab.idx2sym)))
