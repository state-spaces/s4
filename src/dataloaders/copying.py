import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data.dataset import IterableDataset
import numpy as np

from src.utils import distributed


def np_copying_data(L, M, A, batch_shape=()):
    seq = np.random.randint(low=1, high=A-1, size=batch_shape+(M,))
    zeros_x = np.zeros(batch_shape+(L,))
    markers = (A-1) * np.ones(batch_shape+(M,))
    zeros_y = np.zeros(batch_shape+(M+L,))

    x_ = np.concatenate([seq, zeros_x, markers], axis=-1)
    y_ = np.concatenate([zeros_y, seq], axis=-1)
    x = F.one_hot(torch.tensor(x_, dtype=torch.int64), A).float()
    y = torch.tensor(y_, dtype=torch.int64)
    return x, y

def torch_copying_data(L, M, A, variable=False, batch_shape=()):
    tokens = torch.randint(low=1, high=A-1, size=batch_shape+(M,))
    # zeros_x = torch.zeros(batch_shape+(L,), dtype=torch.long)
    if variable:
        # inds = torch.randint(low=0, high=L+M, size=batch_shape+(M,))
        total_batch = np.prod(batch_shape)
        inds = torch.stack([
            torch.randperm(L+M)[:M]
            for _ in range(total_batch)
            ], 0)
        inds = inds.reshape(batch_shape+(M,))
        inds, _ = inds.sort()
    else:
        inds = torch.arange(M).repeat(batch_shape+(1,))
    zeros_x = torch.zeros(batch_shape+(M+L,), dtype=torch.long)
    zeros_x.scatter_(-1, inds, tokens)
    markers = (A-1) * torch.ones(batch_shape+(M,), dtype=torch.long)
    # zeros_y = torch.zeros(batch_shape+(M+L,), dtype=torch.long)

    x_ = torch.cat([zeros_x, markers], dim=-1)
    y_ = torch.cat([tokens], dim=-1)
    x = F.one_hot(x_, A).float()
    # y = torch.tensor(y_, dtype=torch.int64)
    # y = y_.long()
    y = y_
    # print("TYPE", x.dtype, y.dtype)
    return x, y

class CopyingTrainDataset(torch.utils.data.IterableDataset):
    def __init__(self, L, M, A, samples_per_epoch=-1):
        """
        L: number of noise tokens
        M: number of memorization tokens
        A: size of dictionary
        """
        super().__init__()
        self.L = L
        self.M = M
        self.A = A
        self.samples_per_epoch = samples_per_epoch

    def __iter__(self):
        if self.samples_per_epoch < 0:
            while True:
                x, y = torch_copying_data(self.L, self.M, self.A)
                yield x, y
        else:
            for _ in range(self.samples_per_epoch):
                x, y = torch_copying_data(self.L, self.M, self.A)
                yield x, y


class CopyingEvalDataset(torch.utils.data.TensorDataset):
    def __init__(self, L, M, A, samples):
        self.L = L
        self.M = M
        self.A = A
        self.samples = samples
        all_x, all_y = torch_copying_data(self.L, self.M, self.A, batch_shape=(self.samples,))
        super().__init__(all_x, all_y)

def copying_static_dataset(L, M, A, variable, samples):
    all_x, all_y = torch_copying_data(L, M, A, variable, batch_shape=(samples,))
    print("Constructing Copying dataset of shape", all_x.shape)
    ds = torch.utils.data.TensorDataset(all_x, all_y)
    return ds


class CopyOrderedIterator:
    def __init__(
        self,
        data,
        batch_size,
        l_max,
        offset=0,
        # device="cpu",
        # mem_len=None,
        # ext_len=None,
        # warmup=True,
        # roll_seed=None, # roll data based on seed
    ):
        """
        data -- LongTensor -- the LongTensor is strictly ordered
        pad_last: whether to pad the last sequence in the batch so that all sequences
            have the same length (l_max).
        """
        assert len(data.shape) == 1
        self.data_x = data[offset:]
        self.data_y = data[:-offset]
        self.data = torch.stack([data_x, data_y], dim=0) # (2, L, D)

        self.batch_size = batch_size
        self.l_max = l_max
        # self.roll_seed = roll_seed

        self.epoch = 0

        # DDP
        self.world_size = distributed.get_world_size()
        self.rank = distributed.get_rank()

        self.process()

    def process(self):
        """ Process the data. All logic involving sequence length and batch size should go here """
        assert self.l_max % self.n_overlaps == 0
        self.l_inc = self.l_max // self.n_overlaps

        global_batch_size = self.world_size * self.batch_size

        # Work out how cleanly we can divide the dataset into batch_size parts.
        n_tokens_per_batch = global_batch_size * self.l_max
        n_step = self.data.size(-2) // n_tokens_per_batch

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        self.data = self.data[: n_step * global_batch_size]

        # Evenly divide the data across the batches.
        self.data = self.data.view(2, global_batch_size, -1, self.data.size(-1)).contiguous().pin_memory() # (2, global_batch_size, length)

        # Partition data for DistributedDataParallel
        self.data = self.data.chunk(self.world_size, dim=-3)[self.rank] # (2, batch_size, length, dim)

        # Number of mini-batches
        # Need to subtract 1 because target is data shifted by 1
        assert self.data.size(-2) % self.l_max == 0
        self.n_batch = (self.data.size(-2)) // self.l_max


    def roll(self, seed):
        rng = torch.Generator()
        rng.manual_seed(seed)
        for i in range(self.data.size(1)):
            row = self.data[:, i]
            shift = torch.randint(0, self.data.size(0), (1,), generator=rng)
            row = torch.cat((row[shift:], row[:shift]))
            self.data[:, i] = row

    def get_batch(self, i, l_max=None):
        """ Get batch starting at token index i """
        if l_max is None: l_max = self.l_max
        seq_len = l_max # min(l_max, self.data.size(0) - 1 - i)

        end_idx = i + seq_len
        # beg_idx = max(0, i - self.ext_len)
        beg_idx = i

        data = self.data[:, :, beg_idx:end_idx, :] # (2, B, L, D)
        x, y = data # (B, L, D)

        return x, y

    def get_fixlen_iter(self):
        for i in range(0, self.data.size(-1), self.l_max):
            yield self.get_batch(i)

    def __iter__(self):
        self.epoch += 1
        # if self.roll_seed is not None:
        #     self.roll(self.roll_seed + self.epoch)
        return self.get_fixlen_iter()

    def __len__(self):
        return self.n_batch



if __name__ == '__main__':
    # a = torch_copying_data(20, 5, 10, batch_shape=(3,))
    # print(a)

    ds = CopyingTrainDataset(10, 5, 10, samples_per_epoch=5)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=2)
    for (x, y) in enumerate(loader):
        print(x, y)

    print("Copying Evaluation Dataset")
    # eval_ds = CopyingEvalDataset(10, 5, 10, samples=5)
    eval_ds = copying_static_dataset(10, 3, 10, variable=True, samples=5)
    loader = torch.utils.data.DataLoader(eval_ds, batch_size=2, num_workers=2)
    for (x, y) in loader:
        print(x, y)

