"""Read pre-tokenized uint32 blocks without downloading data."""
import math

import numpy as np
import torch
from torch.utils.data import Dataset


class Blocks(Dataset):
    def __init__(self, path, seq, offset=0, training=False, seed=20261102):
        if type(seq) is not int or seq < 2:
            raise ValueError('Sequence length must be an integer >= 2')
        self.data = np.memmap(path, dtype='<u4', mode='r')
        self.seq = seq
        self.n = len(self.data) // seq
        self.offset, self.training, self.seed = offset, training, seed
        if self.n < 1:
            raise ValueError('Corpus shorter than one sequence')
        self.stride = 104729
        while math.gcd(self.stride, self.n) != 1:
            self.stride += 2

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        index = ((i + self.offset) * self.stride + self.seed) % self.n if self.training else i
        return torch.from_numpy(self.data[index * self.seq:(index + 1) * self.seq].astype(np.int64))
