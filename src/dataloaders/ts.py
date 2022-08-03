"""Time series datasets, especially for medical time series"""


import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.dataloaders.base import default_data_path, SequenceDataset, deprecated

class BIDMC(SequenceDataset):
    """BIDMC datasets for Respiratory Rate / Heart Rate / Oxygen Saturation regression"""

    _name_ = "bidmc"
    d_input = 2

    @property
    def d_output(self):
        return 2 if self.prediction else 1

    @property
    def l_output(self):
        return 4000 if self.prediction else 0

    @property
    def init_defaults(self):
        return {
            "target": "RR",  # 'RR' | 'HR' | 'SpO2'
            "prediction": False,
            "reshuffle": True,
        }

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / self._name_

        split = "reshuffle" if self.reshuffle else "original"
        # X: (dataset_size, length, d_input)
        # y: (dataset_size)
        X_train = np.load(self.data_dir / self.target / split / "trainx.npy")
        y_train = np.load(self.data_dir / self.target / split / "trainy.npy")
        X_val = np.load(self.data_dir / self.target / split / "validx.npy")
        y_val = np.load(self.data_dir / self.target / split / "validy.npy")
        X_test = np.load(self.data_dir / self.target / split / "testx.npy")
        y_test = np.load(self.data_dir / self.target / split / "testy.npy")

        if self.prediction:
            y_train = np.pad(X_train[:, 1:, :], ((0, 0), (0, 1), (0, 0)))
            y_val = np.pad(X_val[:, 1:, :], ((0, 0), (0, 1), (0, 0)))
            y_test = np.pad(X_test[:, 1:, :], ((0, 0), (0, 1), (0, 0)))

        self.dataset_train = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )

        self.dataset_val = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val)
        )

        self.dataset_test = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test), torch.FloatTensor(y_test)
        )

    def __str__(self):
        split = "reshuffle" if self.reshuffle else "original"
        return f"BIDMC{self.target}_{split}"

