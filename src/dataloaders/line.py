from src.dataloaders.base import SequenceDataset
from .datasets.line import LineDataset

class Line(SequenceDataset):
    _name_ = "line"
    d_input = 1
    d_output = 1

    @property
    def init_defaults(self):
        return {
            "seq_len": 24,
            "pred_len": 12,
            "n_train": 1000,
            "n_val": 200,
            "n_test": 200,
            "slope_range": (0.1, 1.0),
            "intercept_range": (0.0, 1.0),
            "noise_std": 0.0,
            "seed": 0,
        }

    @property
    def l_output(self):
        return self.pred_len

    def setup(self):
        self.dataset_train = LineDataset(
            self.seq_len, self.pred_len, self.n_train,
            self.slope_range, self.intercept_range,
            self.noise_std, seed=self.seed
        )
        self.dataset_val = LineDataset(
            self.seq_len, self.pred_len, self.n_val,
            self.slope_range, self.intercept_range,
            self.noise_std, seed=self.seed + 1
        )
        self.dataset_test = LineDataset(
            self.seq_len, self.pred_len, self.n_test,
            self.slope_range, self.intercept_range,
            self.noise_std, seed=self.seed + 2
        )
        # forecast horizon property used by forecasting task
        self.dataset_train.forecast_horizon = self.pred_len
        self.dataset_val.forecast_horizon = self.pred_len
        self.dataset_test.forecast_horizon = self.pred_len

    def __str__(self):
        return f"line{self.seq_len}_{self.pred_len}"
