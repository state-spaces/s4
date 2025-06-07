import torch

class LineDataset(torch.utils.data.TensorDataset):
    def __init__(self, seq_len=24, pred_len=12, n_samples=1000,
                 slope_range=(0.1, 1.0), intercept_range=(0.0, 1.0),
                 noise_std=0.0, seed=0):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_samples = n_samples
        self.slope_range = slope_range
        self.intercept_range = intercept_range
        self.noise_std = noise_std
        self.seed = seed

        generator = torch.Generator().manual_seed(seed)
        total_len = seq_len + pred_len
        t = torch.arange(total_len, dtype=torch.float32)
        slopes = torch.empty(n_samples).uniform_(slope_range[0], slope_range[1], generator=generator)
        intercepts = torch.empty(n_samples).uniform_(intercept_range[0], intercept_range[1], generator=generator)
        lines = slopes[:, None] * t + intercepts[:, None]
        if noise_std > 0:
            lines += noise_std * torch.randn(n_samples, total_len, generator=generator)
        x = lines[:, :seq_len].unsqueeze(-1)
        y = lines[:, seq_len:].unsqueeze(-1)
        super().__init__(x, y)
        self.forecast_horizon = pred_len
