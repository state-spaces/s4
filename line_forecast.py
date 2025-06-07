import torch
import torch.nn as nn
from models.s4.s4d import S4D
from torch.utils.data import DataLoader

from src.dataloaders.datasets.line import LineDataset


class ForecastModel(nn.Module):
    def __init__(self, d_model=64, n_layers=2, dropout=0.0):
        super().__init__()
        self.encoder = nn.Linear(1, d_model)
        self.s4_layers = nn.ModuleList(
            [S4D(d_model, dropout=dropout, transposed=True) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = x.transpose(-1, -2)
        for layer, norm in zip(self.s4_layers, self.norms):
            z, _ = layer(x)
            x = norm((x + z).transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(-1, -2)
        x_last = x[:, -1]
        out = self.decoder(x_last)
        return out


def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = LineDataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = ForecastModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(10):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y.squeeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(loader.dataset)
        print(f"Epoch {epoch+1}: loss {avg_loss:.6f}")


if __name__ == "__main__":
    train_model()
