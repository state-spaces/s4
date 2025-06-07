import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.s4.s4 import S4Block
from src.dataloaders.datasets.line import LineDataset


def plot_predictions(x, y, preds, filename="line_forecast_plot.png"):
    """Plot the input sequence, target, and generated prediction."""
    x = x.squeeze(-1).cpu().numpy()
    y = y.squeeze(-1).cpu().numpy()
    preds = preds.squeeze(-1).cpu().numpy()
    seq_len = len(x)
    t_input = list(range(seq_len))
    t_future = list(range(seq_len, seq_len + len(y)))

    plt.figure()
    plt.plot(t_input, x, label="input")
    plt.plot(t_future, y, label="target")
    plt.plot(t_future, preds, "--", label="generated")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("value")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to {filename}")



class ForecastModel(nn.Module):
    def __init__(self, d_model=64, n_layers=2, dropout=0.0):
        super().__init__()
        self.encoder = nn.Linear(1, d_model)
        self.s4_layers = nn.ModuleList(

            [
                S4Block(d_model, transposed=False, dropout=dropout)
                for _ in range(n_layers)
            ]

        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.decoder = nn.Linear(d_model, 1)


    def setup_step(self):
        for layer in self.s4_layers:
            layer.setup_step()

    def default_state(self, batch_size, device=None):
        return [
            layer.default_state(batch_size, device=device) for layer in self.s4_layers
        ]

    def step(self, x_t, states):
        x = self.encoder(x_t)
        new_states = []
        for layer, norm, s in zip(self.s4_layers, self.norms, states):
            y, s = layer.step(x, s)
            x = norm(x + y)
            new_states.append(s)
        out = self.decoder(x)
        return out, new_states

    def forward(self, x):
        x = self.encoder(x)
        for layer, norm in zip(self.s4_layers, self.norms):
            z, _ = layer(x)
            x = norm(x + z)
        x_last = x[:, -1]
        out = self.decoder(x_last)
        return out


def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use a 1-step forecast horizon so the model output matches the target
    train_dataset = LineDataset(pred_len=1)
    val_dataset = LineDataset(pred_len=1, seed=1)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = ForecastModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(10):
        model.train()

        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).squeeze(1)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        avg_train = train_loss / len(train_loader.dataset)

        model.eval()
        plot_predictions(x[0], y_true[0], preds[0])
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device).squeeze(1)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
        avg_val = val_loss / len(val_loader.dataset)

        print(f"Epoch {epoch+1}: train {avg_train:.6f}, val {avg_val:.6f}")

    # Demonstrate autoregressive generation using the trained model
    model.eval()
    with torch.no_grad():
        x, y_true = next(iter(val_loader))
        x = x.to(device)
        y_true = y_true.to(device)
        model.setup_step()
        state = model.default_state(x.size(0), device=device)
        for t in range(x.size(1)):
            _, state = model.step(x[:, t], state)
        preds = []
        x_t = x[:, -1]
        for _ in range(y_true.size(1)):
            out, state = model.step(x_t, state)
            preds.append(out)
            x_t = out
        preds = torch.stack(preds, dim=1)
        print("Target:", y_true[0].squeeze().cpu().numpy())
        print("Preds :", preds[0].squeeze().cpu().numpy())



if __name__ == "__main__":
    train_model()
