"""Tiny reproducible synthetic training example; writes an optional gate artifact."""

from pathlib import Path

import numpy as np
import torch
from torch import nn


def main() -> None:
    torch.manual_seed(7)
    features = torch.randn(128, 300)
    utilities = torch.stack((features[:, -5], features[:, -4] * 0.6, -features[:, -1]), dim=1)
    model = nn.Sequential(
        nn.Linear(300, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(40):
        weights = torch.softmax(model(features), dim=1)
        loss = -(weights * utilities).sum(dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    destination = Path("models/gate-demo.npz")
    destination.parent.mkdir(exist_ok=True)
    first, second, third = (layer for layer in model if isinstance(layer, nn.Linear))
    np.savez(
        destination,
        schema_version=np.array("gate-npz-v1"),
        w1=first.weight.detach().numpy().T,
        b1=first.bias.detach().numpy(),
        w2=second.weight.detach().numpy().T,
        b2=second.bias.detach().numpy(),
        w3=third.weight.detach().numpy().T,
        b3=third.bias.detach().numpy(),
    )
    print(f"saved {destination}")


if __name__ == "__main__":
    main()
