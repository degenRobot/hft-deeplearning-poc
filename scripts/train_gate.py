"""Tiny reproducible synthetic training example; writes an optional gate artifact."""

from pathlib import Path

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
    destination = Path("artifacts/gate-demo.pt")
    destination.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), destination)
    print(f"saved {destination}")


if __name__ == "__main__":
    main()
