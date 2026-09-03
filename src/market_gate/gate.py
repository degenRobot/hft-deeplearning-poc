"""Weight policy and optional PyTorch 300 -> 64 -> 32 -> 3 gate."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .experts import EXPERT_IDS


def normalize(weights: Sequence[float]) -> dict[str, float]:
    values = np.maximum(np.asarray(weights, dtype=float), 0.0)
    total = float(values.sum())
    values = values / total if total else np.full(3, 1 / 3)
    return dict(zip(EXPERT_IDS, (float(item) for item in values), strict=True))


def static_weights() -> dict[str, float]:
    return normalize((0.50, 0.35, 0.15))


def uniform_weights() -> dict[str, float]:
    return normalize((1.0, 1.0, 1.0))


class TinyMLPGate:
    """Optional trainable gate. It is never an execution authority."""

    model_version = "unloaded-demo-v1"

    def __init__(self, model_path: str | None = None) -> None:
        self.model = None
        if model_path and Path(model_path).exists():
            import torch

            self.model = torch.nn.Sequential(
                torch.nn.Linear(300, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 3),
            )
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state)
            self.model.eval()
            self.model_version = Path(model_path).stem

    def predict(self, window: Sequence[Sequence[float]]) -> dict[str, float]:
        flat = np.asarray(window, dtype=np.float32).reshape(-1)
        if flat.size != 300:
            raise ValueError("gate requires a causal 30 x 10 feature window")
        if self.model is not None:
            import torch

            with torch.no_grad():
                return normalize(torch.softmax(self.model(torch.from_numpy(flat)), dim=0).numpy())
        # A deterministic demonstration policy keeps replay useful before training.
        logits = (flat[-5] * 4 + flat[-6] * 1000, flat[-4] * 2, -flat[-1] * 800)
        return normalize(np.exp(np.clip(logits, -4, 4)))


def blend_and_smooth(
    proposed: dict[str, float],
    previous: dict[str, float],
    influence: float,
    smoothing: float = 0.45,
) -> dict[str, float]:
    baseline = uniform_weights()
    blended = {
        key: (1 - influence) * baseline[key] + influence * proposed[key] for key in EXPERT_IDS
    }
    smoothed = {
        key: smoothing * previous[key] + (1 - smoothing) * blended[key] for key in EXPERT_IDS
    }
    return normalize(tuple(smoothed[key] for key in EXPERT_IDS))
