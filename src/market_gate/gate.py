"""Torch-free runtime inference for the compact 300 -> 64 -> 32 -> 3 gate."""

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


class NumpyMLPGate:
    """A small exported MLP. Loading/inference deliberately has no Torch dependency."""

    def __init__(self, model_path: str | Path) -> None:
        self.path = Path(model_path)
        with np.load(self.path) as artifact:
            self.w1 = artifact["w1"]
            self.b1 = artifact["b1"]
            self.w2 = artifact["w2"]
            self.b2 = artifact["b2"]
            self.w3 = artifact["w3"]
            self.b3 = artifact["b3"]
            schema = artifact["schema_version"]
        if schema.item() != "gate-npz-v1" or self.w1.shape != (300, 64) or self.w3.shape != (32, 3):
            raise ValueError("unsupported gate artifact schema or dimensions")
        self.model_version = self.path.stem

    def predict(self, window: Sequence[Sequence[float]]) -> dict[str, float]:
        flat = np.asarray(window, dtype=np.float32).reshape(-1)
        if flat.size != 300:
            raise ValueError("gate requires a causal 30 x 10 feature window")
        hidden_1 = np.maximum(flat @ self.w1 + self.b1, 0)
        hidden_2 = np.maximum(hidden_1 @ self.w2 + self.b2, 0)
        logits = hidden_2 @ self.w3 + self.b3
        logits -= logits.max()
        return normalize(np.exp(np.clip(logits, -50, 50)))


def load_numpy_gate(model_path: str | Path | None) -> NumpyMLPGate | None:
    return NumpyMLPGate(model_path) if model_path and Path(model_path).exists() else None


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
