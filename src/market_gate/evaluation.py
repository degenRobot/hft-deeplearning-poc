"""Identical held-out proxy utilities for a gate and fixed teaching baselines."""

from pathlib import Path

import numpy as np

from .experts import EXPERT_IDS
from .gate import NumpyMLPGate, static_weights, uniform_weights
from .training import (
    TrainingExample,
    build_examples,
    build_frame_dataset,
    chronological_split,
    dataset_evidence,
    load_recording,
    provenance,
)


def compare_utilities(
    examples: list[TrainingExample],
    neural_weights: np.ndarray,
) -> dict[str, object]:
    """Score every method on exactly the same future-mid proxy labels."""
    utilities = np.stack([example.utilities for example in examples])
    weights = np.asarray(neural_weights, dtype=np.float64)
    if (
        weights.shape != utilities.shape
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0)
        or not np.allclose(weights.sum(axis=1), 1)
    ):
        raise ValueError("neural weights must be finite probability rows matching examples")
    strategies = {
        "neural": weights,
        "uniform": np.array(list(uniform_weights().values())),
        "static": np.array(list(static_weights().values())),
        **{name: np.eye(len(EXPERT_IDS))[i] for i, name in enumerate(EXPERT_IDS)},
    }
    means = {
        name: float((utilities * value).sum(axis=1).mean()) for name, value in strategies.items()
    }
    return {
        "objective": "mean expert-proxy score times future mid-return bps; higher is better",
        "held_out_examples": len(examples),
        "mean_proxy_utility": means,
        "neural_minus_baseline": {
            name: means["neural"] - score for name, score in means.items() if name != "neural"
        },
        "limitations": "Overlapping windows; no P&L, search, significance or generalisation claim.",
    }


def evaluate_recording(
    recording_path: Path,
    model_path: Path,
    *,
    horizon_frames: int = 5,
) -> dict[str, object]:
    """Read-only evaluation; never trains, selects a model, or writes an artifact."""
    data = build_frame_dataset(load_recording(recording_path))
    examples = build_examples(
        data.values, data.mids, 30, horizon_frames, close_ts_ms=data.close_ts_ms
    )
    train, validation = chronological_split(examples, len(data.values))
    gate = NumpyMLPGate(model_path)
    weights = np.array(
        [list(gate.predict(e.features.reshape(30, 10)).values()) for e in validation]
    )
    return {
        "schema_version": 2,
        "dataset": dataset_evidence(data, examples, train, validation, 30, horizon_frames),
        "evaluation": compare_utilities(validation, weights),
        "provenance": provenance(
            recording_path,
            model_path,
            {
                "lookback_frames": 30,
                "horizon_frames": horizon_frames,
                "validation_fraction": 0.2,
                "clock": "event_ts_ms",
                "gap_policy": "exclude_crossing_windows",
                "feature_history": "reset_after_gap",
                "label": "offline_expert_proxy_future_mid_return_bps_v2",
            },
        ),
        "limitations": [
            "Offline event-time proxies are not equivalent to live receive-time state.",
            "Model provenance does not establish training/holdout independence.",
        ],
    }
