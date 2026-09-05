"""Bounded supervised learning and online policy updates on recorded public markets.

This educational contextual-bandit experiment never places orders or changes the
running gate. Both local and remote callers consume the same event generator.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .training import (
    EXPERT_NAMES,
    FEATURE_NAMES,
    TrainingExample,
    _save_gate_model,
    build_examples,
    build_frame_dataset,
    file_sha256,
    load_recording,
)

LOOKBACK = 30
HORIZON = 5
HISTORY_EMBARGO = 30
MAX_EPOCHS = 50
MAX_RL_STEPS = 300
MAX_RECORDING_BYTES = 100_000_000
MAX_FRAMES = 20_000


def split_lab_examples(examples: list[TrainingExample], frame_count: int, max_rl_steps: int):
    """Disjoint 50/30/20 periods, including the hidden feature-history dependency."""
    first, second = int(frame_count * 0.5), int(frame_count * 0.8)
    supervised = [e for e in examples if e.target_frame < first]
    candidates = [
        e for e in examples if e.start_frame >= first + HISTORY_EMBARGO and e.target_frame < second
    ]
    holdout = [e for e in examples if e.start_frame >= second + HISTORY_EMBARGO]
    # Finish observing one sampled action's reward before choosing the next action.
    rl: list[TrainingExample] = []
    previous_target = -1
    for example in candidates:
        if example.input_end_ts_ms >= previous_target:
            rl.append(example)
            previous_target = example.target_ts_ms
        if len(rl) >= max_rl_steps:
            break
    if len(supervised) < 30 or len(rl) < 15 or len(holdout) < 15:
        raise ValueError(
            "insufficient contiguous data after history embargo and causal RL sampling: "
            f"need 30 supervised / 15 RL / 15 holdout; got "
            f"{len(supervised)} / {len(rl)} / {len(holdout)}"
        )
    return supervised, rl, holdout


def _make_model():
    from torch import nn

    return nn.Sequential(
        nn.Linear(300, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3)
    )


def _forward(model, inputs):
    import torch

    hidden_1 = model[1](model[0](inputs))
    hidden_2 = model[3](model[2](hidden_1))
    logits = model[4](hidden_2)
    return logits, torch.softmax(logits, dim=-1), hidden_1, hidden_2


def _export(model, mean, scale, destination: Path):
    """Fold train-only normalization into the portable raw-input first layer."""
    import torch

    exported = copy.deepcopy(model)
    with torch.no_grad():
        weight = model[0].weight.detach()
        exported[0].weight.copy_(weight / scale)
        exported[0].bias.copy_(model[0].bias - (weight * (mean / scale)).sum(dim=1))
    _save_gate_model(exported, destination)


def _evaluate(model, features, utilities):
    import torch

    with torch.no_grad():
        return float((_forward(model, features)[1] * utilities).sum(dim=1).mean())


def _update(model, optimizer, loss):
    import torch

    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
    layers = []
    for index, name in (
        (0, "input_to_hidden_1"),
        (2, "hidden_1_to_hidden_2"),
        (4, "hidden_2_to_output"),
    ):
        parameters = [model[index].weight, model[index].bias]
        norm = math.sqrt(sum(float((p.grad.detach() ** 2).sum()) for p in parameters))
        layers.append({"name": name, "gradient_norm": norm})
    optimizer.step()
    for row, index in zip(layers, (0, 2, 4), strict=True):
        parameters = [
            (f"{index}.weight", model[index].weight),
            (f"{index}.bias", model[index].bias),
        ]
        row["weight_delta_norm"] = math.sqrt(
            sum(float(((value.detach() - before[name]) ** 2).sum()) for name, value in parameters)
        )
        row["weight_norm"] = math.sqrt(
            sum(float((value.detach() ** 2).sum()) for _, value in parameters)
        )
    return layers


def _event(value: dict):
    # Reject NaN/Infinity at the producer boundary; callers can safely JSON-stream.
    json.dumps(value, allow_nan=False)
    return value


def train_lab(
    recording_path: Path,
    output_dir: Path,
    *,
    epochs: int = 12,
    max_rl_steps: int = 120,
    seed: int = 7,
):
    """Yield dataset, genuine optimizer steps, then a frozen-holdout result.

    Replay policy updates are contextual-bandit REINFORCE with a running reward
    baseline, not exchange execution or a full market-making RL simulator.
    """
    import torch

    if type(epochs) is not int or not 1 <= epochs <= MAX_EPOCHS:
        raise ValueError(f"epochs must be an integer in 1..{MAX_EPOCHS}")
    if type(max_rl_steps) is not int or not 15 <= max_rl_steps <= MAX_RL_STEPS:
        raise ValueError(f"max_rl_steps must be an integer in 15..{MAX_RL_STEPS}")
    if type(seed) is not int or not 0 <= seed <= 2**31 - 1:
        raise ValueError("seed must be an integer in 0..2147483647")
    recording_path, output_dir = Path(recording_path), Path(output_dir)
    if not recording_path.is_file() or recording_path.stat().st_size > MAX_RECORDING_BYTES:
        raise ValueError("recording must exist and be at most 100 MB")
    paths = {
        "supervised": output_dir / "gate-supervised.npz",
        "adapted": output_dir / "gate-adapted.npz",
        "receipt": output_dir / "training-lab.json",
    }
    if any(path.exists() or path.is_symlink() for path in paths.values()):
        raise ValueError("training outputs must be new files; refusing to overwrite")
    if recording_path.resolve() in {p.resolve() for p in paths.values()}:
        raise ValueError("recording and training outputs must be separate")
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    events = load_recording(recording_path)
    dataset = build_frame_dataset(events)
    if len(dataset.values) > MAX_FRAMES:
        raise ValueError(f"bounded teaching run supports at most {MAX_FRAMES} closed frames")
    examples = build_examples(
        dataset.values, dataset.mids, LOOKBACK, HORIZON, close_ts_ms=dataset.close_ts_ms
    )
    supervised, rl, holdout = split_lab_examples(examples, len(dataset.values), max_rl_steps)
    x_train = torch.from_numpy(np.stack([e.features for e in supervised]))
    y_train = torch.from_numpy(np.stack([e.utilities for e in supervised]))
    mean = x_train.mean(dim=0)
    scale = x_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    reward_scale = max(float(y_train.std(unbiased=False)), 1e-3)
    inputs = (x_train - mean) / scale
    target_probabilities = torch.softmax(y_train / reward_scale, dim=1)
    validation_inputs = (torch.from_numpy(np.stack([e.features for e in holdout])) - mean) / scale
    validation_utilities = torch.from_numpy(np.stack([e.utilities for e in holdout]))
    dataset_info = {
        "symbol": events[0].symbol,
        "venue": events[0].venue,
        "event_count": len(events),
        "frame_count": len(dataset.values),
        "supervised_examples": len(supervised),
        "rl_examples": len(rl),
        "holdout_examples": len(holdout),
        "sha256": file_sha256(recording_path),
        "feature_names": FEATURE_NAMES,
        "expert_names": list(EXPERT_NAMES),
        "lookback_frames": LOOKBACK,
        "horizon_frames": HORIZON,
        "history_embargo_frames": HISTORY_EMBARGO,
        "clock": "event_ts_ms",
        "normalization_fit": "supervised prefix only; folded into exported first layer",
        "displayed_features": "raw features; Torch training uses train-only normalized inputs",
        "supervised_last_target_ts_ms": supervised[-1].target_ts_ms,
        "rl_first_start_ts_ms": rl[0].start_ts_ms,
        "rl_last_target_ts_ms": rl[-1].target_ts_ms,
        "holdout_first_start_ts_ms": holdout[0].start_ts_ms,
        "holdout_last_target_ts_ms": holdout[-1].target_ts_ms,
        "source_mode": "recorded market replay",
    }
    yield _event({"kind": "dataset", "dataset": dataset_info})
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    step_number = 0
    for epoch in range(epochs):
        logits, outputs, hidden_1, hidden_2 = _forward(model, inputs)
        loss = -(target_probabilities * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
        sample = epoch % len(supervised)
        example = supervised[sample]
        before = outputs[sample].detach().tolist()
        activations = {
            "hidden_1": hidden_1[sample].detach().tolist(),
            "hidden_2": hidden_2[sample].detach().tolist(),
        }
        layers = _update(model, optimizer, loss)
        with torch.no_grad():
            after = _forward(model, inputs[sample])[1].tolist()
        step_number += 1
        yield _event(
            {
                "kind": "step",
                "step": {
                    "step": step_number,
                    "phase": "supervised",
                    "phase_step": epoch + 1,
                    "phase_steps": epochs,
                    "loss": float(loss.detach()),
                    "reward": None,
                    "action": None,
                    "input_end_ts_ms": example.input_end_ts_ms,
                    "target_ts_ms": example.target_ts_ms,
                    "features": example.features.reshape(30, 10).tolist(),
                    "activations": activations,
                    "outputs_before": before,
                    "outputs_after": after,
                    "target_probabilities": target_probabilities[sample].tolist(),
                    "layers": layers,
                    "batch_size": len(supervised),
                    "loss_scope": "full supervised batch; pictured window is one batch example",
                },
            }
        )
    supervised_model = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    baseline = 0.0
    for index, example in enumerate(rl):
        features = (torch.from_numpy(example.features) - mean) / scale
        logits, outputs, hidden_1, hidden_2 = _forward(model, features)
        # The policy sees inputs only. Sampling precedes accessing future utility.
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        reward = float(example.utilities[int(action)])
        scaled_reward = math.tanh(reward / reward_scale)
        advantage = scaled_reward - baseline
        loss = -distribution.log_prob(action) * advantage
        before = outputs.detach().tolist()
        activations = {
            "hidden_1": hidden_1.detach().tolist(),
            "hidden_2": hidden_2.detach().tolist(),
        }
        layers = _update(model, optimizer, loss)
        with torch.no_grad():
            after = _forward(model, features)[1].tolist()
        step_number += 1
        yield _event(
            {
                "kind": "step",
                "step": {
                    "step": step_number,
                    "phase": "rl",
                    "phase_step": index + 1,
                    "phase_steps": len(rl),
                    "loss": float(loss.detach()),
                    "reward": reward,
                    "action": int(action),
                    "advantage": advantage,
                    "baseline": baseline,
                    "scaled_reward": scaled_reward,
                    "input_end_ts_ms": example.input_end_ts_ms,
                    "target_ts_ms": example.target_ts_ms,
                    "reward_observed_ts_ms": example.target_ts_ms,
                    "features": example.features.reshape(30, 10).tolist(),
                    "activations": activations,
                    "outputs_before": before,
                    "outputs_after": after,
                    "layers": layers,
                },
            }
        )
        baseline = 0.95 * baseline + 0.05 * scaled_reward
    evaluation = {
        "supervised": _evaluate(supervised_model, validation_inputs, validation_utilities),
        "adapted": _evaluate(model, validation_inputs, validation_utilities),
        "uniform": float(validation_utilities.mean()),
        **{
            name: float(validation_utilities[:, i].mean())
            for i, name in enumerate(("microprice", "trade_flow", "reversion"))
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _export(supervised_model, mean, scale, paths["supervised"])
    _export(model, mean, scale, paths["adapted"])
    artifacts = {
        name: {"path": str(path), "sha256": file_sha256(path)}
        for name, path in paths.items()
        if name != "receipt"
    }
    receipt = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": dataset_info,
        "config": {
            "epochs": epochs,
            "max_rl_steps": max_rl_steps,
            "seed": seed,
            "supervised_learning_rate": 1e-3,
            "rl_learning_rate": 2e-4,
            "reward_scale_supervised_only": reward_scale,
            "torch_threads": 1,
            "deterministic_algorithms": True,
        },
        "evaluation": evaluation,
        "evaluation_objective": "frozen final holdout mean expert-proxy utility in bps",
        "artifacts": {
            name: value | {"path": Path(value["path"]).name} for name, value in artifacts.items()
        },
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "steps": step_number,
        "duration_seconds": round(time.monotonic() - started, 3),
        "provenance": {
            "source_sha256": {
                name: file_sha256(Path(__file__).parent / name)
                for name in (
                    "training_lab.py",
                    "training.py",
                    "features.py",
                    "experts.py",
                    "gate.py",
                )
            },
            "normalization_sha256": hashlib.sha256(
                mean.numpy().tobytes() + scale.numpy().tobytes()
            ).hexdigest(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "limitations": [
            "Online policy updates consume recorded public data, not a live exchange feed.",
            "REINFORCE is an expert-selection contextual bandit, not an execution simulator.",
            "Rewards are future-mid expert proxies; they are not fills or realized trading P&L.",
            "Offline event-time ordering differs from live Binance receive-time sequencing.",
            "quote_updates counts retained recorded books after sampling, not all live updates.",
            "Held-out windows overlap; this short experiment does not establish generalisation.",
            "Outputs are separate teaching artifacts; the live inference model is not replaced.",
        ],
    }
    _event(receipt)
    with paths["receipt"].open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    yield _event(
        {
            "kind": "completed",
            **receipt,
            "artifacts": artifacts,
            "receipt_path": str(paths["receipt"]),
        }
    )
