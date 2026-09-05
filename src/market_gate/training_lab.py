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
from .training_options import TrainingOptions

LOOKBACK = 30
HORIZON = 5
HISTORY_EMBARGO = 30
MAX_RECORDING_BYTES = 100_000_000
MAX_FRAMES = 20_000


def load_lab_dataset(path: Path):
    """Load real book/trade frames or explicitly identified historical candle proxies."""
    if not path.is_file() or path.stat().st_size > MAX_RECORDING_BYTES:
        raise ValueError("recording must exist and be at most 100 MB")
    with path.open(encoding="utf-8") as handle:
        first = next((json.loads(line) for line in handle if line.strip()), None)
    if isinstance(first, dict) and first.get("kind") == "candle":
        from .historical import load_candle_frames

        dataset, metadata = load_candle_frames(path)
    else:
        events = load_recording(path)
        dataset = build_frame_dataset(events)
        metadata = {
            "symbol": events[0].symbol,
            "venue": events[0].venue,
            "event_count": len(events),
            "feature_names": FEATURE_NAMES,
            "source_mode": "recorded market replay",
            "limitations": [],
        }
    if len(dataset.values) > MAX_FRAMES:
        raise ValueError(f"bounded teaching run supports at most {MAX_FRAMES} closed frames")
    return dataset, metadata


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


def _make_model(hidden_1: int = 64, hidden_2: int = 32):
    from torch import nn

    return nn.Sequential(
        nn.Linear(300, hidden_1),
        nn.ReLU(),
        nn.Linear(hidden_1, hidden_2),
        nn.ReLU(),
        nn.Linear(hidden_2, 3),
    )


def _forward(model, inputs):
    import torch

    hidden_1 = model[1](model[0](inputs))
    hidden_2 = model[3](model[2](hidden_1))
    logits = model[4](hidden_2)
    return logits, torch.softmax(logits, dim=-1), hidden_1, hidden_2


def _export(model, mean, scale, destination: Path, *, teaching_only: bool = False):
    """Fold train-only normalization into the portable raw-input first layer."""
    import torch

    # Folding can subtract large nearly equal terms for constant input columns.
    # Keep the portable weights in float64 so raw-input inference retains precision.
    exported = copy.deepcopy(model).double()
    with torch.no_grad():
        weight = model[0].weight.detach().double()
        mean, scale = mean.double(), scale.double()
        exported[0].weight.copy_(weight / scale)
        exported[0].bias.copy_(model[0].bias.double() - (weight * (mean / scale)).sum(dim=1))
    widths = [exported[0].out_features, exported[2].out_features]
    if widths == [64, 32] and not teaching_only:
        _save_gate_model(exported, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Teaching architectures are explicitly incompatible with the fixed live gate.
        with destination.open("xb") as handle:
            np.savez(
                handle,
                schema_version=np.array("training-mlp-v1"),
                layer_sizes=np.array([300, *widths, 3]),
                **{
                    key: value.detach().cpu().numpy()
                    for number, index in enumerate((0, 2, 4), start=1)
                    for key, value in (
                        (f"w{number}", exported[index].weight.T),
                        (f"b{number}", exported[index].bias),
                    )
                },
            )


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
    hidden_1: int = 64,
    hidden_2: int = 32,
    learning_rate: float = 0.001,
):
    """Yield dataset, genuine optimizer steps, then a frozen-holdout result.

    Replay policy updates are contextual-bandit REINFORCE with a running reward
    baseline, not exchange execution or a full market-making RL simulator.
    """
    options = TrainingOptions(
        hidden_1=hidden_1,
        hidden_2=hidden_2,
        epochs=epochs,
        learning_rate=learning_rate,
        max_rl_steps=max_rl_steps,
        seed=seed,
    ).validate()
    import torch

    recording_path, output_dir = Path(recording_path), Path(output_dir)
    if not recording_path.is_file() or recording_path.stat().st_size > MAX_RECORDING_BYTES:
        raise ValueError("recording must exist and be at most 100 MB")
    dataset, input_metadata = load_lab_dataset(recording_path)
    candle_mode = input_metadata["source_mode"] == "historical_candles_1s"
    runtime_compatible = [options.hidden_1, options.hidden_2] == [64, 32] and not candle_mode
    artifact_prefix = "gate" if runtime_compatible else "training-mlp"
    artifact_schema = "gate-npz-v1" if runtime_compatible else "training-mlp-v1"
    paths = {
        "supervised": output_dir / f"{artifact_prefix}-supervised.npz",
        "adapted": output_dir / f"{artifact_prefix}-adapted.npz",
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
        "hidden_sizes": [options.hidden_1, options.hidden_2],
        "activation_sample_sizes": [min(options.hidden_1, 64), min(options.hidden_2, 32)],
        "parameter_count": options.parameter_count,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "runtime_compatible": runtime_compatible,
        "symbol": input_metadata["symbol"],
        "venue": input_metadata["venue"],
        "event_count": input_metadata["event_count"],
        "frame_count": len(dataset.values),
        "supervised_examples": len(supervised),
        "rl_examples": len(rl),
        "holdout_examples": len(holdout),
        "sha256": file_sha256(recording_path),
        "feature_names": input_metadata["feature_names"],
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
        "source_mode": input_metadata["source_mode"],
        "limitations": input_metadata["limitations"],
        "target_price": "candle close" if candle_mode else "book mid",
        "candle_count": input_metadata.get("candle_count", 0),
    }
    yield _event({"kind": "dataset", "dataset": dataset_info})
    model = _make_model(options.hidden_1, options.hidden_2)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    step_number = 0
    for epoch in range(epochs):
        logits, outputs, hidden_1, hidden_2 = _forward(model, inputs)
        loss = -(target_probabilities * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
        sample = epoch % len(supervised)
        example = supervised[sample]
        before = outputs[sample].detach().tolist()
        activations = {
            "hidden_1": hidden_1[sample, :64].detach().tolist(),
            "hidden_2": hidden_2[sample, :32].detach().tolist(),
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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate * 0.2)
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
            "hidden_1": hidden_1[:64].detach().tolist(),
            "hidden_2": hidden_2[:32].detach().tolist(),
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
    _export(supervised_model, mean, scale, paths["supervised"], teaching_only=candle_mode)
    _export(model, mean, scale, paths["adapted"], teaching_only=candle_mode)
    artifacts = {
        name: {
            "path": str(path),
            "sha256": file_sha256(path),
            "schema": artifact_schema,
            "runtime_compatible": runtime_compatible,
        }
        for name, path in paths.items()
        if name != "receipt"
    }
    receipt = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": dataset_info,
        "architecture": {
            "layer_sizes": [300, options.hidden_1, options.hidden_2, 3],
            "activation": "ReLU",
            "parameter_count": options.parameter_count,
            "artifact_schema": artifact_schema,
            "runtime_compatible": runtime_compatible,
        },
        "config": {
            **options.to_dict(),
            "supervised_learning_rate": learning_rate,
            "rl_learning_rate": learning_rate * 0.2,
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
                    "historical.py",
                    "training_options.py",
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
            "Rewards use future candle-close proxy utility, not fills or realized trading P&L."
            if candle_mode
            else "Rewards are future-mid expert proxies; not fills or realized trading P&L.",
            "Offline event-time ordering differs from live Binance receive-time sequencing.",
            "quote_updates counts retained recorded books after sampling, not all live updates.",
            "Held-out windows overlap; this short experiment does not establish generalisation.",
            "Outputs are separate teaching artifacts; the live inference model is not replaced.",
            *input_metadata["limitations"],
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
