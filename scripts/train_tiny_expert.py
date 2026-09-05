#!/usr/bin/env python3
"""Train a shadow direction model from the actual runtime rule scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from market_gate.config import LabConfig  # noqa: E402
from market_gate.engine import MarketEngine  # noqa: E402
from market_gate.experts import EXPERT_IDS  # noqa: E402
from market_gate.tiny_expert import SCHEMA, TinyExpert  # noqa: E402
from market_gate.training import _record_to_event, file_sha256  # noqa: E402


def load_runtime_recording(source: Path):
    """Validate public records while retaining file arrival order for clock ties.

    The generic training loader sorts exchange event time, which is appropriate
    for frame training but loses the receive-clock tie order this runtime uses.
    """
    events = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            events.append(_record_to_event(json.loads(line)))
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"invalid recording line {line_number}: {error}") from error
    if not events:
        raise ValueError("recording is empty")
    if len({(event.venue, event.symbol) for event in events}) != 1:
        raise ValueError("recording must contain one venue and symbol")
    return events


def build_samples(events):
    """Replay the runtime clock and score state; sample one fresh observation/second."""
    engine = MarketEngine(
        LabConfig(source=events[0].venue, symbol=events[0].symbol, gate_mode="uniform"),
        tiny_expert_path=None,
    )
    # Live Binance uses receive time. Reuse that ordering instead of approximating
    # its 64-trade flow and 12-event fair-value histories with frame features.
    ordered = sorted(
        events,
        key=lambda event: event.receive_ts_ms if event.venue == "binance" else event.event_ts_ms,
    )
    sampled = []
    last_second = None
    for event in ordered:
        decision = engine.process(event)
        if decision is None or not engine._book_fresh(event.receive_ts_ms):
            continue
        if event.receive_ts_ms - event.event_ts_ms > engine.config.stale_after_ms:
            continue
        second = decision.timestamp // 1000
        if second == last_second:
            continue
        last_second = second
        inputs = [decision.scores[key] for key in EXPERT_IDS]
        if not np.isfinite(inputs).all():
            continue
        sampled.append((decision.timestamp, inputs, (engine.bid + engine.ask) / 2))
    timestamps = np.array([row[0] for row in sampled], dtype=np.int64)
    rows = []
    for index, (timestamp, inputs, mid) in enumerate(sampled):
        future = int(np.searchsorted(timestamps, timestamp + 5000))
        if future >= len(sampled) or timestamps[future] - timestamp > 6500:
            continue
        # Require continuous observed seconds across the target horizon.
        if np.diff(timestamps[index : future + 1]).max() > 2000:
            continue
        future_mid = sampled[future][2]
        target = np.tanh(10_000 * (future_mid - mid) / mid)
        rows.append((timestamp, int(timestamps[future]), inputs, float(target)))
    if len(rows) < 100:
        raise ValueError("need at least 100 continuous five-second training examples")
    return rows


def split_samples(rows):
    boundary = rows[int(len(rows) * 0.7)][0]
    train = [row for row in rows if row[1] < boundary]
    holdout = [row for row in rows if row[0] >= boundary]
    if len(train) < 50 or len(holdout) < 20:
        raise ValueError("need 50 training and 20 chronological holdout examples")
    assert max(row[1] for row in train) < min(row[0] for row in holdout)
    return train, holdout, boundary


def train(source: Path, output: Path, *, epochs: int = 300, seed: int = 17):
    if type(epochs) is not int or not 1 <= epochs <= 1000:
        raise ValueError("epochs must be an integer in 1..1000")
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer in 0..2147483647")
    receipt_path = output.with_suffix(".json")
    if any(path.exists() or path.is_symlink() for path in (output, receipt_path)):
        raise ValueError("output files must be new")
    if source.stat().st_size > 100_000_000:
        raise ValueError("recording must be at most 100 MB")
    rows = build_samples(load_runtime_recording(source))
    train_rows, holdout_rows, boundary = split_samples(rows)
    import torch
    from torch import nn

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 1), nn.Tanh())
    inputs = torch.tensor([row[2] for row in train_rows], dtype=torch.float32)
    targets = torch.tensor([[row[3]] for row in train_rows], dtype=torch.float32)
    holdout_inputs = torch.tensor([row[2] for row in holdout_rows], dtype=torch.float32)
    holdout_targets = torch.tensor([[row[3]] for row in holdout_rows], dtype=torch.float32)
    initial_parameters = torch.cat([p.detach().flatten().clone() for p in model.parameters()])
    with torch.no_grad():
        initial_loss = float(nn.functional.mse_loss(model(inputs), targets))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
    with torch.no_grad():
        final_loss = float(nn.functional.mse_loss(model(inputs), targets))
        holdout_loss = float(nn.functional.mse_loss(model(holdout_inputs), holdout_targets))
        zero_loss = float((holdout_targets**2).mean())
        delta = float(
            torch.linalg.vector_norm(
                torch.cat([p.detach().flatten() for p in model.parameters()]) - initial_parameters
            )
        )
    arrays = {
        "w1": model[0].weight.detach().numpy().T,
        "b1": model[0].bias.detach().numpy(),
        "w2": model[2].weight.detach().numpy().T,
        "b2": model[2].bias.detach().numpy(),
    }
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("training produced nonfinite weights")
    weight_hash = hashlib.sha256(b"".join(value.tobytes() for value in arrays.values())).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        np.savez(
            handle,
            schema_version=np.array(SCHEMA),
            input_order=np.array(EXPERT_IDS),
            model_version=np.array(f"tiny-expert-{weight_hash[:12]}"),
            **arrays,
        )
    runtime = TinyExpert(output)
    raw_holdout = holdout_inputs.numpy().tolist()
    with torch.no_grad():
        expected = model(holdout_inputs).numpy().ravel()
    observed = np.array([runtime.predict(row) for row in raw_holdout])
    np.testing.assert_allclose(observed, expected, atol=1e-6, rtol=1e-5)
    for _ in range(200):
        runtime.predict(raw_holdout[0])
    latencies = []
    for index in range(5000):
        started = time.perf_counter_ns()
        runtime.predict(raw_holdout[index % len(raw_holdout)])
        latencies.append((time.perf_counter_ns() - started) / 1000)
    module_root = Path(__file__).resolve().parents[1] / "src/market_gate"
    receipt = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "model_version": runtime.model_version,
        "model_sha256": file_sha256(output),
        "dataset_sha256": file_sha256(source),
        "source_sha256": {
            **{
                name: file_sha256(module_root / name)
                for name in (
                    "tiny_expert.py",
                    "engine.py",
                    "experts.py",
                    "config.py",
                    "training.py",
                    "features.py",
                    "contracts.py",
                    "gate.py",
                    "paper.py",
                    "risk.py",
                )
            },
            "scripts/train_tiny_expert.py": file_sha256(Path(__file__)),
        },
        "architecture": {"layer_sizes": [3, 8, 1], "activations": ["tanh", "tanh"]},
        "parameter_count": runtime.parameter_count,
        "input_order": list(EXPERT_IDS),
        "input_source": "MarketEngine.process current bounded rule scores, sampled once per second",
        "score_config": {"flow_window_trades": 64, "fair_history_events": 12},
        "clock": (
            "Binance receive_ts_ms with recording arrival order for ties; "
            "same rule state as runtime"
        ),
        "target": "tanh(mid return in bps to first fresh sample at least 5 seconds later)",
        "target_horizon_ms": [5000, 6500],
        "split": {
            "method": "chronological 70/30; training target strictly before holdout start",
            "boundary_ts_ms": boundary,
            "train_examples": len(train_rows),
            "holdout_examples": len(holdout_rows),
            "train_last_target_ts_ms": max(row[1] for row in train_rows),
            "holdout_first_input_ts_ms": min(row[0] for row in holdout_rows),
        },
        "training": {
            "epochs": epochs,
            "seed": seed,
            "learning_rate": 0.01,
            "optimizer": "Adam",
            "initial_mse": initial_loss,
            "final_mse": final_loss,
            "parameter_delta_l2": delta,
        },
        "holdout": {
            "model_mse": holdout_loss,
            "zero_prediction_mse": zero_loss,
            "target_standard_deviation": float(holdout_targets.std(unbiased=False)),
        },
        "runtime_parity_max_absolute_error": float(np.max(np.abs(observed - expected))),
        "benchmark": {
            "scope": "local warm NumPy predict including input validation; no engine or networking",
            "samples": len(latencies),
            "warmup_calls": 200,
            "median_us": float(np.median(latencies)),
            "p95_us": float(np.percentile(latencies, 95)),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "limitations": [
            "Shadow only; output never changes the 3-expert gate, contributions or quotes.",
            "Short public capture and overlapping labels do not establish generalization.",
            "Recorded books were sampled; replay cannot reconstruct all live quote updates.",
            "Predicted mid movement is not fills, costs or realized trading P&L.",
            "Local warm timings are not an HFT latency guarantee.",
        ],
    }
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    receipt = train(args.input, args.output, epochs=args.epochs, seed=args.seed)
    print(json.dumps({key: receipt[key] for key in ("training", "holdout", "benchmark")}))


if __name__ == "__main__":
    main()
