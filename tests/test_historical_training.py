"""Integration checks for candle teaching artifacts and chronological training."""

import json

import numpy as np
import pytest

pytest.importorskip("torch")

from market_gate.gate import NumpyMLPGate  # noqa: E402
from market_gate.training_lab import load_lab_dataset, train_lab  # noqa: E402


def candles(path, future_from=None):
    with path.open("w") as handle:
        for i in range(900):
            price = 100 + 0.03 * np.sin(i / 8) + i * 0.001
            if future_from is not None and i >= future_from:
                price += np.sin(i / 3) * 0.02
            handle.write(
                json.dumps(
                    {
                        "kind": "candle",
                        "venue": "binance_spot_candles",
                        "symbol": "ETHUSDT",
                        "interval": "1s",
                        "open_ts_ms": 1788220800000 + i * 1000,
                        "close_ts_ms": 1788220800999 + i * 1000,
                        "open": price,
                        "high": price + 0.01,
                        "low": price - 0.01,
                        "close": price,
                        "volume": 10.0,
                        "taker_buy_volume": float(2 + i % 7),
                        "trades": 12,
                    }
                )
                + "\n"
            )
    return path


def test_candle_training_is_real_but_not_loadable_as_live_book_model(tmp_path):
    source = candles(tmp_path / "candles.jsonl")
    events = list(train_lab(source, tmp_path / "run", epochs=2, max_rl_steps=15))
    dataset = events[0]["dataset"]
    assert dataset["symbol"] == "ETHUSDT"
    assert dataset["source_mode"] == "historical_candles_1s"
    assert dataset["target_price"] == "candle close"
    assert dataset["event_count"] == dataset["candle_count"] == 900
    assert dataset["runtime_compatible"] is False
    assert dataset["limitations"]
    steps = [e["step"] for e in events if e["kind"] == "step"]
    assert len(steps) == 17
    assert any(layer["weight_delta_norm"] > 0 for step in steps for layer in step["layers"])
    for step in steps:
        np.testing.assert_array_equal(np.asarray(step["features"])[:, [3, 4, 5, 8]], 0)
        assert step["target_ts_ms"] - step["input_end_ts_ms"] == 5000
    final = events[-1]
    assert final["architecture"]["artifact_schema"] == "training-mlp-v1"
    assert final["architecture"]["runtime_compatible"] is False
    with pytest.raises(ValueError, match="unsupported gate"):
        NumpyMLPGate(tmp_path / "run/training-mlp-adapted.npz")


def test_candle_holdout_changes_do_not_change_training_inputs_or_checkpoint(tmp_path):
    a, b = candles(tmp_path / "a.jsonl"), candles(tmp_path / "b.jsonl", future_from=720)
    frames_a, _ = load_lab_dataset(a)
    frames_b, _ = load_lab_dataset(b)
    np.testing.assert_array_equal(frames_a.values[:720], frames_b.values[:720])
    first = list(train_lab(a, tmp_path / "a", epochs=1, max_rl_steps=15))[-1]
    second = list(train_lab(b, tmp_path / "b", epochs=1, max_rl_steps=15))[-1]
    assert (
        first["provenance"]["normalization_sha256"] == second["provenance"]["normalization_sha256"]
    )
    for suffix in ["supervised", "adapted"]:
        with np.load(tmp_path / f"a/training-mlp-{suffix}.npz") as one:
            with np.load(tmp_path / f"b/training-mlp-{suffix}.npz") as two:
                for key in one.files:
                    np.testing.assert_array_equal(one[key], two[key])
