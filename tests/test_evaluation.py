import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_training import book, recording_events, trade

from market_gate.evaluation import compare_utilities, evaluate_recording
from market_gate.training import (
    TrainingExample,
    build_examples,
    build_frame_dataset,
    chronological_split,
    event_to_record,
    load_recording,
    train_recording,
    write_recording,
)


@pytest.mark.parametrize(
    "change",
    [
        {"bid_price": float("nan")},
        {"bid_price": float("inf")},
        {"bid_price": 0},
        {"bid_size": -1},
        {"ask_price": 99},
        {"bid_price": "100"},
        {"bid_price": True},
        {"event_ts_ms": 1.5},
        {"update_id": True},
        {"receive_ts_ms": -1},
        {"symbol": 12},
    ],
)
def test_recording_rejects_invalid_numeric_contract(tmp_path, change):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(event_to_record(book(0, 1)) | change))
    with pytest.raises(ValueError, match="invalid recording line 1"):
        load_recording(path)


def test_recording_rejects_mixed_symbols(tmp_path):
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(event_to_record(e))
            for e in [book(0, 1), replace(trade(1, 2), symbol="ETHUSDT")]
        )
    )
    with pytest.raises(ValueError, match="one venue and symbol"):
        load_recording(path)


def test_gap_resets_history_and_excludes_input_or_target_crossing():
    events = [
        book(0, 0),
        book(1000, 1),
        replace(book(4000, 2), bid_price=199, ask_price=201),
        book(5000, 3),
        book(6000, 4),
        book(7000, 5),
    ]
    data = build_frame_dataset(events)
    assert data.close_ts_ms.tolist() == [999, 1999, 4999, 5999, 6999]
    assert data.values[2, [0, 1, 9]].tolist() == [0, 0, 0]
    examples = build_examples(data.values, data.mids, 1, 1, close_ts_ms=data.close_ts_ms)
    assert [e.start_frame for e in examples] == [0, 2, 3]
    assert all(e.target_ts_ms - e.input_end_ts_ms == 1000 for e in examples)


def test_delayed_trade_uses_documented_event_clock(tmp_path):
    path = tmp_path / "delayed.jsonl"
    events = [
        book(1000, 0),
        book(2000, 1),
        replace(trade(1500, 2), receive_ts_ms=3500),
        book(4000, 3),
    ]
    path.write_text("\n".join(json.dumps(event_to_record(e)) for e in events))
    data = build_frame_dataset(load_recording(path))
    assert data.close_ts_ms[0] == 1999
    assert data.values[0, 7] == 1


def test_filtering_cannot_move_validation_boundary_earlier():
    data = build_frame_dataset(recording_events())
    examples = build_examples(data.values, data.mids, 30, 5, close_ts_ms=data.close_ts_ms)
    train, validation = chronological_split(examples, len(data.values))
    filtered = [e for e in examples if e.start_frame <= 65]
    with pytest.raises(ValueError, match="record a longer sample"):
        chronological_split(filtered, len(data.values))
    assert max(e.target_ts_ms for e in train) < min(e.start_ts_ms for e in validation)


def test_every_baseline_uses_identical_examples():
    examples = [
        TrainingExample(i, i + 1, np.zeros(300), np.asarray(y))
        for i, y in enumerate([[1, 2, 3], [-4, 5, -6]])
    ]
    weights = np.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]])
    result = compare_utilities(examples, weights)
    means = result["mean_proxy_utility"]
    y = np.stack([e.utilities for e in examples])
    assert means == pytest.approx(
        {
            "neural": np.mean(np.sum(y * weights, axis=1)),
            "uniform": np.mean(y),
            "static": np.mean(y @ [0.5, 0.35, 0.15]),
            "microprice": -1.5,
            "flow": 3.5,
            "reversion": -1.5,
        }
    )


def test_export_requires_runtime_shape_before_writing(tmp_path):
    with pytest.raises(ValueError, match="lookback_frames=30"):
        train_recording(
            tmp_path / "absent", tmp_path / "model.npz", tmp_path / "r.json", lookback_frames=2
        )
    assert not list(tmp_path.iterdir())


def test_exported_evaluator_matches_training_and_provenance(tmp_path: Path):
    pytest.importorskip("torch")
    recording, model, receipt = (tmp_path / name for name in ["r.jsonl", "m.npz", "r.json"])
    events = [
        replace(e, bid_price=e.bid_price + i * 0.001, ask_price=e.ask_price + i * 0.001)
        if hasattr(e, "bid_price")
        else e
        for i, e in enumerate(recording_events())
    ]
    write_recording(recording, events, 1)
    result = train_recording(recording, model, receipt, epochs=2)
    evaluation = evaluate_recording(recording, model)
    assert result["schema_version"] == 2
    assert result["evaluation"]["mean_proxy_utility"] == pytest.approx(
        evaluation["evaluation"]["mean_proxy_utility"], abs=1e-6
    )
    assert result["training"]["validation_loss"] == pytest.approx(
        -result["evaluation"]["mean_proxy_utility"]["neural"], abs=1e-6
    )
    assert result["provenance"]["model_sha256"] == evaluation["provenance"]["model_sha256"]
    assert (
        result["dataset"]["train_last_target_ts_ms"]
        < result["dataset"]["validation_first_start_ts_ms"]
    )
    assert len(result["provenance"]["source_sha256"]) == 64


@pytest.mark.parametrize("prefix", [[], [book(0, 0), book(1000, 1)]])
def test_trade_only_seconds_never_borrow_a_future_book(prefix):
    data = build_frame_dataset(
        prefix
        + [
            trade(4000, 2),
            trade(5000, 3),
            replace(book(5500, 4), bid_price=199, ask_price=201),
            trade(5700, 5),
            book(6000, 6),
            book(7000, 7),
        ]
    )
    assert 4999 not in data.close_ts_ms
    index = list(data.close_ts_ms).index(5999)
    assert data.mids[index] == 200
    assert data.values[index, 7] == 1  # Only the trade after the fresh book.
    assert data.values[index, 0] == 0  # No pre-gap return history.


@pytest.mark.parametrize("existing", ["model.npz", "receipt.json"])
def test_training_refuses_existing_output_before_reading_or_training(tmp_path, existing):
    protected = tmp_path / existing
    protected.write_bytes(b"preserve exact evidence")
    with pytest.raises(ValueError, match="outputs must be new"):
        train_recording(
            tmp_path / "absent.jsonl", tmp_path / "model.npz", tmp_path / "receipt.json"
        )
    assert protected.read_bytes() == b"preserve exact evidence"
    assert len(list(tmp_path.iterdir())) == 1
