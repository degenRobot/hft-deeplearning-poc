import json
from pathlib import Path

import pytest

from market_gate.contracts import BookEvent, TradeEvent
from market_gate.training import FEATURE_NAMES, train_recording, write_recording


def book(timestamp_ms: int, update_id: int) -> BookEvent:
    return BookEvent(
        "binance", "BTCUSDT", timestamp_ms, timestamp_ms + 1, update_id, 100.0, 2.0, 101.0, 1.0
    )


def trade(timestamp_ms: int, trade_id: int) -> TradeEvent:
    return TradeEvent(
        "binance", "BTCUSDT", timestamp_ms, timestamp_ms + 1, trade_id, 100.5, 0.2, "buy"
    )


def recording_events(seconds: int = 120) -> list[BookEvent | TradeEvent]:
    events: list[BookEvent | TradeEvent] = []
    for second in range(seconds):
        timestamp_ms = second * 1_000
        events.extend([book(timestamp_ms, second), trade(timestamp_ms + 200, second)])
    return events


def test_training_writes_the_complete_receipt_contract(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    recording_path = tmp_path / "sample.jsonl"
    write_recording(recording_path, recording_events(), book_interval_ms=1)

    receipt = train_recording(
        recording_path,
        tmp_path / "gate-binance-demo.npz",
        tmp_path / "training-demo.json",
        epochs=2,
        lookback_frames=30,
        horizon_frames=5,
    )

    assert set(receipt) == {
        "schema_version",
        "generated_at",
        "source",
        "dataset",
        "training",
        "limitations",
    }
    assert receipt["dataset"]["feature_names"] == FEATURE_NAMES
    assert receipt["dataset"]["train_examples"] > 0
    assert receipt["dataset"]["validation_examples"] > 0
    assert (tmp_path / "gate-binance-demo.npz").is_file()
    assert (tmp_path / "training-demo.json").is_file()
    assert str(tmp_path) not in json.dumps(receipt)
