import json
from pathlib import Path

import numpy as np
import pytest

from market_gate.contracts import BookEvent, TradeEvent
from market_gate.training import (
    FEATURE_NAMES,
    build_examples,
    build_frames,
    chronological_split,
    event_to_record,
    load_recording,
    should_keep_book,
    train_recording,
    write_recording,
)


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


def test_recording_is_normalized_and_book_updates_are_downsampled(tmp_path: Path) -> None:
    path = tmp_path / "sample.jsonl"
    counts = write_recording(path, [book(0, 1), book(300, 2), trade(400, 1), book(1_000, 3)], 1_000)

    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert counts == {"book": 2, "trade": 1, "total": 3}
    assert [line["kind"] for line in lines] == ["book", "trade", "book"]
    assert "data" not in lines[0]
    assert [type(event) for event in load_recording(path)] == [BookEvent, TradeEvent, BookEvent]
    assert should_keep_book(1_000, 0, 1_000)
    assert not should_keep_book(999, 0, 1_000)
    assert event_to_record(trade(1_200, 2))["aggressor"] == "buy"


def test_windows_and_chronological_split_are_causal_without_overlap() -> None:
    frames, mids = build_frames(recording_events())
    examples = build_examples(frames, mids, lookback_frames=30, horizon_frames=5)
    train, validation = chronological_split(examples, len(frames))

    assert frames.shape[1] == len(FEATURE_NAMES)
    assert all(example.target_frame >= example.start_frame + 30 for example in examples)
    assert max(example.target_frame for example in train) < min(
        example.start_frame for example in validation
    )
    assert np.all(np.isfinite(train[0].features))
    assert examples[0].utilities.shape == (3,)


def test_utility_labels_follow_runtime_expert_order() -> None:
    frames = np.zeros((3, len(FEATURE_NAMES)), dtype=np.float32)
    frames[0, 4:7] = [0.5, 0.0001, 0.75]
    frames[0, 7] = 4
    frames[0, 9] = 0.0002
    mids = np.asarray([100.0, 100.1, 100.2], dtype=np.float32)

    example = build_examples(frames, mids, lookback_frames=1, horizon_frames=1)[0]

    assert example.utilities[0] > 0  # microprice pressure agreed with the later rise
    assert example.utilities[1] > 0  # buy flow agreed with the later rise
    assert example.utilities[2] < 0  # reversion pointed against the later rise


def test_load_recording_rejects_invalid_public_event(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    record = event_to_record(trade(0, 0))
    path.write_text(json.dumps(record | {"aggressor": "sideways"}))
    with pytest.raises(ValueError, match="invalid recording line 1"):
        load_recording(path)


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
        "evaluation",
        "provenance",
    }
    assert receipt["dataset"]["feature_names"] == FEATURE_NAMES
    assert receipt["dataset"]["train_examples"] > 0
    assert receipt["dataset"]["validation_examples"] > 0
    assert (tmp_path / "gate-binance-demo.npz").is_file()
    assert (tmp_path / "training-demo.json").is_file()
    assert str(tmp_path) not in json.dumps(receipt)
