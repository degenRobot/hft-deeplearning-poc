import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from market_gate.config import LabConfig
from market_gate.contracts import BookEvent, TradeEvent
from market_gate.engine import MarketEngine
from market_gate.experts import EXPERT_IDS
from market_gate.tiny_expert import DEFAULT_TINY_EXPERT_PATH, TinyExpert, load_tiny_expert
from market_gate.training import file_sha256, load_recording, write_recording


def book(timestamp, mid=100.0):
    return BookEvent(
        "replay", "BTCUSDT", timestamp, timestamp, timestamp, mid - 0.01, 2, mid + 0.01, 1
    )


def synthetic_events(changed_from=None):
    rows = []
    for second in range(180):
        mid = 100 + 0.02 * np.sin(second / 7)
        if changed_from is not None and second >= changed_from:
            mid += 0.1 * np.cos(second / 3)
        rows.append(book(1000 + second * 1000, mid))
        rows.append(
            TradeEvent(
                "replay",
                "BTCUSDT",
                1200 + second * 1000,
                1200 + second * 1000,
                second,
                mid,
                0.1,
                "buy" if second % 3 else "sell",
            )
        )
    return rows


def training_module():
    path = Path(__file__).resolve().parents[1] / "scripts/train_tiny_expert.py"
    spec = importlib.util.spec_from_file_location("train_tiny_expert_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundled_expert_is_trained_finite_and_has_receipt():
    model = TinyExpert(DEFAULT_TINY_EXPERT_PATH)
    receipt = json.loads(DEFAULT_TINY_EXPERT_PATH.with_suffix(".json").read_text())
    assert receipt["model_sha256"] == file_sha256(DEFAULT_TINY_EXPERT_PATH)
    assert receipt["parameter_count"] == model.parameter_count == 41
    assert receipt["input_order"] == list(EXPERT_IDS)
    assert receipt["training"]["parameter_delta_l2"] > 0
    assert receipt["training"]["final_mse"] < receipt["training"]["initial_mse"]
    assert (
        receipt["split"]["train_last_target_ts_ms"] < receipt["split"]["holdout_first_input_ts_ms"]
    )
    predictions = [model.predict(row) for row in ([0, 0, 0], [-1, 1, -1], [1, -1, 1])]
    assert all(np.isfinite(value) and -1 <= value <= 1 for value in predictions)
    assert len(set(predictions)) > 1


@pytest.mark.parametrize("inputs", [[0, 0], [0, 0, 1.1], [0, np.nan, 0], [np.inf, 0, 0]])
def test_expert_rejects_invalid_inputs(inputs):
    with pytest.raises(ValueError):
        TinyExpert(DEFAULT_TINY_EXPERT_PATH).predict(inputs)


def test_missing_corrupt_and_wrong_shape_artifacts_fail_closed(tmp_path):
    assert load_tiny_expert(tmp_path / "missing.npz") is None
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"not a model")
    assert load_tiny_expert(bad) is None
    with np.load(DEFAULT_TINY_EXPERT_PATH) as model:
        arrays = {name: model[name] for name in model.files}
    for changed in (
        {"w1": np.zeros((4, 8))},
        {"b1": np.full(8, np.nan)},
        {"input_order": np.array(["flow", "microprice", "reversion"])},
        {"schema_version": np.array("unknown")},
    ):
        np.savez(bad, **(arrays | changed))
        assert load_tiny_expert(bad) is None


def test_shadow_keeps_decisions_identical_and_matches_current_rule_scores():
    enabled = MarketEngine(LabConfig(gate_mode="uniform"))
    disabled = MarketEngine(LabConfig(gate_mode="uniform"), tiny_expert_path=None)
    enabled.feed_status = disabled.feed_status = "running"
    for event in synthetic_events()[:80]:
        actual, baseline = enabled.process(event), disabled.process(event)
        assert actual == baseline
        assert enabled.paper.inventory == disabled.paper.inventory
        assert enabled.paper.pnl == disabled.paper.pnl
        assert enabled.weights == disabled.weights
        snapshot = enabled.snapshot(event.receive_ts_ms)
        neural = snapshot["visual"]["neural_expert"]
        assert neural["inputs"] == [actual.scores[key] for key in EXPERT_IDS]
        assert neural["score"] == enabled.tiny_expert.predict(neural["inputs"])
        assert neural["parameter_count"] == 41
        assert neural["inference_us"] >= 0
        assert len(snapshot["experts"]) == len(snapshot["visual"]["events"][-1]["scores"]) == 3
        assert snapshot["visual"]["events"][-1]["neural_score"] == neural["score"]
        assert disabled.snapshot(event.receive_ts_ms)["visual"]["neural_expert"] is None


def test_shadow_loads_once_and_stale_books_do_not_run_inference(monkeypatch):
    engine = MarketEngine(LabConfig(stale_after_ms=100, gate_mode="uniform"))
    engine.feed_status = "running"
    engine.process(book(1000))
    assert engine.snapshot(1000)["visual"]["neural_expert"] is not None
    calls = []
    original = engine.tiny_expert.predict

    def observed(inputs):
        calls.append(inputs)
        return original(inputs)

    monkeypatch.setattr(engine.tiny_expert, "predict", observed)
    monkeypatch.setattr("market_gate.engine.load_tiny_expert", lambda *_: pytest.fail("reloaded"))
    stale_trade = TradeEvent("replay", "BTCUSDT", 1200, 1200, 2, 100, 0.1, "buy")
    engine.process(stale_trade)
    assert calls == []
    assert engine.neural_expert is None
    assert engine.snapshot(1200)["visual"]["neural_expert"] is None
    engine.process(book(1201))
    assert len(calls) == 1
    assert engine.snapshot(1302)["visual"]["neural_expert"] is None
    engine.process(replace(book(1202), venue="binance", receive_ts_ms=1400))
    assert len(calls) == 1
    assert engine.neural_expert is None


def test_training_inputs_are_exact_engine_scores_and_split_has_no_future_overlap():
    module = training_module()
    events = synthetic_events()
    rows = module.build_samples(events)
    engine = MarketEngine(LabConfig(gate_mode="uniform"), tiny_expert_path=None)
    expected = {}
    for event in events:
        decision = engine.process(event)
        expected[decision.timestamp] = [decision.scores[key] for key in EXPERT_IDS]
    for timestamp, target_timestamp, inputs, target in rows:
        assert inputs == expected[timestamp]
        assert 5000 <= target_timestamp - timestamp <= 6500
        assert -1 <= target <= 1
    train, holdout, _ = module.split_samples(rows)
    assert max(row[1] for row in train) < min(row[0] for row in holdout)


def test_binance_receive_clock_ties_preserve_actual_arrival_state(tmp_path):
    module = training_module()
    events = []
    for event in synthetic_events():
        receive_time = (event.receive_ts_ms // 1000) * 1000
        # Arrival is book then trade within the same millisecond, although the
        # trade's exchange timestamp precedes the book's receive timestamp.
        events.append(
            replace(
                event,
                venue="binance",
                receive_ts_ms=receive_time,
                event_ts_ms=receive_time - (10 if isinstance(event, TradeEvent) else 0),
            )
        )
    source = tmp_path / "tied-arrivals.jsonl"
    write_recording(source, events, 1)
    restored = module.load_runtime_recording(source)
    assert restored == events
    expected = {}
    engine = MarketEngine(LabConfig(source="binance", gate_mode="uniform"), tiny_expert_path=None)
    for event in restored:
        decision = engine.process(event)
        expected.setdefault(decision.timestamp, [decision.scores[key] for key in EXPERT_IDS])
    rows = module.build_samples(restored)
    assert all(inputs == expected[timestamp] for timestamp, _, inputs, _ in rows)
    # Keep the generic event-time loader unchanged and prove that these ties
    # would have produced different training inputs through that old path.
    assert any(
        inputs != expected[timestamp]
        for timestamp, _, inputs, _ in module.build_samples(load_recording(source))
    )


@pytest.mark.parametrize("content", ["", "{}\n", '{"kind":"unknown"}\n', "null\n"])
def test_runtime_recording_still_rejects_invalid_public_records(tmp_path, content):
    source = tmp_path / "invalid.jsonl"
    source.write_text(content)
    with pytest.raises(ValueError):
        training_module().load_runtime_recording(source)


def test_holdout_prices_cannot_change_trained_weights(tmp_path):
    pytest.importorskip("torch")
    module = training_module()
    original = tmp_path / "original.jsonl"
    changed = tmp_path / "changed.jsonl"
    write_recording(original, synthetic_events(), 1)
    write_recording(changed, synthetic_events(changed_from=130), 1)
    first = module.train(original, tmp_path / "first.npz", epochs=5)
    second = module.train(changed, tmp_path / "second.npz", epochs=5)
    assert first["training"] == second["training"]
    assert first["model_version"] == second["model_version"]
    assert first["holdout"] != second["holdout"]
