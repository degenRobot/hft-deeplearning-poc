"""Real output-head updates remain causal, bounded and resettable."""

from collections import deque
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from market_gate.engine import MarketEngine
from market_gate.experts import EXPERT_IDS
from market_gate.gate import load_numpy_gate
from market_gate.live_learning import LiveLearner

LAYERS = ("w1", "b1", "w2", "b2", "w3", "b3")


@pytest.fixture(autouse=True)
def learning_clock(monkeypatch):
    monkeypatch.setattr("market_gate.live_learning.time.time", lambda: 70.0)


def engine_at(now=100_000):
    gate = load_numpy_gate(Path(__file__).parents[1] / "models/gate-demo.npz")
    engine = SimpleNamespace(
        gate=gate,
        config=SimpleNamespace(
            gate_mode="neural",
            gate_interval_ms=100,
            stale_after_ms=1500,
            source="binance",
            symbol="BTCUSDT",
        ),
        feed_status="running",
        book_valid=True,
        decision=SimpleNamespace(scores=dict.fromkeys(EXPERT_IDS, 1.0)),
        last_book_event_ts_ms=now,
        last_receive_ts_ms=now,
        last_book_receive_ts_ms=now,
        frame_times=deque(range(now - 29_000, now + 1, 1000), maxlen=30),
        frames=deque(np.random.default_rng(21).normal(0, 0.1, (30, 10)), maxlen=30),
        bid=99.99,
        ask=100.01,
        last_gate_ts_ms=now,
        run_id="test-run",
    )
    engine._book_fresh = MethodType(MarketEngine._book_fresh, engine)
    engine._book_age = MethodType(MarketEngine._book_age, engine)
    return engine


def advance(engine, now, mid=100.02):
    engine.frame_times = deque(range(now - 29_000, now + 1, 1000), maxlen=30)
    engine.last_receive_ts_ms = engine.last_book_receive_ts_ms = engine.last_book_event_ts_ms = now
    engine.bid, engine.ask = mid - 0.01, mid + 0.01


def start(now=100_000):
    engine = engine_at(now)
    learner = LiveLearner(engine)
    learner.configure({"enabled": True})
    learner.tick(now)
    assert learner.pending is not None
    return engine, learner


def weights(engine):
    return {name: getattr(engine.gate, name).copy() for name in LAYERS}


def assert_weights_equal(engine, expected):
    for name, value in expected.items():
        np.testing.assert_array_equal(getattr(engine.gate, name), value)


def test_actual_update_changes_only_head_and_reports_exact_probabilities():
    engine, learner = start()
    original = weights(engine)
    sample = learner.pending
    features = sample["features"].copy()
    before = engine.gate.predict(features)
    np.testing.assert_allclose(sample["before"], list(before.values()), rtol=1e-6)
    advance(engine, 105_000)
    learner.tick(105_000)

    assert learner.updates == 1
    assert learner.pending is None
    assert learner.history[-1]["reward"] > 0
    assert_weights_equal(engine, {key: original[key] for key in LAYERS[:4]})
    assert not np.array_equal(engine.gate.w3, original["w3"])
    assert not np.array_equal(engine.gate.b3, original["b3"])
    assert np.isfinite(engine.gate.w3).all() and np.isfinite(engine.gate.b3).all()
    after = engine.gate.predict(features)
    receipt = learner.history[-1]
    np.testing.assert_allclose(receipt["outputs_before"], list(before.values()), rtol=1e-6)
    np.testing.assert_allclose(receipt["outputs_after"], list(after.values()), rtol=1e-6)
    assert receipt["outputs_after"][sample["action"]] > receipt["outputs_before"][sample["action"]]
    assert sum(receipt["outputs_before"]) == pytest.approx(1)
    assert sum(receipt["outputs_after"]) == pytest.approx(1)
    assert receipt["input_end_ts_ms"] <= receipt["selected_ts_ms"]
    assert receipt["target_ts_ms"] >= receipt["selected_ts_ms"] + 5000
    assert receipt["layers"][0]["weight_delta_norm"] == 0
    assert receipt["layers"][1]["gradient_norm"] == 0
    assert receipt["layers"][2]["weight_delta_norm"] > 0
    assert engine.gate.model_version.endswith("-live-rl-1")
    assert engine.last_gate_ts_ms == -engine.config.gate_interval_ms


def test_one_pending_sample_needs_full_reward_horizon_and_a_new_book():
    engine, learner = start()
    original = weights(engine)
    sample = learner.pending
    for now in (101_000, 104_000, 104_999):
        advance(engine, now)
        learner.tick(now)
        assert learner.pending is sample
        assert learner.updates == 0
        assert_weights_equal(engine, original)
    advance(engine, 105_000)
    engine.last_book_receive_ts_ms = 104_999
    learner.tick(105_000)
    assert learner.pending is sample
    assert learner.updates == 0
    engine.last_book_receive_ts_ms = 105_000
    learner.tick(105_000)
    assert learner.updates == 1
    assert learner.pending is None
    updated = weights(engine)
    # The already consumed on-policy sample cannot train twice.
    learner.tick(105_000)
    assert learner.pending is None
    assert learner.updates == 1
    assert_weights_equal(engine, updated)
    advance(engine, 110_000)
    learner.tick(110_000)
    assert learner.pending is not None and learner.pending is not sample
    assert learner.pending["selected_ms"] == 110_000


@pytest.mark.parametrize("failure", ["disconnect", "stale", "invalid_book", "gap"])
def test_feed_break_discards_sample_and_preserves_weights(failure):
    engine, learner = start()
    original = weights(engine)
    advance(engine, 103_000)
    if failure == "disconnect":
        engine.feed_status = "reconnecting"
    elif failure == "stale":
        engine.last_book_receive_ts_ms = 100_000
    elif failure == "invalid_book":
        engine.book_valid = False
    else:
        engine.frame_times[15] += 1
    learner.tick(103_000)
    assert learner.pending is None
    assert learner.discarded == 1
    assert learner.updates == 0
    assert_weights_equal(engine, original)
    engine.feed_status, engine.book_valid = "running", True
    advance(engine, 105_000)
    learner.tick(105_000)
    assert learner.updates == 0
    assert_weights_equal(engine, original)
    if failure != "gap":
        assert learner.pending is None  # Thirty new contiguous seconds after a feed break.
        advance(engine, 133_000)
        learner.tick(133_000)
        assert learner.pending is not None


def test_pause_drops_pending_without_updating_and_reset_restores_original_head():
    engine, learner = start()
    original = weights(engine)
    version = engine.gate.model_version
    learner.configure({"enabled": False})
    assert learner.pending is None
    advance(engine, 105_000)
    learner.tick(105_000)
    assert learner.updates == 0
    assert_weights_equal(engine, original)
    learner.configure({"enabled": True})
    learner.tick(105_000)
    advance(engine, 110_000, mid=100.04)
    learner.tick(110_000)
    assert learner.updates == 1
    assert not np.array_equal(engine.gate.w3, original["w3"])
    learner.configure({"reset": True})
    assert_weights_equal(engine, original)
    assert engine.gate.model_version == version
    assert learner.updates == 0
    assert learner.pending is None
    assert not learner.enabled
    assert not learner.history
    assert learner.baseline == 0


@pytest.mark.parametrize(
    "invalid",
    [
        {"enabled": "true"},
        {"reset": 1},
        {"interval_seconds": True},
        {"interval_seconds": 4},
        {"interval_seconds": 61},
        {"interval_seconds": 5.0},
        {"enabled": False, "unknown": 1},
        {"reset": True, "interval_seconds": 0},
    ],
)
def test_invalid_settings_are_atomic(invalid):
    engine, learner = start()
    original = weights(engine)
    sample = learner.pending
    prior = (learner.enabled, learner.interval_seconds, learner.stage, learner.next_sample_ms)
    with pytest.raises(ValueError):
        learner.configure(invalid)
    assert learner.pending is sample
    assert (
        learner.enabled,
        learner.interval_seconds,
        learner.stage,
        learner.next_sample_ms,
    ) == prior
    assert_weights_equal(engine, original)


def test_non_neural_gate_cannot_enable_learning():
    engine = engine_at()
    engine.config.gate_mode = "static"
    learner = LiveLearner(engine)
    with pytest.raises(ValueError, match="neural"):
        learner.configure({"enabled": True, "reset": True})
    assert not learner.enabled
    assert learner.pending is None


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_features_are_rejected_before_sampling(invalid):
    engine = engine_at()
    engine.frames[0][0] = invalid
    original = weights(engine)
    learner = LiveLearner(engine)
    learner.configure({"enabled": True})
    learner.tick(100_000)
    assert learner.pending is None
    assert learner.updates == 0
    assert_weights_equal(engine, original)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_reward_inputs_do_not_mutate_head(invalid):
    engine, learner = start()
    original = weights(engine)
    advance(engine, 105_000)
    engine.bid = invalid
    learner.tick(105_000)
    assert learner.updates == 0
    assert learner.pending is None
    assert_weights_equal(engine, original)


def test_late_outcome_discards_sample_instead_of_relabeling_it():
    engine, learner = start()
    original = weights(engine)
    advance(engine, 107_000)
    learner.tick(107_000)
    assert learner.updates == 0
    assert learner.pending is None
    assert learner.discarded == 1
    assert_weights_equal(engine, original)


def test_reward_keeps_sampled_score_and_features_when_market_state_changes():
    engine, learner = start()
    sampled = learner.pending["features"].copy()
    engine.decision.scores = dict.fromkeys(EXPERT_IDS, -1.0)
    for frame in engine.frames:
        frame[:] = -10
    advance(engine, 105_000)
    learner.tick(105_000)
    assert learner.updates == 1
    assert learner.history[-1]["reward"] > 0  # Score was +1 when the action was sampled.
    np.testing.assert_array_equal(learner.history[-1]["features"], sampled)
    expected = list(engine.gate.predict(sampled).values())
    np.testing.assert_allclose(learner.history[-1]["outputs_after"], expected, rtol=1e-6)


def test_enable_requires_new_warmup_and_rejects_future_frame(monkeypatch):
    engine = engine_at()
    monkeypatch.setattr("market_gate.live_learning.time.time", lambda: 100.0)
    learner = LiveLearner(engine)
    learner.configure({"enabled": True})
    learner.tick(100_000)
    assert learner.pending is None
    assert learner.stage == "warming_up"
    advance(engine, 130_000)
    engine.frame_times[-1] = 131_000
    learner.tick(130_000)
    assert learner.pending is None
    advance(engine, 130_000)
    learner.tick(130_000)
    assert learner.pending is not None


@pytest.mark.parametrize("field", ["score", "mid"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_invalid_sample_numerics_never_update(field, invalid):
    engine, learner = start()
    original = weights(engine)
    learner.pending[field] = invalid
    advance(engine, 105_000)
    learner.tick(105_000)
    assert learner.updates == 0
    assert learner.pending is None
    assert_weights_equal(engine, original)


def test_repeated_sampling_normalizes_in_double_precision():
    engine, learner = start()
    for i in range(25):
        selected = 100_000 + i * 10_000
        advance(engine, selected + 5000, mid=100 + (i + 1) * 0.02)
        learner.tick(selected + 5000)
        assert learner.updates == i + 1
        advance(engine, selected + 10_000, mid=100 + (i + 1) * 0.02)
        learner.tick(selected + 10_000)
        assert learner.pending is not None
    assert learner.enabled


def test_delayed_book_event_cannot_teach_while_terminal_considers_it_stale():
    engine, learner = start()
    original = weights(engine)
    advance(engine, 105_000)
    engine.last_book_event_ts_ms = 100_000
    assert not engine._book_fresh(105_000)
    learner.tick(105_000)
    assert learner.updates == 0
    assert learner.pending is None
    assert learner.stage == "waiting_for_feed"
    assert_weights_equal(engine, original)
