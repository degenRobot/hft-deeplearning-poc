from market_gate.config import LabConfig
from market_gate.contracts import BookEvent, TradeEvent
from market_gate.engine import MarketEngine


def book(ts, identifier=1):
    return BookEvent("replay", "BTCUSDT", ts, ts, identifier, 99.0, 2.0, 101.0, 1.0)


def engine():
    value = MarketEngine(LabConfig(gate_interval_ms=2000), "models/gate-demo.npz")
    value.feed_status = "running"
    return value


def test_visual_tape_matches_decisions_and_is_bounded_and_hidden_when_stale():
    value = engine()
    for i in range(60):
        value.process(book(1000 + i * 10, i))
    value.process(TradeEvent("replay", "BTCUSDT", 1600, 1600, 99, 101.0, 0.3, "buy"))
    snap = value.snapshot(1600)
    tape = snap["visual"]["events"]
    assert len(tape) == 48
    assert tape[-1]["kind"] == "buy"
    assert tape[-1]["size"] == 0.3
    assert tape[-1]["signal"] == sum(e["contribution"] for e in snap["experts"])
    assert tape[-1]["scores"] == [e["score"] for e in snap["experts"]]
    assert [e["id"] for e in tape] == list(range(14, 62))
    assert value.snapshot(10000)["visual"]["events"] == []
    assert value.snapshot(10000)["visual"]["window"] == []


def test_visual_window_is_the_causal_window_used_at_the_last_gate_refresh():
    value = engine()
    value.process(book(1000))
    value.process(book(2000))  # Closes one frame but does not refresh the 2s gate.
    assert len(value.frames) == 1
    assert value.snapshot(2000)["visual"]["window"] == []
    value.process(book(3000))
    snap = value.snapshot(3000)
    visual = snap["visual"]
    assert [f["timestamp_ms"] for f in visual["window"]] == [1999, 2999]
    assert [f["values"] for f in visual["window"]] == [list(f) for f in value.frames]
    assert visual["proposed"] == value.gate.predict([(0.0,) * 10] * 28 + list(value.frames))
    value.process(book(4000))
    assert value.snapshot(4000)["visual"]["window"] == visual["window"]
    assert len(value.frames) == 3
    assert engine().snapshot(4000)["visual"]["events"] == []
