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
    for i in range(80):
        value.process(book(1000 + i * 10, i))
    value.process(TradeEvent("replay", "BTCUSDT", 1800, 1800, 99, 101.0, 0.3, "buy"))
    snap = value.snapshot(1800)
    tape = snap["visual"]["events"]
    assert len(tape) == 64
    assert tape[-1]["kind"] == "buy"
    assert tape[-1]["size"] == 0.3
    assert tape[-1]["signal"] == sum(e["contribution"] for e in snap["experts"])
    assert tape[-1]["scores"] == [e["score"] for e in snap["experts"]]
    assert [e["id"] for e in tape] == list(range(18, 82))
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


def test_candles_use_all_trades_not_sampled_tape_and_preserve_gaps():
    value = engine()
    value.process(book(1000))
    for i, price in enumerate([100.0, 103.0, 98.0, 101.0]):
        value.process(TradeEvent("replay", "BTCUSDT", 1100 + i, 1100 + i, i, price, 0.5, "buy"))
    value.process(book(2100))  # A book update is not a trade candle.
    value.process(TradeEvent("replay", "BTCUSDT", 3100, 3100, 5, 102.0, 1.0, "sell"))
    snap = value.snapshot(3100)["visual"]["candles"]
    assert snap == [
        {
            "timestamp_ms": 1000,
            "open": 100.0,
            "high": 103.0,
            "low": 98.0,
            "close": 101.0,
            "volume": 2.0,
        },
        {
            "timestamp_ms": 3000,
            "open": 102.0,
            "high": 102.0,
            "low": 102.0,
            "close": 102.0,
            "volume": 1.0,
        },
    ]
    value.process(TradeEvent("replay", "BTCUSDT", 3200, 3200, 6, 105.0, 1.0, "buy"))
    assert snap[-1]["close"] == 102.0  # Published snapshots do not mutate in place.
    for i in range(100):
        ts = 4000 + i * 1000
        value.process(book(ts, i))
        value.process(TradeEvent("replay", "BTCUSDT", ts + 1, ts + 1, i, 100.0, 0.1, "buy"))
    assert len(value.candles) == 90
    assert value.snapshot(103001)["visual"]["candles"][0]["timestamp_ms"] == 14000
    assert value.snapshot(200000)["visual"]["candles"] == []
    assert engine().snapshot(1000)["visual"]["candles"] == []


def test_activations_are_the_exact_last_forward_pass_and_baselines_clear_them():
    import numpy as np

    value = engine()
    value.process(book(1000))
    value.process(book(2000))
    value.process(book(3000))
    visual = value.snapshot(3000)["visual"]
    inputs = np.asarray([(0.0,) * 10] * 28 + list(value.frames), dtype=np.float32).reshape(-1)
    h1 = np.maximum(inputs @ value.gate.w1 + value.gate.b1, 0)
    h2 = np.maximum(h1 @ value.gate.w2 + value.gate.b2, 0)
    assert visual["activations"]["hidden_1"] == h1.tolist()
    assert visual["activations"]["hidden_2"] == h2.tolist()
    logits = h2 @ value.gate.w3 + value.gate.b3
    logits -= logits.max()
    from market_gate.gate import normalize

    assert visual["proposed"] == normalize(np.exp(np.clip(logits, -50, 50)))
    value.process(book(4000))
    assert value.snapshot(4000)["visual"]["activations"] == visual["activations"]
    assert value.snapshot(10000)["visual"]["activations"] is None
    baseline = MarketEngine(LabConfig(gate_mode="uniform"), "models/gate-demo.npz")
    baseline.feed_status = "running"
    baseline.process(book(1000))
    assert baseline.snapshot(1000)["visual"]["activations"] is None
