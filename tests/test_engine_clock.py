from market_gate.config import LabConfig
from market_gate.contracts import BookEvent, TradeEvent
from market_gate.engine import MarketEngine


def book(timestamp_ms: int, bid: float = 99.0) -> BookEvent:
    return BookEvent(
        "replay", "BTCUSDT", timestamp_ms, timestamp_ms, timestamp_ms, bid, 1, bid + 2, 1
    )


def test_gate_revision_increments_and_refresh_value_is_remaining_duration() -> None:
    engine = MarketEngine(LabConfig(gate_mode="uniform", gate_interval_ms=1_000))
    assert engine.snapshot(now_ms=0)["gate"]["revision"] == 0
    engine.process(book(1_000))
    assert engine.snapshot(now_ms=1_000)["gate"].get("revision") == 1
    assert engine.snapshot(now_ms=1_000)["gate"]["next_refresh_ms"] == 1_000
    engine.process(book(1_500))
    assert engine.snapshot(now_ms=1_500)["gate"]["revision"] == 1
    assert engine.snapshot(now_ms=1_500)["gate"]["next_refresh_ms"] == 500
    engine.process(book(2_000))
    assert engine.snapshot(now_ms=2_000)["gate"]["revision"] == 2


def test_out_of_order_event_cannot_mutate_engine_state() -> None:
    engine = MarketEngine(LabConfig(gate_mode="uniform"))
    engine.process(book(2_000, bid=199.0))
    before = engine.snapshot(now_ms=2_000)
    result = engine.process(TradeEvent("replay", "BTCUSDT", 1_000, 1_000, 8, 99.0, 4.0, "sell"))
    after = engine.snapshot(now_ms=2_000)
    assert result is None
    assert after["timestamp"] == before["timestamp"]
    assert after["market"] == before["market"]
    assert after["paper"] == before["paper"]
