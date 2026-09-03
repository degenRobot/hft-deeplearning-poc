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
    assert engine.snapshot(now_ms=1_000)["gate"]["model_version"] == "not used"
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


def test_missing_neural_artifact_is_reported_as_uniform_fallback() -> None:
    engine = MarketEngine(LabConfig(gate_mode="neural"), "models/does-not-exist.npz")
    engine.process(book(1_000))
    gate = engine.snapshot(now_ms=1_000)["gate"]
    assert gate["mode"] == "uniform-fallback"
    assert gate["model_version"] == "unavailable"


def test_binance_events_share_receive_clock_when_exchange_times_differ() -> None:
    engine = MarketEngine(LabConfig(gate_mode="uniform"))
    live_book = BookEvent("binance", "BTCUSDT", 10_000, 10_000, 1, 99.0, 1.0, 101.0, 1.0)
    delayed_trade = TradeEvent("binance", "BTCUSDT", 9_000, 10_001, 2, 101.0, 0.1, "buy")

    assert engine.process(live_book) is not None
    assert engine.process(delayed_trade) is not None
    assert engine.last_ts_ms == 10_001


def test_flow_expert_uses_a_bounded_recent_trade_window() -> None:
    engine = MarketEngine(LabConfig(gate_mode="uniform", flow_window_trades=4))
    engine.process(book(1_000))
    trades = [TradeEvent("replay", "BTCUSDT", 1_001, 1_001, 1, 101.0, 10.0, "buy")] + [
        TradeEvent("replay", "BTCUSDT", 1_002 + index, 1_002 + index, index + 2, 99.0, 1.0, "sell")
        for index in range(4)
    ]
    for trade in trades:
        engine.process(trade)
    assert list(engine.recent_trades) == [-1.0, -1.0, -1.0, -1.0]
    assert engine.snapshot(now_ms=2_000)["market"]["trade_flow"] == -4.0


def test_event_counters_distinguish_accepted_and_late_events() -> None:
    engine = MarketEngine(LabConfig(gate_mode="uniform"))
    engine.process(book(2_000))
    engine.process(TradeEvent("replay", "BTCUSDT", 1_000, 1_000, 8, 99.0, 4.0, "sell"))
    health = engine.snapshot(now_ms=2_000)["health"]
    assert health["events_processed"] == 1
    assert health["late_events_dropped"] == 1
    assert health["feed_generation"] == 0


def test_snapshot_before_first_message_is_waiting_not_stale() -> None:
    engine = MarketEngine(LabConfig(gate_mode="uniform"))
    engine.feed_status = "connecting"
    snapshot = engine.snapshot(now_ms=1_800_000_000_000)
    assert snapshot["timestamp"] == 0
    assert snapshot["quote"] is None
    assert snapshot["health"]["ready"] is False
    assert snapshot["health"]["risk_reason"] == "waiting_for_data"
    assert snapshot["health"]["message_age_ms"] == 0
