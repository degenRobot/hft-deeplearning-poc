from market_gate.config import LabConfig
from market_gate.contracts import BookEvent
from market_gate.engine import MarketEngine
from market_gate.risk import make_quote


def test_stale_data_suppresses_quote() -> None:
    result = make_quote(
        mid=100.0,
        observed_spread_bps=2.0,
        signal=0.2,
        inventory=0.0,
        now_ms=5_000,
        message_ts_ms=1_000,
        base_spread_bps=2.0,
        max_inventory=1.0,
        stale_after_ms=2_000,
    )
    assert result.quote is None
    assert result.reason == "stale_data"


def test_stopped_feed_snapshot_hides_prior_quote() -> None:
    engine = MarketEngine(LabConfig(stale_after_ms=100))
    engine.process(BookEvent("replay", "BTCUSDT", 1_000, 1_000, 1, 99.0, 1.0, 101.0, 1.0))
    assert engine.snapshot(now_ms=1_101)["quote"] is None
