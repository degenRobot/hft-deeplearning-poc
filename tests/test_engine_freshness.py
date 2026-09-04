from dataclasses import replace

import pytest

from market_gate.config import LabConfig
from market_gate.contracts import BookEvent, TradeEvent
from market_gate.engine import MarketEngine
from market_gate.feeds.replay import replay_schedule


def book(timestamp: int, bid: float = 99.0) -> BookEvent:
    return BookEvent("replay", "BTCUSDT", timestamp, timestamp, timestamp, bid, 1, bid + 2, 1)


def trade(timestamp: int, price: float = 102.0) -> TradeEvent:
    return TradeEvent("replay", "BTCUSDT", timestamp, timestamp, timestamp, price, 0.1, "buy")


def engine() -> MarketEngine:
    result = MarketEngine(LabConfig(gate_mode="uniform", stale_after_ms=100, max_inventory=1))
    result.feed_status = "running"
    result.process(book(1_000))
    return result


def test_gap_expires_fill_eligibility_even_without_a_snapshot_read() -> None:
    market = engine()
    decision = market.process(trade(1_101))
    assert market.paper.inventory == 0
    assert market.prior_quote is None
    assert decision is not None and decision.quote is None
    assert market.ledger_rows()[-1]["risk_reason"] == "stale_data"
    assert market.snapshot(now_ms=1_101)["health"]["ready"] is False


def test_fresh_trades_do_not_refresh_book_and_recovered_book_restores_quotes() -> None:
    market = engine()
    for timestamp in (1_050, 1_100, 1_150, 1_200):
        market.process(trade(timestamp, price=100))
    snapshot = market.snapshot(now_ms=1_200)
    assert snapshot["health"]["message_age_ms"] == 0
    assert snapshot["health"]["book_age_ms"] == 200
    assert snapshot["health"]["ready"] is False
    assert snapshot["quote"] is None
    market.process(book(1_201))
    assert market.snapshot(now_ms=1_201)["health"]["ready"] is True
    market.process(trade(1_202))
    assert market.paper.inventory == pytest.approx(-0.1)


def test_delayed_trade_does_not_fill_or_quote_against_fresh_book() -> None:
    market = engine()
    market.process(book(1_200))
    delayed = replace(trade(1_001), venue="binance", receive_ts_ms=1_201)
    decision = market.process(delayed)
    assert market.paper.inventory == 0
    assert decision is not None and decision.quote is None
    assert decision.risk_reason == "stale_data"


def test_delayed_book_is_not_fresh_just_because_it_arrived_now() -> None:
    market = engine()
    market.process(replace(book(1_001), venue="binance", receive_ts_ms=1_201))
    snapshot = market.snapshot(now_ms=1_201)
    assert snapshot["health"]["book_age_ms"] == 200
    assert snapshot["health"]["ready"] is False
    assert snapshot["quote"] is None


def test_book_only_move_marks_position_without_creating_a_fill() -> None:
    market = engine()
    market.process(trade(1_001))
    cash = market.paper.cash
    market.process(book(1_002, bid=109))
    snapshot = market.snapshot(now_ms=1_002)
    assert market.paper.cash == cash
    assert snapshot["paper"]["inventory"] == pytest.approx(-0.1)
    assert snapshot["paper"]["pnl"] == pytest.approx(-0.9)


@pytest.mark.parametrize("event", [book(1_001), trade(1_001)])
def test_wrong_symbol_event_does_not_mutate_state(event: BookEvent | TradeEvent) -> None:
    market = engine()
    before = market.snapshot(now_ms=1_001)
    rows = market.ledger_rows()
    assert market.process(replace(event, symbol="ETHUSDT")) is None
    assert market.snapshot(now_ms=1_001) == before
    assert market.ledger_rows() == rows


def test_rebased_replay_uses_book_age_not_original_fixture_epoch() -> None:
    market = MarketEngine(LabConfig(gate_mode="uniform", stale_after_ms=100))
    market.feed_status = "running"
    scheduled = list(replay_schedule([book(10), trade(60)], anchor_ts_ms=10_000))
    for item in scheduled:
        market.process(item.event, arrival_ts_ms=item.event.receive_ts_ms + 5)
    snapshot = market.snapshot(now_ms=10_055)
    assert snapshot["health"]["ready"] is True
    assert snapshot["health"]["book_age_ms"] == 55
    assert snapshot["quote"] is not None


def test_invalid_book_cancels_quote_until_valid_book_recovers() -> None:
    market = engine()
    market.process(replace(book(1_001), ask_price=98))
    market.process(trade(1_002))
    assert market.paper.inventory == 0
    assert market.prior_quote is None
    assert market.snapshot(now_ms=1_002)["health"]["ready"] is False
    market.process(book(1_003))
    assert market.snapshot(now_ms=1_003)["quote"] is not None
