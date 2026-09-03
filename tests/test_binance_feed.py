from market_gate.contracts import BookEvent, TradeEvent
from market_gate.feeds.binance import BinancePublicFeed


def test_book_ticker_without_event_time_uses_receive_time() -> None:
    event = BinancePublicFeed.normalize(
        {
            "data": {
                "e": "bookTicker",
                "u": 7,
                "s": "BTCUSDT",
                "b": "99",
                "B": "1",
                "a": "101",
                "A": "2",
            }
        },
        receive_ts_ms=123_456,
    )
    assert isinstance(event, BookEvent)
    assert event.event_ts_ms == 123_456
    assert event.receive_ts_ms == 123_456


def test_aggregate_trade_maker_flag_maps_to_sell_aggression() -> None:
    event = BinancePublicFeed.normalize(
        {
            "data": {
                "e": "aggTrade",
                "E": 123,
                "a": 8,
                "s": "BTCUSDT",
                "p": "100",
                "q": "0.3",
                "m": True,
            }
        },
        receive_ts_ms=124,
    )
    assert isinstance(event, TradeEvent)
    assert event.aggressor == "sell"
