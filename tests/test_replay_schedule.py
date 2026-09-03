from market_gate.contracts import BookEvent
from market_gate.feeds.replay import replay_schedule


def test_replay_cycles_anchor_event_time_monotonically() -> None:
    events = [
        BookEvent("replay", "BTCUSDT", 10, 10, 1, 99, 1, 101, 1),
        BookEvent("replay", "BTCUSDT", 1_010, 1_010, 2, 99, 1, 101, 1),
    ]
    scheduled = list(replay_schedule(events, anchor_ts_ms=1_000, cycles=2))
    timestamps = [item.event.event_ts_ms for item in scheduled]
    assert timestamps == [1_000, 2_000, 3_000, 4_000]
    assert [item.delay_ms for item in scheduled] == [0, 1_000, 1_000, 1_000]
