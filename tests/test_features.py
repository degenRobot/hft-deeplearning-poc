from market_gate.config import LabConfig
from market_gate.contracts import BookEvent, TradeEvent
from market_gate.engine import MarketEngine
from market_gate.features import FeatureBuilder


def test_next_second_book_does_not_leak_into_prior_frame() -> None:
    engine = MarketEngine(LabConfig(), "models/gate-demo.npz")
    engine.process(BookEvent("replay", "BTCUSDT", 1_000, 1_000, 1, 99.0, 1.0, 101.0, 1.0))
    engine.process(TradeEvent("replay", "BTCUSDT", 1_500, 1_500, 2, 101.0, 0.2, "buy"))
    engine.process(BookEvent("replay", "BTCUSDT", 2_000, 2_000, 3, 199.0, 1.0, 201.0, 1.0))
    frame = engine.frames[-1]
    assert frame[3] == 200.0  # Prior book's spread, not the new 200-price book.
    assert frame[7] == 1.0
    assert frame[8] == 1.0  # Trade events are not quote updates.


def test_five_second_return_uses_five_frames_not_full_history() -> None:
    builder = FeatureBuilder()
    builder.begin_second(0)
    for second, mid in enumerate(range(100, 106), start=1):
        builder.observe_book()
        frame = builder.close_before(second * 1_000, float(mid), 1.0, 0.0, float(mid))
    assert frame is not None
    assert frame.values[1] == 105 / 100 - 1
