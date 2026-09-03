from market_gate.contracts import TradeEvent
from market_gate.paper import PaperLedger


def test_toy_paper_fill_uses_prior_quote_and_marks_pnl() -> None:
    paper = PaperLedger()
    paper.observe_trade(
        TradeEvent("replay", "BTCUSDT", 1, 1, 1, 101.0, 0.25, "buy"),
        {"bid": 99.0, "ask": 100.0},
        mark=100.0,
        max_inventory=1.0,
    )
    assert paper.inventory == -0.25
    assert paper.pnl == 0.0


def test_toy_fill_respects_inventory_cap() -> None:
    paper = PaperLedger(inventory=-0.9)
    paper.observe_trade(
        TradeEvent("replay", "BTCUSDT", 1, 1, 1, 101.0, 0.5, "buy"),
        {"bid": 99.0, "ask": 100.0},
        mark=100.0,
        max_inventory=1.0,
    )
    assert paper.inventory == -1.0
