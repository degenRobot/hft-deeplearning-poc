"""Explicit toy paper fills; this is not a queue, latency, or execution simulator."""

from dataclasses import dataclass

from .contracts import TradeEvent


@dataclass
class PaperLedger:
    inventory: float = 0.0
    cash: float = 0.0
    pnl: float = 0.0

    def observe_trade(
        self, trade: TradeEvent, prior_quote: dict[str, float], mark: float, max_inventory: float
    ) -> None:
        """Apply a toy maker fill only when a taker crosses the prior synthetic quote."""
        if trade.aggressor == "buy" and trade.price >= prior_quote["ask"]:
            filled = min(trade.size, max(0.0, self.inventory + max_inventory))
            self.inventory -= filled
            self.cash += prior_quote["ask"] * filled
        elif trade.aggressor == "sell" and trade.price <= prior_quote["bid"]:
            filled = min(trade.size, max(0.0, max_inventory - self.inventory))
            self.inventory += filled
            self.cash -= prior_quote["bid"] * filled
        self.pnl = self.cash + self.inventory * mark
