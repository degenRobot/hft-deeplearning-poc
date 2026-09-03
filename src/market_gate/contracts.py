"""Normalized event and decision contracts shared by feeds, engine, and API."""

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class BookEvent:
    venue: str
    symbol: str
    event_ts_ms: int
    receive_ts_ms: int
    update_id: int
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float


@dataclass(frozen=True)
class TradeEvent:
    venue: str
    symbol: str
    event_ts_ms: int
    receive_ts_ms: int
    trade_id: int
    price: float
    size: float
    aggressor: Literal["buy", "sell"]


@dataclass(frozen=True)
class FeatureFrame:
    close_ts_ms: int
    values: tuple[float, ...]
    healthy: bool


@dataclass(frozen=True)
class DecisionFrame:
    timestamp: int
    scores: dict[str, float]
    weights: dict[str, float]
    contribution: dict[str, float]
    quote: dict[str, float] | None
    risk_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
