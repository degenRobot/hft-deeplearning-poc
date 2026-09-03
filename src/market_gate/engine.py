"""Single-writer event engine: experts advise; deterministic risk owns synthetic quotes."""

import time
from collections import deque

from .config import LabConfig
from .contracts import BookEvent, DecisionFrame, TradeEvent
from .experts import EXPERT_IDS, microprice_pressure, short_reversion, trade_flow_impulse
from .features import FeatureBuilder
from .gate import TinyMLPGate, blend_and_smooth, static_weights, uniform_weights
from .risk import make_quote


class MarketEngine:
    def __init__(self, config: LabConfig, model_path: str | None = None) -> None:
        self.config = config
        self.gate = TinyMLPGate(model_path)
        self.features = FeatureBuilder()
        self.frames: deque[tuple[float, ...]] = deque(maxlen=30)
        self.ledger: deque[dict[str, object]] = deque(maxlen=300)
        self.bid = self.ask = self.bid_size = self.ask_size = 0.0
        self.last_trade: float | None = None
        self.last_ts_ms = 0
        self.last_receive_ts_ms = 0
        self.venue = config.source
        self.reconnects = 0
        self.inventory = self.pnl = 0.0
        self.weights = uniform_weights()
        self.last_gate_ts_ms = -config.gate_interval_ms
        self.signed_volume = self.total_volume = 0.0
        self.trade_arrivals = 0
        self.fair_history: deque[float] = deque(maxlen=12)
        self.decision: DecisionFrame | None = None

    def process(
        self, event: BookEvent | TradeEvent, arrival_ts_ms: int | None = None
    ) -> DecisionFrame | None:
        self.venue = event.venue
        self.last_ts_ms = max(self.last_ts_ms, event.event_ts_ms)
        arrival_ts_ms = event.receive_ts_ms if arrival_ts_ms is None else arrival_ts_ms
        self.last_receive_ts_ms = max(self.last_receive_ts_ms, arrival_ts_ms)
        if isinstance(event, BookEvent):
            self.bid, self.bid_size = event.bid_price, event.bid_size
            self.ask, self.ask_size = event.ask_price, event.ask_size
        else:
            self.last_trade = event.price
            signed = event.size if event.aggressor == "buy" else -event.size
            self.signed_volume += signed
            self.total_volume += event.size
            self.trade_arrivals += 1
            self.features.observe_trade(event.size, event.aggressor)
        if self.bid <= 0 or self.ask <= 0:
            return None
        mid = (self.bid + self.ask) / 2
        spread_bps = 10_000 * (self.ask - self.bid) / mid
        imbalance = (self.bid_size - self.ask_size) / max(self.bid_size + self.ask_size, 1e-9)
        microprice = (self.ask * self.bid_size + self.bid * self.ask_size) / max(
            self.bid_size + self.ask_size, 1e-9
        )
        self.features.observe_book(mid)
        frame = self.features.advance(event.event_ts_ms, mid, spread_bps, imbalance, microprice)
        if frame is not None:
            self.frames.append(frame.values)
        if event.event_ts_ms - self.last_gate_ts_ms >= self.config.gate_interval_ms:
            self._refresh_weights(event.event_ts_ms)
        fair = sum(self.fair_history) / len(self.fair_history) if self.fair_history else mid
        self.fair_history.append(mid)
        scores = {
            "microprice": microprice_pressure(mid, imbalance, microprice),
            "flow": trade_flow_impulse(self.signed_volume, self.total_volume, self.trade_arrivals),
            "reversion": short_reversion(mid, fair),
        }
        contributions = {
            key: scores[key] * self.weights[key] * self.config.expert_strength for key in EXPERT_IDS
        }
        signal = sum(contributions.values())
        outcome = make_quote(
            mid=mid,
            observed_spread_bps=spread_bps,
            signal=signal,
            inventory=self.inventory,
            now_ms=event.event_ts_ms,
            message_ts_ms=self.last_ts_ms,
            base_spread_bps=self.config.base_spread_bps,
            max_inventory=self.config.max_inventory,
            stale_after_ms=self.config.stale_after_ms,
        )
        self.decision = DecisionFrame(
            event.event_ts_ms,
            scores,
            self.weights.copy(),
            contributions,
            outcome.quote,
            outcome.reason,
        )
        self.ledger.append(self.decision.as_dict())
        return self.decision

    def _refresh_weights(self, timestamp_ms: int) -> None:
        self.last_gate_ts_ms = timestamp_ms
        if self.config.gate_mode == "uniform":
            proposed = uniform_weights()
        elif self.config.gate_mode == "static":
            proposed = static_weights()
        else:
            padded = [(0.0,) * 10] * max(0, 30 - len(self.frames)) + list(self.frames)
            proposed = self.gate.predict(padded)
        self.weights = blend_and_smooth(proposed, self.weights, self.config.higher_level_influence)

    def snapshot(self, now_ms: int | None = None) -> dict[str, object]:
        """Expose current state; a stopped feed must not retain an actionable-looking quote."""
        now_ms = int(time.time() * 1000) if now_ms is None else now_ms
        mid = (self.bid + self.ask) / 2 if self.bid and self.ask else 0.0
        spread_bps = 10_000 * (self.ask - self.bid) / mid if mid else 0.0
        imbalance = (self.bid_size - self.ask_size) / max(self.bid_size + self.ask_size, 1e-9)
        scores = self.decision.scores if self.decision else dict.fromkeys(EXPERT_IDS, 0.0)
        contribution = (
            self.decision.contribution if self.decision else dict.fromkeys(EXPERT_IDS, 0.0)
        )
        message_age_ms = max(0, now_ms - self.last_receive_ts_ms)
        quote = self.decision.quote if self.decision else None
        if message_age_ms > self.config.stale_after_ms:
            quote = None
        return {
            "timestamp": self.last_ts_ms,
            "source": self.venue,
            "symbol": self.config.symbol,
            "market": {
                "mid": mid,
                "spread_bps": spread_bps,
                "imbalance": imbalance,
                "last_trade": self.last_trade,
                "trade_flow": self.signed_volume,
            },
            "gate": {
                "mode": self.config.gate_mode,
                "regime": "demo",
                "confidence": max(self.weights.values()),
                "weights": self.weights.copy(),
                "cadence_ms": self.config.gate_interval_ms,
                "model_version": self.gate.model_version,
            },
            "experts": [
                {
                    "id": key,
                    "label": key.replace("_", " ").title(),
                    "score": scores[key],
                    "weight": self.weights[key],
                    "contribution": contribution[key],
                }
                for key in EXPERT_IDS
            ],
            "quote": quote,
            "paper": {"inventory": self.inventory, "pnl": self.pnl},
            "health": {
                "status": "ok" if quote else "guarded",
                "message_age_ms": message_age_ms,
                "reconnects": self.reconnects,
            },
        }

    def ledger_rows(self) -> list[dict[str, object]]:
        return list(self.ledger)
