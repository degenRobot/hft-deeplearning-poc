"""Single-writer event engine: experts advise; deterministic risk owns synthetic quotes."""

import time
from collections import deque
from math import isfinite

from .config import LabConfig
from .contracts import BookEvent, DecisionFrame, TradeEvent
from .experts import EXPERT_IDS, microprice_pressure, short_reversion, trade_flow_impulse
from .features import FeatureBuilder
from .gate import blend_and_smooth, load_numpy_gate, static_weights, uniform_weights
from .paper import PaperLedger
from .risk import make_quote


class MarketEngine:
    def __init__(self, config: LabConfig, model_path: str | None = None) -> None:
        self.config = config
        self.gate = load_numpy_gate(model_path)
        self.features = FeatureBuilder()
        self.frames: deque[tuple[float, ...]] = deque(maxlen=30)
        self.ledger: deque[dict[str, object]] = deque(maxlen=300)
        self.bid = self.ask = self.bid_size = self.ask_size = 0.0
        self.last_trade: float | None = None
        self.last_ts_ms = 0
        self.last_receive_ts_ms = 0
        self.last_book_event_ts_ms: int | None = None
        self.last_book_receive_ts_ms: int | None = None
        self.book_valid = False
        self.venue = config.source
        self.reconnects = 0
        self.feed_status = "starting"
        self.feed_generation = 0
        self.events_processed = 0
        self.late_events_dropped = 0
        self.run_id = "unstarted"
        self.paper = PaperLedger()
        self.prior_quote: dict[str, float] | None = None
        self.weights = uniform_weights()
        self.last_gate_ts_ms = -config.gate_interval_ms
        self.gate_revision = 0
        self.effective_gate_mode = config.gate_mode
        self.recent_trades: deque[float] = deque(maxlen=config.flow_window_trades)
        self.fair_history: deque[float] = deque(maxlen=12)
        self.decision: DecisionFrame | None = None

    def process(
        self, event: BookEvent | TradeEvent, arrival_ts_ms: int | None = None
    ) -> DecisionFrame | None:
        if event.symbol != self.config.symbol:
            return None
        # Binance bookTicker has no exchange event time, so live events share the local
        # receive clock for sequencing. Replay keeps its deterministic fixture clock.
        sequence_ts_ms = event.receive_ts_ms if event.venue == "binance" else event.event_ts_ms
        if sequence_ts_ms < self.last_ts_ms:
            self.late_events_dropped += 1
            return None
        arrival_ts_ms = event.receive_ts_ms if arrival_ts_ms is None else arrival_ts_ms
        # Display expiry and fill eligibility use the same book clock. A fresh trade
        # cannot revive an old book, and a delayed trade cannot fill a current quote.
        if (
            not self._book_fresh(arrival_ts_ms)
            or arrival_ts_ms - event.event_ts_ms > self.config.stale_after_ms
        ):
            self.prior_quote = None
        self.events_processed += 1
        # Close the prior second before this event mutates book/trade state.
        if self.bid > 0 and self.ask > 0:
            prior_mid = (self.bid + self.ask) / 2
            prior_spread_bps = 10_000 * (self.ask - self.bid) / prior_mid
            prior_imbalance = (self.bid_size - self.ask_size) / max(
                self.bid_size + self.ask_size, 1e-9
            )
            prior_microprice = (self.ask * self.bid_size + self.bid * self.ask_size) / max(
                self.bid_size + self.ask_size, 1e-9
            )
            frame = self.features.close_before(
                sequence_ts_ms,
                prior_mid,
                prior_spread_bps,
                prior_imbalance,
                prior_microprice,
            )
            if frame is not None:
                self.frames.append(frame.values)
        self.venue = event.venue
        self.last_ts_ms = sequence_ts_ms
        self.last_receive_ts_ms = max(self.last_receive_ts_ms, arrival_ts_ms)
        if isinstance(event, BookEvent):
            self.book_valid = (
                all(
                    isfinite(value)
                    for value in (event.bid_price, event.ask_price, event.bid_size, event.ask_size)
                )
                and 0 < event.bid_price <= event.ask_price
                and event.bid_size >= 0
                and event.ask_size >= 0
                and event.bid_size + event.ask_size > 0
            )
            if not self.book_valid:
                self.prior_quote = None
                return None
            self.bid, self.bid_size = event.bid_price, event.bid_size
            self.ask, self.ask_size = event.ask_price, event.ask_size
            self.last_book_event_ts_ms = event.event_ts_ms
            self.last_book_receive_ts_ms = arrival_ts_ms
            self.paper.mark_to_market((self.bid + self.ask) / 2)
            self.features.observe_book()
        else:
            self.last_trade = event.price
            prior_mid = (self.bid + self.ask) / 2 if self.bid and self.ask else event.price
            if self.prior_quote is not None:
                self.paper.observe_trade(
                    event, self.prior_quote, prior_mid, self.config.max_inventory
                )
            signed = event.size if event.aggressor == "buy" else -event.size
            self.recent_trades.append(signed)
            self.features.observe_trade(event.size, event.aggressor)
        if not self.book_valid:
            return None
        mid = (self.bid + self.ask) / 2
        spread_bps = 10_000 * (self.ask - self.bid) / mid
        imbalance = (self.bid_size - self.ask_size) / max(self.bid_size + self.ask_size, 1e-9)
        microprice = (self.ask * self.bid_size + self.bid * self.ask_size) / max(
            self.bid_size + self.ask_size, 1e-9
        )
        self.features.begin_second(sequence_ts_ms)
        if sequence_ts_ms - self.last_gate_ts_ms >= self.config.gate_interval_ms:
            self._refresh_weights(sequence_ts_ms)
        fair = sum(self.fair_history) / len(self.fair_history) if self.fair_history else mid
        self.fair_history.append(mid)
        signed_volume = sum(self.recent_trades)
        total_volume = sum(abs(size) for size in self.recent_trades)
        scores = {
            "microprice": microprice_pressure(mid, imbalance, microprice),
            "flow": trade_flow_impulse(signed_volume, total_volume, len(self.recent_trades)),
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
            inventory=self.paper.inventory,
            now_ms=arrival_ts_ms,
            message_ts_ms=min(
                event.event_ts_ms,
                self.last_book_event_ts_ms or 0,
                self.last_book_receive_ts_ms or 0,
            ),
            base_spread_bps=self.config.base_spread_bps,
            max_inventory=self.config.max_inventory,
            stale_after_ms=self.config.stale_after_ms,
        )
        self.decision = DecisionFrame(
            sequence_ts_ms,
            scores,
            self.weights.copy(),
            contributions,
            outcome.quote,
            outcome.reason,
        )
        self.ledger.append(self.decision.as_dict())
        self.prior_quote = outcome.quote
        return self.decision

    def _book_age(self, now_ms: int) -> int:
        if self.last_book_event_ts_ms is None or self.last_book_receive_ts_ms is None:
            return 0
        return max(0, now_ms - min(self.last_book_event_ts_ms, self.last_book_receive_ts_ms))

    def _book_fresh(self, now_ms: int) -> bool:
        return self.book_valid and self._book_age(now_ms) <= self.config.stale_after_ms

    def _refresh_weights(self, timestamp_ms: int) -> None:
        self.last_gate_ts_ms = timestamp_ms
        self.gate_revision += 1
        if self.config.gate_mode == "uniform":
            proposed = uniform_weights()
            self.effective_gate_mode = "uniform"
        elif self.config.gate_mode == "static":
            proposed = static_weights()
            self.effective_gate_mode = "static"
        elif self.gate is None:
            proposed = uniform_weights()
            self.effective_gate_mode = "uniform-fallback"
        else:
            padded = [(0.0,) * 10] * max(0, 30 - len(self.frames)) + list(self.frames)
            proposed = self.gate.predict(padded)
            self.effective_gate_mode = "neural"
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
        has_message = self.last_receive_ts_ms > 0
        message_age_ms = max(0, now_ms - self.last_receive_ts_ms) if has_message else 0
        quote = self.decision.quote if self.decision else None
        risk_reason = self.decision.risk_reason if self.decision else "waiting_for_data"
        if has_message and message_age_ms > self.config.stale_after_ms:
            quote = None
            risk_reason = "stale_data"
        if has_message and not self._book_fresh(now_ms):
            quote = None
            risk_reason = "stale_data" if self.book_valid else "waiting_for_valid_book"
        ready = (
            self.last_ts_ms > 0
            and has_message
            and self.feed_status == "running"
            and message_age_ms <= self.config.stale_after_ms
            and self._book_fresh(now_ms)
        )
        if self.effective_gate_mode == "neural" and self.gate is not None:
            model_version = self.gate.model_version
        elif self.effective_gate_mode == "uniform-fallback":
            model_version = "unavailable"
        else:
            model_version = "not used"
        return {
            "timestamp": self.last_ts_ms,
            "source": self.venue,
            "symbol": self.config.symbol,
            "market": {
                "mid": mid,
                "spread_bps": spread_bps,
                "imbalance": imbalance,
                "last_trade": self.last_trade,
                "trade_flow": sum(self.recent_trades),
            },
            "gate": {
                "mode": self.effective_gate_mode,
                "regime": "demo",
                "confidence": max(self.weights.values()),
                "weights": self.weights.copy(),
                "cadence_ms": self.config.gate_interval_ms,
                "model_version": model_version,
                "revision": self.gate_revision,
                "next_refresh_ms": max(
                    0, self.last_gate_ts_ms + self.config.gate_interval_ms - now_ms
                ),
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
            "paper": {"inventory": self.paper.inventory, "pnl": self.paper.pnl},
            "health": {
                "status": "ok" if quote else "guarded",
                "ready": ready,
                "feed_status": self.feed_status,
                "risk_reason": risk_reason,
                "run_id": self.run_id,
                "message_age_ms": message_age_ms,
                "book_age_ms": self._book_age(now_ms),
                "reconnects": self.reconnects,
                "events_processed": self.events_processed,
                "late_events_dropped": self.late_events_dropped,
                "feed_generation": self.feed_generation,
            },
        }

    def ledger_rows(self) -> list[dict[str, object]]:
        return list(self.ledger)
