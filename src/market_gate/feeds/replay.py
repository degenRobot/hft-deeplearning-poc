"""Deterministic local JSONL events and a wall-clock anchored replay timeline."""

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from ..contracts import BookEvent, TradeEvent

MarketEvent = BookEvent | TradeEvent


@dataclass(frozen=True)
class ScheduledReplayEvent:
    event: MarketEvent
    delay_ms: int


class ReplayFeed:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def __iter__(self) -> Iterator[MarketEvent]:
        for line in self.path.read_text().splitlines():
            raw = json.loads(line)
            event_type = raw.pop("type")
            yield BookEvent(**raw) if event_type == "book" else TradeEvent(**raw)


def replay_schedule(
    events: Sequence[MarketEvent], anchor_ts_ms: int, cycles: int | None = 1
) -> Iterator[ScheduledReplayEvent]:
    """Shift fixture timestamps to a monotonic wall-clock timeline and retain event deltas.

    Fixture delays are replayed exactly. A cycle reserves one second after its final event.
    """
    if not events:
        return
    first_ts_ms = events[0].event_ts_ms
    duration_ms = max(event.event_ts_ms for event in events) - first_ts_ms + 1_000
    cycle = 0
    previous_target_ms: int | None = None
    while cycles is None or cycle < cycles:
        base_ts_ms = anchor_ts_ms + cycle * duration_ms
        for event in events:
            target_ts_ms = base_ts_ms + event.event_ts_ms - first_ts_ms
            shifted = replace(event, event_ts_ms=target_ts_ms, receive_ts_ms=target_ts_ms)
            delay_ms = 0 if previous_target_ms is None else target_ts_ms - previous_target_ms
            yield ScheduledReplayEvent(shifted, delay_ms)
            previous_target_ms = target_ts_ms
        cycle += 1
