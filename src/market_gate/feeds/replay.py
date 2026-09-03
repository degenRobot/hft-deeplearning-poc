"""Deterministic local JSONL event feed used by tests, demos, and training."""

import json
from collections.abc import Iterator
from pathlib import Path

from ..contracts import BookEvent, TradeEvent


class ReplayFeed:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def __iter__(self) -> Iterator[BookEvent | TradeEvent]:
        for line in self.path.read_text().splitlines():
            raw = json.loads(line)
            event_type = raw.pop("type")
            yield BookEvent(**raw) if event_type == "book" else TradeEvent(**raw)
