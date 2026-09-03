"""Public Binance market-data adapter: bookTicker plus aggregate trades only."""

import asyncio
import json
import time
from collections.abc import AsyncIterator

from ..contracts import BookEvent, TradeEvent


class BinancePublicFeed:
    """Reconnects with bounded exponential backoff; it never accepts credentials."""

    def __init__(self, symbol: str) -> None:
        stream = symbol.lower()
        self.url = (
            "wss://data-stream.binance.vision:443/stream?streams="
            f"{stream}@bookTicker/{stream}@aggTrade"
        )
        self.reconnects = 0

    async def events(self) -> AsyncIterator[BookEvent | TradeEvent]:
        import websockets

        delay = 0.5
        while True:
            try:
                async with websockets.connect(
                    self.url, ping_interval=20, ping_timeout=20
                ) as socket:
                    delay = 0.5
                    async for message in socket:
                        event = self.normalize(json.loads(message), int(time.time() * 1000))
                        if event is not None:
                            yield event
            except Exception:
                # Network errors are represented by reconnect count, not propagated to UI.
                self.reconnects += 1
                await asyncio.sleep(delay)
                delay = min(delay * 2, 10.0)

    @staticmethod
    def normalize(raw: dict[str, object], receive_ts_ms: int) -> BookEvent | TradeEvent | None:
        data = raw.get("data", raw)
        if not isinstance(data, dict):
            return None
        kind = data.get("e")
        if kind == "bookTicker":
            return BookEvent(
                "binance",
                str(data["s"]),
                # bookTicker's core schema does not require an event-time field.
                int(data.get("E", receive_ts_ms)),
                receive_ts_ms,
                int(data["u"]),
                float(data["b"]),
                float(data["B"]),
                float(data["a"]),
                float(data["A"]),
            )
        if kind == "aggTrade":
            # Binance's m=true means buyer is maker, therefore sell aggression.
            aggressor = "sell" if bool(data["m"]) else "buy"
            return TradeEvent(
                "binance",
                str(data["s"]),
                int(data["E"]),
                receive_ts_ms,
                int(data["a"]),
                float(data["p"]),
                float(data["q"]),
                aggressor,
            )
        return None
