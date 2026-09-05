"""Record a small public Binance book-and-trade sample for the training demo."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from market_gate.contracts import BookEvent, TradeEvent
from market_gate.feeds.binance import BinancePublicFeed
from market_gate.training import write_recording


async def collect_events(
    symbol: str, seconds: float, max_events: int, book_interval_ms: int
) -> list[BookEvent | TradeEvent]:
    """Collect bounded *saved* events, with a deadline even if the feed is silent."""
    feed = BinancePublicFeed(symbol)
    events: list[BookEvent | TradeEvent] = []
    last_book_ts_ms: int | None = None
    try:
        async with asyncio.timeout(seconds):
            async for event in feed.events():
                if isinstance(event, BookEvent):
                    elapsed_ms = event.event_ts_ms - (last_book_ts_ms or event.event_ts_ms)
                    if last_book_ts_ms is not None and elapsed_ms < book_interval_ms:
                        continue
                    last_book_ts_ms = event.event_ts_ms
                events.append(event)
                if len(events) >= max_events:
                    return events
    except TimeoutError:
        return events
    return events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--max-events", type=int, default=20_000)
    parser.add_argument("--book-interval-ms", type=int, default=1_000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seconds <= 0 or args.max_events < 1 or args.book_interval_ms < 1:
        raise SystemExit("--seconds, --max-events, and --book-interval-ms must be positive")
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("--output must be a new file; refusing to overwrite")
    events = asyncio.run(
        collect_events(args.symbol, args.seconds, args.max_events, args.book_interval_ms)
    )
    if not events:
        raise SystemExit("no public events recorded before the deadline")
    counts = write_recording(args.output, events, args.book_interval_ms)
    print(f"recorded {counts['total']} normalized events to {args.output}")


if __name__ == "__main__":
    main()
