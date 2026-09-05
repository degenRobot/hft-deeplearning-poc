"""Record bounded public Binance book-and-trade data for the training demo."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from collections.abc import Callable
from pathlib import Path

from market_gate.contracts import BookEvent, TradeEvent
from market_gate.feeds.binance import BinancePublicFeed
from market_gate.training import event_to_record, should_keep_book


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
                    if not should_keep_book(event.event_ts_ms, last_book_ts_ms, book_interval_ms):
                        continue
                    last_book_ts_ms = event.event_ts_ms
                events.append(event)
                if len(events) >= max_events:
                    return events
    except TimeoutError:
        return events
    return events


async def record_events(
    output: Path,
    symbol: str,
    seconds: float,
    max_events: int,
    book_interval_ms: int,
    max_bytes: int,
    progress_seconds: float = 30,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Stream a new file with hard duration, saved-event and serialized-byte caps."""
    feed = BinancePublicFeed(symbol)
    counts = {"book": 0, "trade": 0, "total": 0}
    size_bytes = 0
    first_ts_ms: int | None = None
    last_ts_ms: int | None = None
    last_book_ts_ms: int | None = None
    started = time.monotonic()
    next_progress = started + progress_seconds
    stop_reason = "running"

    def receipt() -> dict[str, object]:
        return {
            "output": str(output.resolve()),
            "symbol": symbol.upper(),
            **counts,
            "bytes": size_bytes,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "first_event_ts_ms": first_ts_ms,
            "last_event_ts_ms": last_ts_ms,
            "reconnects": feed.reconnects,
            "feed_status": feed.status,
            "book_interval_ms": book_interval_ms,
            "stop_reason": stop_reason,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        try:
            async with asyncio.timeout(seconds):
                async for event in feed.events():
                    if isinstance(event, BookEvent):
                        if not should_keep_book(
                            event.event_ts_ms, last_book_ts_ms, book_interval_ms
                        ):
                            continue
                        last_book_ts_ms = event.event_ts_ms
                    line = (
                        json.dumps(event_to_record(event), sort_keys=True, separators=(",", ":"))
                        + "\n"
                    ).encode("utf-8")
                    if size_bytes + len(line) > max_bytes:
                        stop_reason = "max_bytes"
                        break
                    handle.write(line)
                    size_bytes += len(line)
                    counts["book" if isinstance(event, BookEvent) else "trade"] += 1
                    counts["total"] += 1
                    first_ts_ms = (
                        event.event_ts_ms
                        if first_ts_ms is None
                        else min(first_ts_ms, event.event_ts_ms)
                    )
                    last_ts_ms = (
                        event.event_ts_ms
                        if last_ts_ms is None
                        else max(last_ts_ms, event.event_ts_ms)
                    )
                    if progress_callback is not None:
                        progress_callback(receipt())
                    if time.monotonic() >= next_progress:
                        handle.flush()
                        print(json.dumps(receipt(), sort_keys=True), flush=True)
                        next_progress = time.monotonic() + progress_seconds
                    if counts["total"] >= max_events:
                        stop_reason = "max_events"
                        break
                else:
                    stop_reason = "feed_ended"
        except TimeoutError:
            stop_reason = "deadline"
    result = receipt()
    if progress_callback is not None:
        progress_callback(result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--max-events", type=int, default=20_000)
    parser.add_argument("--max-bytes", type=int, default=100_000_000)
    parser.add_argument("--book-interval-ms", type=int, default=1_000)
    parser.add_argument("--progress-seconds", type=float, default=30)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        not math.isfinite(args.seconds)
        or args.seconds <= 0
        or args.max_events < 1
        or args.book_interval_ms < 1
        or args.max_bytes < 1
        or not math.isfinite(args.progress_seconds)
        or args.progress_seconds <= 0
    ):
        raise SystemExit("duration, event/byte caps, and intervals must be positive")
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("--output must be a new file; refusing to overwrite")
    counts = asyncio.run(
        record_events(
            args.output,
            args.symbol,
            args.seconds,
            args.max_events,
            args.book_interval_ms,
            args.max_bytes,
            args.progress_seconds,
        )
    )
    print(json.dumps(counts, sort_keys=True), flush=True)
    if not counts["total"]:
        raise SystemExit("no public events recorded before the deadline")


if __name__ == "__main__":
    main()
