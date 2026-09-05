"""Bounded public one-second candle downloads and causal proxy feature frames."""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

from .features import FeatureBuilder
from .training import FEATURE_NAMES, FrameDataset

SUPPORTED_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT")
VENUE = "binance_spot_candles"
SOURCE_MODE = "historical_candles_1s"
HISTORICAL_FEATURE_NAMES = [
    name + " (unavailable)"
    if index in (3, 4, 5, 8)
    else "close_price_distance"
    if index == 9
    else name
    for index, name in enumerate(FEATURE_NAMES)
]
MAX_ROWS = 18_000
MAX_BYTES = 100_000_000
MAX_RESPONSE_BYTES = 2_000_000
MAX_PAGES = 40
PAGE_LIMIT = 1000
REQUEST_TIMEOUT = 15
RATE_LIMIT_SECONDS = 0.15
MAX_RETRIES = 3
LIMITATIONS = [
    "Public Binance spot one-second candles; historical availability may contain gaps.",
    "Candle closes proxy midprices; candle taker-buy volume proxies signed trade flow.",
    "Spread, book imbalance, microprice displacement, and quote updates are unavailable and zero.",
    "No reconstructed order books or synthetic trades; results are an offline approximation.",
]


def validate_history_request(symbol: str, start: str, end: str) -> dict[str, object]:
    """Validate a UTC whole-second, end-exclusive interval of ten minutes to five hours."""
    if not isinstance(symbol, str) or symbol.upper() not in SUPPORTED_SYMBOLS:
        raise ValueError("unsupported historical symbol")

    def parse(value: str) -> datetime:
        if not isinstance(value, str) or not value.endswith(("Z", "+00:00")):
            raise ValueError("history timestamps must use explicit UTC Z or +00:00")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.0+)?(?:Z|\+00:00)", value):
            raise ValueError("history timestamps must be ISO UTC timestamps with whole seconds")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("history timestamps must be ISO UTC timestamps") from error
        if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
            raise ValueError("history timestamps must use UTC")
        if parsed.microsecond:
            raise ValueError("history timestamps must use whole seconds")
        if parsed.year < 1970:
            raise ValueError("history timestamps must be after the Unix epoch")
        return parsed

    first, last = parse(start), parse(end)
    seconds = int((last - first).total_seconds())
    if not 600 <= seconds <= MAX_ROWS:
        raise ValueError("historical interval must be between 600 and 18000 seconds")
    if last > datetime.now(UTC).replace(microsecond=0):
        raise ValueError("historical end must include only closed bars")
    return {
        "symbol": symbol.upper(),
        "start": first.isoformat().replace("+00:00", "Z"),
        "end": last.isoformat().replace("+00:00", "Z"),
        "start_ms": int(first.timestamp()) * 1000,
        "end_ms": int(last.timestamp()) * 1000,
        "seconds": seconds,
    }


def _validate_candle(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError("candle must be an object")
    if raw.get("kind") != "candle" or raw.get("venue") != VENUE:
        raise ValueError("expected native Binance spot candle record")
    if raw.get("symbol") not in SUPPORTED_SYMBOLS or raw.get("interval") != "1s":
        raise ValueError("unsupported candle symbol or interval")
    for key in ("open_ts_ms", "close_ts_ms", "trades"):
        value = raw.get(key)
        if type(value) is not int or value < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if raw["trades"] > 2**31 - 1:
        raise ValueError("candle trade count exceeds bound")
    if raw["open_ts_ms"] % 1000 or raw["close_ts_ms"] != raw["open_ts_ms"] + 999:
        raise ValueError("candle timestamps must describe one complete second")
    if raw["close_ts_ms"] >= int(datetime.now(UTC).timestamp()) * 1000:
        raise ValueError("candle must be closed")
    result = dict(raw)
    for key in ("open", "high", "low", "close", "volume", "taker_buy_volume"):
        value = raw.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be a finite nonnegative number")
        if key in ("open", "high", "low", "close") and value == 0:
            raise ValueError(f"{key} must be positive")
        result[key] = float(value)
    if not (
        result["low"]
        <= min(result["open"], result["close"])
        <= max(result["open"], result["close"])
        <= result["high"]
    ):
        raise ValueError("candle OHLC ordering is invalid")
    if result["taker_buy_volume"] > result["volume"]:
        raise ValueError("taker buy volume exceeds total volume")
    return result


def _api_candle(row: object, symbol: str) -> dict[str, object]:
    if not isinstance(row, list) or len(row) != 12:
        raise ValueError("malformed Binance candle response")
    # Binance represents prices and volumes as decimal strings; timestamps/counts are integers.
    for index in (1, 2, 3, 4, 5, 9):
        if type(row[index]) not in (str, int, float):
            raise ValueError("invalid Binance candle numeric value")
    try:
        return _validate_candle(
            {
                "kind": "candle",
                "venue": VENUE,
                "symbol": symbol,
                "interval": "1s",
                "open_ts_ms": row[0],
                "close_ts_ms": row[6],
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "taker_buy_volume": float(row[9]),
                "trades": row[8],
            }
        )
    except (TypeError, OverflowError) as error:
        raise ValueError("invalid Binance candle numeric value") from error


def _request_page(symbol: str, start_ms: int, end_ms: int) -> bytes:
    """Network only: a cancelled caller can never leave a background file writer."""
    query = urlencode(
        {
            "symbol": symbol,
            "interval": "1s",
            "startTime": start_ms,
            "endTime": end_ms - 1,
            "limit": PAGE_LIMIT,
        }
    )
    request = Request(
        "https://data-api.binance.vision/api/v3/klines?" + query,
        headers={"User-Agent": "market-gate-lab/0.1"},
    )
    with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        data = response.read(MAX_RESPONSE_BYTES + 1)
    if len(data) > MAX_RESPONSE_BYTES:
        raise ValueError("historical response exceeds byte bound")
    return data


def _retry_delay(error: HTTPError, attempt: int) -> float:
    value = error.headers.get("Retry-After") if error.headers else None
    delay = float(2**attempt)
    if value:
        try:
            delay = float(value)
        except ValueError:
            try:
                delay = (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
            except (TypeError, ValueError, OverflowError) as invalid:
                raise ValueError("invalid provider Retry-After") from invalid
    if not math.isfinite(delay) or delay > 30:
        raise ValueError("provider requested a retry beyond the bounded download window")
    return max(RATE_LIMIT_SECONDS, delay)


async def _fetch_page(symbol: str, start_ms: int, end_ms: int) -> bytes:
    # Shield the network-only worker and drain it before a cancelled fetch returns.
    # File ownership never leaves the caller's thread.
    worker = asyncio.create_task(asyncio.to_thread(_request_page, symbol, start_ms, end_ms))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not worker.cancelled():
            worker.exception()
        raise


async def fetch_history(
    output: Path,
    symbol: str,
    start: str,
    end: str,
    *,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Create an exclusive JSONL recording; cancellation leaves at most a closed partial file."""
    request = validate_history_request(symbol, start, end)
    symbol = request["symbol"]
    cursor, end_ms = request["start_ms"], request["end_ms"]
    count = size = response_bytes = 0
    first_ts = last_ts = None
    started = time.monotonic()

    def receipt(completed: bool = False) -> dict[str, object]:
        exhausted = completed and cursor < end_ms
        stop_reason = "source_exhausted" if exhausted else "completed" if completed else "running"
        return {
            "output": str(output.resolve()),
            **request,
            "total": count,
            "book": 0,
            "trade": 0,
            "candle": count,
            "bytes": size,
            "first_event_ts_ms": first_ts,
            "last_event_ts_ms": last_ts,
            "feed_status": stop_reason if completed else "fetching",
            "reconnects": 0,
            "progress": (cursor - request["start_ms"]) / (end_ms - request["start_ms"]),
            "requested_candle_count": request["seconds"],
            "missing_candle_count": request["seconds"] - count,
            "coverage_fraction": count / request["seconds"],
            "source_exhausted": exhausted,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "stop_reason": stop_reason,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        if progress_callback:
            progress_callback(receipt())
        for _ in range(MAX_PAGES):
            if cursor >= end_ms:
                break
            for attempt in range(MAX_RETRIES + 1):
                await asyncio.sleep(RATE_LIMIT_SECONDS)
                try:
                    payload = await _fetch_page(symbol, cursor, end_ms)
                    break
                except HTTPError as error:
                    if error.code not in (429, 500, 502, 503, 504) or attempt == MAX_RETRIES:
                        raise
                    await asyncio.sleep(_retry_delay(error, attempt))
                except (URLError, TimeoutError):
                    if attempt == MAX_RETRIES:
                        raise
                    await asyncio.sleep(2**attempt)
            response_bytes += len(payload)
            if response_bytes > MAX_BYTES or len(payload) > MAX_RESPONSE_BYTES:
                raise ValueError("historical responses exceed byte bound")
            rows = json.loads(payload)
            if not isinstance(rows, list) or len(rows) > PAGE_LIMIT:
                raise ValueError("historical response must be a bounded candle list")
            if not rows:
                break
            candles = [_api_candle(row, symbol) for row in rows]
            previous = cursor - 1000
            for candle in candles:
                opened = candle["open_ts_ms"]
                if opened < cursor or opened >= end_ms or opened <= previous:
                    raise ValueError("historical candles must advance uniquely within request")
                previous = opened
            if count + len(candles) > MAX_ROWS:
                raise ValueError("historical candles exceed row bound")
            lines = [
                (json.dumps(candle, separators=(",", ":"), allow_nan=False) + "\n").encode()
                for candle in candles
            ]
            batch_size = sum(map(len, lines))
            if size + batch_size > MAX_BYTES:
                raise ValueError("historical recording exceeds byte bound")
            # All writes stay on the event loop, with no await before flush/close.
            handle.writelines(lines)
            handle.flush()
            size += batch_size
            count += len(candles)
            first_ts = first_ts if first_ts is not None else candles[0]["close_ts_ms"] + 1
            last_ts = candles[-1]["close_ts_ms"] + 1
            cursor = candles[-1]["open_ts_ms"] + 1000
            if progress_callback:
                progress_callback(receipt())
        else:
            if cursor < end_ms:
                raise ValueError("historical download exceeds pagination bound")
    result = receipt(completed=True)
    if progress_callback:
        progress_callback(result)
    return result


def load_candle_frames(path: Path) -> tuple[FrameDataset, dict[str, object]]:
    """Map native candles to causal features; reset history across every missing second."""
    if not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError("historical recording missing or exceeds byte bound")
    builder = FeatureBuilder()
    values, mids, timestamps = [], [], []
    symbol = None
    previous = None
    consumed = 0
    with path.open("rb") as handle:
        while line := handle.readline(MAX_RESPONSE_BYTES + 1):
            consumed += len(line)
            if len(line) > MAX_RESPONSE_BYTES or consumed > MAX_BYTES:
                raise ValueError("historical recording exceeds byte bound")
            if not line.strip():
                continue
            if len(values) >= MAX_ROWS:
                raise ValueError("historical recording exceeds row bound")
            candle = _validate_candle(json.loads(line))
            opened = candle["open_ts_ms"]
            if symbol is not None and symbol != candle["symbol"]:
                raise ValueError("historical recording must contain one symbol")
            if previous is not None and opened <= previous:
                raise ValueError("historical candles must be unique and chronological")
            if previous is not None and opened != previous + 1000:
                builder = FeatureBuilder()
            symbol = candle["symbol"]
            builder.begin_second(opened)
            builder.total_volume = candle["volume"]
            builder.signed_volume = candle["taker_buy_volume"] - (
                candle["volume"] - candle["taker_buy_volume"]
            )
            builder.arrivals = candle["trades"]
            frame = builder.close_before(opened + 1000, candle["close"], 0, 0, candle["close"])
            if not all(
                math.isfinite(value) and abs(value) <= np.finfo(np.float32).max
                for value in frame.values
            ):
                raise ValueError("candle features exceed finite float32 bounds")
            values.append(frame.values)
            mids.append(candle["close"])
            timestamps.append(candle["close_ts_ms"] + 1)
            previous = opened
    if not values:
        raise ValueError("historical recording is empty")
    dataset = FrameDataset(
        np.asarray(values, dtype=np.float32),
        np.asarray(mids, dtype=np.float64),
        np.asarray(timestamps, dtype=np.int64),
    )
    return dataset, {
        "symbol": symbol,
        "venue": VENUE,
        "event_count": len(values),
        "candle_count": len(values),
        "first_event_ts_ms": timestamps[0],
        "last_event_ts_ms": timestamps[-1],
        "feature_names": list(HISTORICAL_FEATURE_NAMES),
        "gap_count": (timestamps[-1] - timestamps[0]) // 1000 + 1 - len(values),
        "coverage_fraction": len(values) / ((timestamps[-1] - timestamps[0]) // 1000 + 1),
        "source_mode": SOURCE_MODE,
        "limitations": list(LIMITATIONS),
    }
