import asyncio
import json
import threading
from datetime import UTC, datetime, timedelta
from urllib.error import HTTPError

import numpy as np
import pytest

from market_gate import historical
from market_gate.historical import fetch_history, load_candle_frames, validate_history_request

START = "2024-01-01T00:00:00Z"
END = "2024-01-01T00:10:00Z"
BASE = 1_704_067_200_000


def api_row(index, close=100.0):
    opened = BASE + index * 1000
    return [
        opened,
        str(close),
        str(close + 1),
        str(close - 1),
        str(close),
        "4",
        opened + 999,
        "400",
        8,
        "3",
        "300",
        "0",
    ]


def native(index, close=100.0):
    return historical._api_candle(api_row(index, close), "ETHUSDT")


def write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("unit tests must not use the network")

    monkeypatch.setattr(historical, "urlopen", fail)
    monkeypatch.setattr(historical, "RATE_LIMIT_SECONDS", 0)


def test_validate_utc_normalization_and_exact_bounds():
    result = validate_history_request("ethusdt", START, "2024-01-01T00:10:00+00:00")
    assert result == {
        "symbol": "ETHUSDT",
        "start": START,
        "end": END,
        "start_ms": BASE,
        "end_ms": BASE + 600_000,
        "seconds": 600,
    }
    assert validate_history_request("BTCUSDT", START, "2024-01-01T05:00:00Z")["seconds"] == 18000


@pytest.mark.parametrize(
    "start,end",
    [
        ("2024-01-01T00:00:00", END),
        ("2024-01-01T01:00:00+01:00", END),
        ("2024-01-01T00:00:00.5Z", END),
        ("2024-01-01T00:00:00.0000001Z", END),
        (START, "2024-01-01T00:09:59Z"),
        (START, "2024-01-01T05:00:01Z"),
        (END, START),
    ],
)
def test_invalid_time_ranges(start, end):
    with pytest.raises(ValueError):
        validate_history_request("ETHUSDT", start, end)


def test_future_and_unsupported_symbol():
    future = datetime.now(UTC).replace(microsecond=0) + timedelta(hours=1)
    with pytest.raises(ValueError, match="closed"):
        validate_history_request(
            "ETHUSDT", (future - timedelta(minutes=10)).isoformat(), future.isoformat()
        )
    with pytest.raises(ValueError, match="symbol"):
        validate_history_request("ETHUSDT&limit=99999", START, END)


def test_request_end_exclusive_and_response_bound(monkeypatch):
    seen = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, count):
            seen.append(count)
            return b"[]"

    def opener(request, timeout):
        seen.append(request.full_url)
        assert timeout == historical.REQUEST_TIMEOUT
        return Response()

    monkeypatch.setattr(historical, "urlopen", opener)
    assert historical._request_page("ETHUSDT", BASE, BASE + 600000) == b"[]"
    assert "endTime=1704067799999" in seen[0]
    assert "interval=1s" in seen[0] and "limit=1000" in seen[0]
    assert seen[1] == historical.MAX_RESPONSE_BYTES + 1


def test_pagination_advances_and_preserves_missing_seconds(tmp_path, monkeypatch):
    calls = []
    pages = [[api_row(0), api_row(2)], [api_row(599)]]

    def request(symbol, start, end):
        calls.append((symbol, start, end))
        return json.dumps(pages.pop(0)).encode()

    monkeypatch.setattr(historical, "_request_page", request)
    progress = []
    path = tmp_path / "history.jsonl"
    receipt = asyncio.run(
        fetch_history(path, "ETHUSDT", START, END, progress_callback=progress.append)
    )
    assert calls == [("ETHUSDT", BASE, BASE + 600000), ("ETHUSDT", BASE + 3000, BASE + 600000)]
    assert receipt["total"] == receipt["candle"] == 3
    assert receipt["book"] == receipt["trade"] == 0
    assert receipt["stop_reason"] == "completed" and receipt["progress"] == 1
    assert receipt["bytes"] == path.stat().st_size
    assert [item["progress"] for item in progress] == [0, 3 / 600, 1, 1]
    dataset, metadata = load_candle_frames(path)
    assert dataset.close_ts_ms.tolist() == [BASE + 1000, BASE + 3000, BASE + 600000]
    assert metadata["candle_count"] == 3
    assert metadata["source_mode"] == "historical_candles_1s"
    assert metadata["first_event_ts_ms"] == receipt["first_event_ts_ms"]
    assert metadata["last_event_ts_ms"] == receipt["last_event_ts_ms"]
    assert len(path.read_text().splitlines()) == 3
    with pytest.raises(FileExistsError):
        asyncio.run(fetch_history(path, "ETHUSDT", START, END))


@pytest.mark.parametrize("indices", [[0, 0], [1, 0], [-1, 0], [0, 600]])
def test_reject_response_duplicates_order_and_out_of_range(tmp_path, monkeypatch, indices):
    monkeypatch.setattr(
        historical,
        "_request_page",
        lambda *args: json.dumps([api_row(index) for index in indices]).encode(),
    )
    with pytest.raises(ValueError, match="advance uniquely"):
        asyncio.run(fetch_history(tmp_path / "bad.jsonl", "ETHUSDT", START, END))
    assert (tmp_path / "bad.jsonl").read_bytes() == b""


def test_reject_cross_page_duplicates(tmp_path, monkeypatch):
    monkeypatch.setattr(
        historical, "_request_page", lambda *args: json.dumps([api_row(0)]).encode()
    )
    with pytest.raises(ValueError, match="advance uniquely"):
        asyncio.run(fetch_history(tmp_path / "bad.jsonl", "ETHUSDT", START, END))
    assert len((tmp_path / "bad.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize(
    "key,value",
    [
        ("close", float("nan")),
        ("volume", float("inf")),
        ("open", 0),
        ("low", 102),
        ("high", 99),
        ("volume", -1),
        ("trades", 1.5),
        ("trades", True),
        ("trades", -1),
        ("taker_buy_volume", 5),
        ("open_ts_ms", BASE + 1),
        ("close_ts_ms", BASE + 1000),
        ("interval", "1m"),
        ("venue", "binance"),
    ],
)
def test_loader_rejects_invalid_native_rows(tmp_path, key, value):
    row = native(0)
    row[key] = value
    with pytest.raises(ValueError):
        load_candle_frames(write(tmp_path / "bad.jsonl", [row]))


@pytest.mark.parametrize(
    "rows",
    [
        [native(1), native(0)],
        [native(0), native(0)],
        [native(0), native(1) | {"symbol": "BTCUSDT"}],
    ],
)
def test_loader_rejects_order_duplicates_and_mixed_symbol(tmp_path, rows):
    with pytest.raises(ValueError):
        load_candle_frames(write(tmp_path / "bad.jsonl", rows))


def test_causal_features_and_gap_reset(tmp_path):
    prefix = [native(index, 100 + index) for index in range(8)]
    a, _ = load_candle_frames(write(tmp_path / "a.jsonl", prefix))
    b, _ = load_candle_frames(write(tmp_path / "b.jsonl", prefix + [native(8, 900)]))
    np.testing.assert_array_equal(a.values, b.values[:8])
    np.testing.assert_array_equal(a.values[:, [3, 4, 5, 8]], 0)
    assert a.values[5, 0] == pytest.approx(105 / 104 - 1)
    assert a.values[5, 1] == pytest.approx(105 / 100 - 1)
    returns = [101 / 100 - 1, 102 / 101 - 1, 103 / 102 - 1, 104 / 103 - 1]
    assert a.values[5, 2] == pytest.approx(np.sqrt(np.mean(np.square(returns))))
    assert a.values[5, 9] == pytest.approx((105 - 102) / 102)
    assert a.values[5, 6] == 0.5 and a.values[5, 7] == 8
    gap, _ = load_candle_frames(write(tmp_path / "gap.jsonl", prefix + [native(10, 500)]))
    np.testing.assert_array_equal(gap.values[-1, [0, 1, 2, 9]], 0)
    assert gap.close_ts_ms[-1] - gap.close_ts_ms[-2] == 3000


def test_zero_volume_flow_is_finite(tmp_path):
    row = native(0) | {"volume": 0, "taker_buy_volume": 0, "trades": 0}
    frames, _ = load_candle_frames(write(tmp_path / "empty-second.jsonl", [row]))
    assert np.isfinite(frames.values).all() and frames.values[0, 6] == 0


def test_rate_limit_retry_after_is_honored(tmp_path, monkeypatch):
    calls, delays = [], []

    async def sleep(delay):
        delays.append(delay)

    def request(*args):
        calls.append(1)
        if len(calls) == 1:
            raise HTTPError("public", 429, "rate limited", {"Retry-After": "2"}, None)
        return b"[]"

    monkeypatch.setattr(historical.asyncio, "sleep", sleep)
    monkeypatch.setattr(historical, "_request_page", request)
    result = asyncio.run(fetch_history(tmp_path / "history.jsonl", "ETHUSDT", START, END))
    assert 2 in delays and len(calls) == 2 and result["candle"] == 0


def test_retry_count_is_bounded(tmp_path, monkeypatch):
    calls = []

    async def sleep(delay):
        pass

    def request(*args):
        calls.append(1)
        raise HTTPError("public", 429, "rate limited", {"Retry-After": "0"}, None)

    monkeypatch.setattr(historical.asyncio, "sleep", sleep)
    monkeypatch.setattr(historical, "_request_page", request)
    with pytest.raises(HTTPError):
        asyncio.run(fetch_history(tmp_path / "history.jsonl", "ETHUSDT", START, END))
    assert len(calls) == historical.MAX_RETRIES + 1


def test_cancellation_drains_worker_and_never_writes_after_return(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def request(*args):
        entered.set()
        release.wait(timeout=3)
        return json.dumps([api_row(0)]).encode()

    monkeypatch.setattr(historical, "_request_page", request)
    path = tmp_path / "history.jsonl"

    async def run():
        task = asyncio.create_task(fetch_history(path, "ETHUSDT", START, END))
        assert await asyncio.to_thread(entered.wait, 3)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert path.read_bytes() == b""

    asyncio.run(run())
    assert path.read_bytes() == b""


def test_page_and_byte_budgets(tmp_path, monkeypatch):
    monkeypatch.setattr(historical, "MAX_PAGES", 1)
    monkeypatch.setattr(
        historical, "_request_page", lambda *args: json.dumps([api_row(0)]).encode()
    )
    with pytest.raises(ValueError, match="pagination bound"):
        asyncio.run(fetch_history(tmp_path / "pages.jsonl", "ETHUSDT", START, END))
    monkeypatch.setattr(historical, "MAX_BYTES", 10)
    with pytest.raises(ValueError, match="byte bound"):
        asyncio.run(fetch_history(tmp_path / "bytes.jsonl", "ETHUSDT", START, END))


def test_source_exhaustion_reports_partial_requested_coverage(tmp_path, monkeypatch):
    pages = [[api_row(100), api_row(101)], []]
    monkeypatch.setattr(
        historical, "_request_page", lambda *args: json.dumps(pages.pop(0)).encode()
    )
    receipt = asyncio.run(fetch_history(tmp_path / "partial.jsonl", "ETHUSDT", START, END))
    assert receipt["stop_reason"] == receipt["feed_status"] == "source_exhausted"
    assert receipt["source_exhausted"]
    assert receipt["coverage_fraction"] == 2 / 600
    assert receipt["missing_candle_count"] == 598
    assert receipt["progress"] == 102 / 600


def test_metadata_labels_unavailable_features_and_internal_gaps(tmp_path):
    _, metadata = load_candle_frames(write(tmp_path / "gaps.jsonl", [native(0), native(5)]))
    for index in [3, 4, 5, 8]:
        assert metadata["feature_names"][index].endswith(" (unavailable)")
    assert metadata["feature_names"][9] == "close_price_distance"
    assert metadata["gap_count"] == 4
    assert metadata["coverage_fraction"] == 2 / 6
