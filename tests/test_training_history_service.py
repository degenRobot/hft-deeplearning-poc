"""Historical requests use the capture owner, validator and immutable selection."""

import asyncio
import json
import os
import threading

import pytest
from conftest import asgi_request
from test_training_data import fixture_recording, pending

from market_gate.api import create_app
from market_gate.training_data import PublicDatasetService, _write_state, capture_dataset

REQUEST = {"symbol": "ETHUSDT", "start": "2026-01-01T00:00:00Z", "end": "2026-01-01T00:10:00Z"}


@pytest.mark.parametrize(
    "change",
    [
        {"symbol": "INVALID"},
        {"symbol": 1},
        {"start": "2026-01-01T00:00:00"},
        {"start": "2026-01-01T00:00:00.001Z"},
        {"end": "2026-01-01T00:09:59Z"},
        {"end": "2026-01-01T05:00:01Z"},
        {"end": "2099-01-01T00:10:00Z"},
    ],
)
def test_invalid_history_cannot_spawn(tmp_path, monkeypatch, change):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid request spawned a worker")

    monkeypatch.setattr("market_gate.training_data.subprocess.Popen", forbidden)
    with pytest.raises(ValueError):
        PublicDatasetService(tmp_path).start_history(**(REQUEST | change))


def test_history_uses_same_owned_process_lock_as_live_capture(tmp_path, monkeypatch):
    fixture_recording(tmp_path / "data/training-public.jsonl")
    service = PublicDatasetService(tmp_path)
    commands = []

    class Process:
        def __init__(self, command, *, cwd, stdout, stderr, pass_fds):
            self.fd = os.dup(pass_fds[0])
            self.returncode = None
            commands.append(command)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0
            os.close(self.fd)

        def wait(self, timeout):
            return 0

    monkeypatch.setattr("market_gate.training_data.subprocess.Popen", Process)
    capture = service.start_history(**REQUEST)["capture"]
    try:
        assert capture["mode"] == "historical"
        assert capture["requested_seconds"] == 600
        assert capture["symbol"] == "ETHUSDT"
        assert capture["start"] == REQUEST["start"]
        assert capture["end"] == REQUEST["end"]
        assert capture["progress"] == 0
        assert capture["can_stop"]
        command = commands[0]
        assert command[command.index("--mode") + 1] == "historical"
        assert command[command.index("--symbol") + 1] == "ETHUSDT"
        for start in (
            lambda: service.start(30),
            lambda: service.start_history(**REQUEST),
            lambda: PublicDatasetService(tmp_path).start(30),
            lambda: PublicDatasetService(tmp_path).start_history(**REQUEST),
        ):
            with pytest.raises(ValueError, match="capture is active"):
                start()
    finally:
        service.stop()
    assert not service.snapshot()["capture"]["can_stop"]
    service.start(30)
    try:
        with pytest.raises(ValueError, match="capture is active"):
            PublicDatasetService(tmp_path).start_history(**REQUEST)
    finally:
        service.stop()


@pytest.mark.parametrize(
    "payload",
    [
        {"symbol": "ETHUSDT"},
        REQUEST | {"path": "/private/secret"},
        REQUEST | {"start": 0},
        REQUEST | {"end": None},
        REQUEST | {"symbol": "INVALID"},
    ],
)
def test_history_api_rejects_invalid_payload(tmp_path, payload):
    app = create_app()
    app.state.training_service.datasets = PublicDatasetService(tmp_path)
    status, body = asyncio.run(asgi_request(app, "POST", "/training/data/history", payload))
    assert status == 400
    assert "/private/secret" not in body["detail"]


def test_history_api_origin_and_thread_dispatch(monkeypatch):
    app = create_app()
    caller = threading.get_ident()
    observed = []

    def start(**request):
        observed.append((request, threading.get_ident()))
        return {"capture": {"status": "running"}}

    monkeypatch.setattr(app.state.training_service.datasets, "start_history", start)
    status, _ = asyncio.run(
        asgi_request(app, "POST", "/training/data/history", REQUEST, "https://evil.example")
    )
    assert status == 403
    assert observed == []
    status, body = asyncio.run(
        asgi_request(app, "POST", "/training/data/history", REQUEST, "http://localhost:3000")
    )
    assert status == 200
    assert body["capture"]["status"] == "running"
    assert observed[0][0] == REQUEST
    assert observed[0][1] != caller


def candle_recording(path, count=600):
    path.parent.mkdir(parents=True, exist_ok=True)
    from market_gate.historical import validate_history_request

    start = validate_history_request(**REQUEST)["start_ms"]
    with path.open("x") as stream:
        stream.write("\n")  # Format detection accepts the first nonempty record.
        for second in range(count):
            timestamp = start + second * 1000
            stream.write(
                json.dumps(
                    {
                        "kind": "candle",
                        "venue": "binance_spot_candles",
                        "symbol": "ETHUSDT",
                        "interval": "1s",
                        "open_ts_ms": timestamp,
                        "close_ts_ms": timestamp + 999,
                        "open": 100,
                        "close": 100.01,
                        "low": 99,
                        "high": 101,
                        "volume": 10,
                        "taker_buy_volume": 6,
                        "trades": 3,
                    }
                )
                + "\n"
            )


def pending_history(tmp_path):
    service = pending(tmp_path)
    state = service._state()
    state["capture"].update(REQUEST, mode="historical", requested_seconds=600, progress=0.0)
    _write_state(service.path, state)
    return service


@pytest.mark.parametrize("count,expected", [(600, "completed"), (30, "incomplete")])
def test_history_selects_only_training_ready_native_candles(tmp_path, count, expected):
    service = pending_history(tmp_path)
    original = service.selected_recording().read_bytes()

    async def recorder(output, symbol, *args, **kwargs):
        assert symbol == "ETHUSDT"
        candle_recording(output, count)
        return {"stop_reason": "completed"}

    result = asyncio.run(capture_dataset(tmp_path, "a" * 32, 600, recorder, history=REQUEST))
    capture = result["capture"]
    assert capture["status"] == expected
    assert capture["events"] == capture["candle_count"] == count
    assert capture["book_count"] == capture["trade_count"] == 0
    assert (tmp_path / "data/training-public.jsonl").read_bytes() == original
    if expected == "completed":
        selected = result["selected"]
        assert capture["progress"] == 1
        assert selected["id"] == "a" * 32
        assert selected["symbol"] == "ETHUSDT"
        assert selected["source"] == "binance_historical_candles"
        assert selected["source_mode"] == "historical_candles_1s"
        assert selected["label"] == "Historical Binance 1s candles"
        assert selected["book_count"] == selected["trade_count"] == 0
        assert selected["candle_count"] == count
        assert selected["limitations"]
        assert service.selected_recording().name == "a" * 32 + ".jsonl"
    else:
        assert capture["progress"] < 1
        assert result["selected"]["id"] == "builtin"


def test_history_progress_uses_coverage_and_stop_preserves_previous_selection(tmp_path):
    service = pending_history(tmp_path)

    async def recorder(output, *args, progress_callback, **kwargs):
        candle_recording(output)
        progress_callback(
            {
                "total": 600,
                "book": 0,
                "trade": 0,
                "candle": 600,
                "bytes": output.stat().st_size,
                "first_event_ts_ms": 1000,
                "last_event_ts_ms": 600000,
                "feed_status": "fetching",
                "reconnects": 0,
                "progress": 0.5,
            }
        )
        await asyncio.Event().wait()

    async def scenario():
        task = asyncio.create_task(
            capture_dataset(
                tmp_path, "a" * 32, 600, recorder, history=REQUEST, heartbeat_seconds=0.005
            )
        )
        await asyncio.sleep(0.03)
        first = service.snapshot()["capture"]
        assert first["progress"] == 0.5
        assert first["candle_count"] == 600
        await asyncio.sleep(0.03)
        second = service.snapshot()["capture"]
        assert second["progress"] == 0.5
        assert second["updated_at"] != first["updated_at"]
        task.cancel()
        return await task

    result = asyncio.run(scenario())
    assert result["capture"]["status"] == "stopped"
    assert result["capture"]["training_ready"]
    assert result["capture"]["progress"] == 0.5
    assert result["selected"]["id"] == "builtin"


def test_early_source_exhaustion_does_not_select_trainable_partial_range(tmp_path):
    pending_history(tmp_path)

    async def recorder(output, *args, progress_callback, **kwargs):
        candle_recording(output, 599)
        progress_callback(
            {
                "total": 599,
                "book": 0,
                "trade": 0,
                "candle": 599,
                "bytes": output.stat().st_size,
                "first_event_ts_ms": 1000,
                "last_event_ts_ms": 599000,
                "feed_status": "source_exhausted",
                "reconnects": 0,
                "progress": 599 / 600,
                "coverage_fraction": 599 / 600,
                "missing_candle_count": 1,
            }
        )
        return {"stop_reason": "source_exhausted"}

    result = asyncio.run(capture_dataset(tmp_path, "a" * 32, 600, recorder, history=REQUEST))
    assert result["capture"]["status"] == "incomplete"
    assert result["capture"]["training_ready"]
    assert result["capture"]["progress"] < 1
    assert result["capture"]["missing_candle_count"] == 1
    assert "requested end" in result["capture"]["error"]
    assert result["selected"]["id"] == "builtin"
