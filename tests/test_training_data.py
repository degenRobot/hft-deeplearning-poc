import asyncio
import importlib.util
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from market_gate.contracts import BookEvent
from market_gate.training import write_recording
from market_gate.training_data import (
    PublicDatasetService,
    _write_state,
    capture_dataset,
    inspect_recording,
)


def fixture_recording(path, seconds=650):
    events = [
        BookEvent(
            "binance", "BTCUSDT", second * 1000, second * 1000 + 1, second, 100.0, 1.0, 100.01, 2.0
        )
        for second in range(seconds)
    ]
    write_recording(path, events, 100)
    return path


def pending(root, capture_id="a" * 32):
    fixture_recording(root / "data/training-public.jsonl")
    service = PublicDatasetService(root)
    state = service.snapshot()
    state["capture"] = {
        "id": capture_id,
        "status": "running",
        "requested_seconds": 30,
        "elapsed_seconds": 0.0,
        "started_at": "test",
        "events": 0,
        "bytes": 0,
    }
    _write_state(service.path, state)
    return service


def test_readiness_uses_real_contiguous_split(tmp_path):
    short = fixture_recording(tmp_path / "short.jsonl", 30)
    result = inspect_recording(tmp_path, short, "a" * 32)
    assert result["event_count"] == 30
    assert result["frame_count"] == 29
    assert not result["training_ready"]
    assert "insufficient contiguous frames" in result["error"]
    ready = inspect_recording(tmp_path, fixture_recording(tmp_path / "ready.jsonl"), "b" * 32)
    assert ready["training_ready"]
    assert len(ready["sha256"]) == 64


@pytest.mark.parametrize("count,expected", [(30, "incomplete"), (650, "completed")])
def test_capture_selects_only_valid_new_recording(tmp_path, count, expected):
    service = pending(tmp_path)
    builtin = service.selected_recording().read_bytes()

    async def recorder(output, symbol, seconds, max_events, interval, max_bytes, **kwargs):
        assert (symbol, seconds, max_events, interval, max_bytes) == (
            "BTCUSDT",
            30,
            500000,
            100,
            100000000,
        )
        fixture_recording(output, count)
        return {"stop_reason": "deadline"}

    result = asyncio.run(capture_dataset(tmp_path, "a" * 32, 30, recorder))
    assert result["capture"]["status"] == expected
    assert not result["capture"]["can_stop"]
    assert result["capture"]["events"] == count
    assert (
        result["capture"]["progress"] == 1.0
        if count == 650
        else result["capture"]["progress"] < 1.0
    )
    assert result["selected"]["id"] == ("builtin" if count == 30 else "a" * 32)
    assert (tmp_path / "data/training-public.jsonl").read_bytes() == builtin
    assert service.selected_recording().is_file()


def test_quiet_feed_progress_and_cancel_preserve_previous_selection(tmp_path):
    service = pending(tmp_path)

    async def recorder(output, *args, **kwargs):
        fixture_recording(output, 650)  # Even valid data must not be selected after a stop.
        await asyncio.Event().wait()

    async def scenario():
        task = asyncio.create_task(
            capture_dataset(tmp_path, "a" * 32, 30, recorder, heartbeat_seconds=0.01)
        )
        await asyncio.sleep(0.06)
        during = service.snapshot()["capture"]
        assert during["status"] == "running"
        assert during["elapsed_seconds"] > 0
        assert during["progress"] > 0
        task.cancel()
        return await task

    result = asyncio.run(scenario())
    assert result["capture"]["status"] == "stopped"
    assert result["selected"]["id"] == "builtin"
    assert result["capture"]["events"] == 650
    assert service.snapshot()["capture"]["status"] == "stopped"


def test_short_capture_preserves_prior_successful_capture(tmp_path):
    service = pending(tmp_path)

    async def ready(output, *args, **kwargs):
        fixture_recording(output, 650)
        return {"stop_reason": "deadline"}

    asyncio.run(capture_dataset(tmp_path, "a" * 32, 30, ready))
    previous_path = service.selected_recording()
    previous_bytes = previous_path.read_bytes()
    state = service.snapshot()
    state["capture"] = {"id": "b" * 32, "status": "running"}
    _write_state(service.path, state)

    async def short(output, *args, **kwargs):
        fixture_recording(output, 30)
        return {"stop_reason": "deadline"}

    result = asyncio.run(capture_dataset(tmp_path, "b" * 32, 30, short))
    assert result["capture"]["status"] == "incomplete"
    assert service.selected_recording() == previous_path
    assert previous_path.read_bytes() == previous_bytes


def test_restart_detects_dead_worker_and_live_lock(tmp_path):
    service = pending(tmp_path)
    assert service.snapshot()["capture"]["status"] == "failed"
    lock = service._acquire()
    try:
        restarted = PublicDatasetService(tmp_path)
        result = restarted.snapshot()["capture"]
        assert result["status"] == "running"
        assert not result["can_stop"]
        with pytest.raises(ValueError, match="capture is active"):
            restarted.start(30)
    finally:
        lock.close()


def test_start_passes_lock_and_stop_terminates_owned_worker(tmp_path, monkeypatch):
    fixture_recording(tmp_path / "data/training-public.jsonl")
    service = PublicDatasetService(tmp_path)
    observed = {}

    class Process:
        def __init__(self, command, *, cwd, stdout, stderr, pass_fds):
            self.fd = os.dup(pass_fds[0])
            self.returncode = None
            observed["command"] = command

        def poll(self):
            return self.returncode

        def terminate(self):
            observed["terminated"] = True
            state = json.loads(service.path.read_text())
            state["capture"]["status"] = "stopped"
            _write_state(service.path, state)
            self.returncode = 0
            os.close(self.fd)

        def wait(self, timeout):
            return self.returncode

    monkeypatch.setattr("market_gate.training_data.subprocess.Popen", Process)
    result = service.start(1800)
    assert result["capture"]["status"] == "running"
    assert result["capture"]["can_stop"]
    assert "--lock-fd" in observed["command"]
    assert observed["command"][observed["command"].index("--seconds") + 1] == "1800"
    with pytest.raises(ValueError, match="capture is active"):
        service.start(30)
    assert service.stop()["capture"]["status"] == "stopped"
    assert observed["terminated"]


@pytest.mark.parametrize("duration", [True, 29, 1801, 30.0, "30"])
def test_invalid_durations_cannot_spawn(tmp_path, duration):
    with pytest.raises(ValueError, match="30 to 1800"):
        PublicDatasetService(tmp_path).start(duration)


def test_existing_recorder_progress_callback_and_caps(tmp_path, monkeypatch):
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location(
        "capture_recorder_test", root / "scripts/record_binance.py"
    )
    recorder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recorder)

    class Feed:
        reconnects, status = 0, "running"

        def __init__(self, symbol):
            assert symbol == "BTCUSDT"

        async def events(self):
            for timestamp in (0, 50, 100, 200):
                yield BookEvent(
                    "binance", "BTCUSDT", timestamp, timestamp, timestamp, 100, 1, 101, 1
                )

    monkeypatch.setattr(recorder, "BinancePublicFeed", Feed)
    updates = []
    result = asyncio.run(
        recorder.record_events(
            tmp_path / "events.jsonl",
            "BTCUSDT",
            30,
            2,
            100,
            100000000,
            progress_callback=updates.append,
        )
    )
    assert result["total"] == 2
    assert result["stop_reason"] == "max_events"
    assert result["last_event_ts_ms"] == 100
    assert updates[-1] == result
    assert result["bytes"] == (tmp_path / "events.jsonl").stat().st_size


def test_stop_during_validation_keeps_lock_and_previous_selection(tmp_path, monkeypatch):
    import market_gate.training_data as data

    service = pending(tmp_path)
    previous = service.selected_recording()
    original_inspect = data.inspect_recording
    original_write = data._write_state
    release_validation = threading.Event()
    validation_finished = threading.Event()
    writes = []
    clock = [0.0]
    monkeypatch.setattr(data, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    def record_write(path, state):
        writes.append((state["capture"]["status"], state["capture"].get("progress")))
        original_write(path, state)

    monkeypatch.setattr(data, "_write_state", record_write)

    async def recorder(output, *args, **kwargs):
        fixture_recording(output, 650)
        return {"stop_reason": "deadline"}

    async def scenario():
        entered = asyncio.Event()
        loop = asyncio.get_running_loop()

        def inspect(*args):
            clock[0] = 60.0  # Deadline passed; validation must still stay below 100%.
            loop.call_soon_threadsafe(entered.set)
            release_validation.wait()
            try:
                return original_inspect(*args)
            finally:
                validation_finished.set()

        monkeypatch.setattr(data, "inspect_recording", inspect)
        task = asyncio.create_task(
            capture_dataset(tmp_path, "a" * 32, 30, recorder, heartbeat_seconds=0.005)
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            await asyncio.sleep(0.02)
            first = service.snapshot()["capture"]
            assert first["status"] == "validating"
            assert first["progress"] == 0.99
            await asyncio.sleep(0.02)
            assert service.snapshot()["capture"]["updated_at"] != first["updated_at"]
            task.cancel()
            await asyncio.sleep(0.01)
            task.cancel()  # Repeated Stop must also drain the same validator.
            await asyncio.sleep(0.01)
            assert not task.done()
            assert service.snapshot()["capture"]["status"] == "stopping"
            with pytest.raises(ValueError, match="capture is active"):
                service._acquire()
            assert service.selected_recording() == previous
        finally:
            release_validation.set()
            result = await task
        assert validation_finished.is_set()
        assert result["capture"]["status"] == "stopped"
        assert result["capture"]["training_ready"] is True
        assert result["capture"]["progress"] == 0.99
        assert result["selected"]["id"] == "builtin"
        assert service.selected_recording() == previous
        lock = service._acquire()
        lock.close()
        final_write_count = len(writes)
        await asyncio.sleep(0.02)
        assert len(writes) == final_write_count  # No writer survives the released lock.
        assert all(progress < 1.0 for _, progress in writes)

    asyncio.run(scenario())


def test_incomplete_deadline_capture_never_reports_full_progress(tmp_path, monkeypatch):
    import market_gate.training_data as data

    pending(tmp_path)
    clock = [0.0]
    monkeypatch.setattr(data, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    async def recorder(output, *args, **kwargs):
        fixture_recording(output, 30)
        clock[0] = 31.0
        return {"stop_reason": "deadline"}

    result = asyncio.run(capture_dataset(tmp_path, "a" * 32, 30, recorder))
    assert result["capture"]["status"] == "incomplete"
    assert result["capture"]["progress"] == 0.99
    assert result["selected"]["id"] == "builtin"
