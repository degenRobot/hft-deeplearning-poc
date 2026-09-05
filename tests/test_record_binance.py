import importlib.util
import sys
from pathlib import Path

import pytest

from market_gate.training import write_recording


def test_recorder_refuses_existing_output_before_connecting(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "scripts/record_binance.py"
    spec = importlib.util.spec_from_file_location("record_binance_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "sample.jsonl"
    output.write_text("preserve this recording\n")
    monkeypatch.setattr(sys, "argv", [str(path), "--output", str(output)])

    def unexpected_connection(*args):
        pytest.fail("existing output must be rejected before network collection")

    monkeypatch.setattr(module, "collect_events", unexpected_connection)
    with pytest.raises(SystemExit, match="new file"):
        module.main()
    assert output.read_text() == "preserve this recording\n"


def test_recording_writer_refuses_overwrite_even_after_preflight(tmp_path):
    output = tmp_path / "sample.jsonl"
    output.write_text("created by another recorder\n")
    with pytest.raises(FileExistsError):
        write_recording(output, [], 1000)
    assert output.read_text() == "created by another recorder\n"


def _recorder_module():
    path = Path(__file__).parents[1] / "scripts/record_binance.py"
    spec = importlib.util.spec_from_file_location("record_binance_stream_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_feed(monkeypatch, module, events, *, silent=False):
    import asyncio

    class Feed:
        reconnects = 2
        status = "running"

        def __init__(self, symbol):
            assert symbol == "BTCUSDT"

        async def events(self):
            if silent:
                await asyncio.sleep(60)
            for event in events:
                yield event

    monkeypatch.setattr(module, "BinancePublicFeed", Feed)


def test_streamed_recording_caps_bytes_without_partial_json(tmp_path, monkeypatch):
    import asyncio
    import json

    from market_gate.contracts import BookEvent
    from market_gate.training import event_to_record, load_recording

    module = _recorder_module()
    event = BookEvent("binance", "BTCUSDT", 1000, 1000, 1, 100.0, 1.0, 101.0, 2.0)
    second = BookEvent("binance", "BTCUSDT", 2000, 2000, 2, 100.0, 1.0, 101.0, 2.0)
    line_bytes = len(
        (json.dumps(event_to_record(event), sort_keys=True, separators=(",", ":")) + "\n").encode()
    )
    _fake_feed(monkeypatch, module, [event, second])
    output = tmp_path / "bounded.jsonl"
    result = asyncio.run(module.record_events(output, "BTCUSDT", 10, 100, 100, line_bytes + 1))
    assert result["stop_reason"] == "max_bytes"
    assert result["bytes"] == output.stat().st_size == line_bytes
    assert result["reconnects"] == 2
    assert load_recording(output) == [event]


def test_streamed_recording_downsamples_books_preserving_trades(tmp_path, monkeypatch):
    import asyncio

    from market_gate.contracts import BookEvent, TradeEvent
    from market_gate.training import load_recording

    module = _recorder_module()
    first = BookEvent("binance", "BTCUSDT", 1000, 1000, 1, 100.0, 1.0, 101.0, 2.0)
    omitted = BookEvent("binance", "BTCUSDT", 1050, 1050, 2, 100.0, 1.0, 101.0, 2.0)
    trade = TradeEvent("binance", "BTCUSDT", 1050, 1050, 1, 100.0, 1.0, "buy")
    last = BookEvent("binance", "BTCUSDT", 1100, 1100, 3, 100.0, 1.0, 101.0, 2.0)
    _fake_feed(monkeypatch, module, [first, omitted, trade, last, last])
    output = tmp_path / "bounded.jsonl"
    result = asyncio.run(module.record_events(output, "BTCUSDT", 10, 3, 100, 10000))
    assert result["stop_reason"] == "max_events"
    assert (result["book"], result["trade"], result["total"]) == (2, 1, 3)
    assert load_recording(output) == [first, trade, last]


def test_streamed_recording_deadline_stops_silent_feed(tmp_path, monkeypatch):
    import asyncio
    import time

    module = _recorder_module()
    _fake_feed(monkeypatch, module, [], silent=True)
    output = tmp_path / "empty.jsonl"
    started = time.monotonic()
    result = asyncio.run(module.record_events(output, "BTCUSDT", 0.02, 100, 100, 10000))
    assert time.monotonic() - started < 1
    assert result["stop_reason"] == "deadline"
    assert result["total"] == result["bytes"] == 0
    assert output.read_bytes() == b""


def test_streamed_recording_refuses_existing_output(tmp_path, monkeypatch):
    import asyncio

    module = _recorder_module()
    _fake_feed(monkeypatch, module, [])
    output = tmp_path / "existing.jsonl"
    output.write_text("preserve\n")
    with pytest.raises(FileExistsError):
        asyncio.run(module.record_events(output, "BTCUSDT", 10, 100, 100, 10000))
    assert output.read_text() == "preserve\n"


@pytest.mark.parametrize("flag,value", [("--seconds", "nan"), ("--progress-seconds", "inf")])
def test_recorder_rejects_nonfinite_bounds_before_connecting(tmp_path, monkeypatch, flag, value):
    module = _recorder_module()
    output = tmp_path / "bounded.jsonl"
    monkeypatch.setattr(sys, "argv", ["record_binance.py", "--output", str(output), flag, value])
    with pytest.raises(SystemExit, match="must be positive"):
        module.main()
    assert not output.exists()
