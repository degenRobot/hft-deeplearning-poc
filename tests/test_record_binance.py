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
