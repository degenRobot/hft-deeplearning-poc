"""Offline cloud-package checks: no credentials, imports or compute on Modal."""

import importlib.util
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace


def test_exact_modal_source_package_imports_without_web_runtime(tmp_path):
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location(
        "training_cloud", root / "scripts/train_lab_on_modal.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    package = tmp_path / "market_gate"
    package.mkdir()
    for name in module.SOURCE_FILES:
        (package / name).write_bytes((root / "src/market_gate" / name).read_bytes())
    script = (
        "import sys; from market_gate.training_lab import train_lab; "
        "assert 'market_gate.api' not in sys.modules; "
        "assert 'fastapi' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=os.environ | {"PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert set(p.name for p in package.iterdir() if p.is_file()) == set(module.SOURCE_FILES)


def test_sigterm_cancels_remote_call_despite_modal_keyboard_interrupt_suppression(
    tmp_path, monkeypatch
):
    """Exercise main's real signal and snapshot flow without credentials or cloud calls."""
    from market_gate import training, training_lab

    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location(
        "training_cloud_cancel", root / "scripts/train_lab_on_modal.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sources = tmp_path / "src/market_gate"
    sources.mkdir(parents=True)
    for name in module.SOURCE_FILES:
        (sources / name).write_bytes((root / "src/market_gate" / name).read_bytes())
    recording = tmp_path / "recording.jsonl"
    recording.write_text("public fixture\n")
    output = tmp_path / "artifacts/run"
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_lab_on_modal.py", "--run", "--input", str(recording), "--output", str(output)],
    )
    monkeypatch.setattr(module, "load_credentials", lambda path: None)
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda command, cwd: (sources / command[-1].split("/")[-1]).read_bytes(),
    )
    monkeypatch.setattr(training, "load_recording", lambda path: [])
    monkeypatch.setattr(
        training,
        "build_frame_dataset",
        lambda events: SimpleNamespace(values=[], mids=[], close_ts_ms=[]),
    )
    monkeypatch.setattr(training, "build_examples", lambda *args, **kwargs: [])
    monkeypatch.setattr(training_lab, "split_lab_examples", lambda *args: None)
    handlers = {}
    monkeypatch.setattr(
        module.signal, "signal", lambda number, handler: handlers.update({number: handler})
    )
    observed = {}

    class Image:
        @classmethod
        def debian_slim(cls, **kwargs):
            return cls()

        def pip_install(self, *args):
            return self

        def env(self, values):
            return self

        def add_local_file(self, *args):
            return self

    class ModalRun:
        def __enter__(self):
            return self

        def __exit__(self, exception_type, exception, traceback):
            # Modal 1.5.2's app.run() swallows KeyboardInterrupt. The signal must
            # propagate through this context so main can cancel the remote call.
            observed["context_exception"] = exception_type
            return exception_type is KeyboardInterrupt

    class Remote:
        def remote_gen(self, *args):
            yield {"kind": "execution", "function_call_id": "fc-offline-test"}
            handlers[module.signal.SIGTERM](module.signal.SIGTERM, None)
            raise AssertionError("SIGTERM did not interrupt remote iteration")

    class App:
        def __init__(self, *args, **kwargs):
            pass

        def function(self, **kwargs):
            return lambda function: Remote()

        def run(self):
            return ModalRun()

    class FunctionCall:
        @classmethod
        def from_id(cls, call_id):
            observed["cancelled_call"] = call_id
            return cls()

        def cancel(self, *, terminate_containers):
            observed["terminate_containers"] = terminate_containers

    monkeypatch.setitem(
        sys.modules,
        "modal",
        SimpleNamespace(Image=Image, App=App, FunctionCall=FunctionCall, enable_output=nullcontext),
    )
    module.main()

    assert observed == {
        "context_exception": module.TrainingCancelled,
        "cancelled_call": "fc-offline-test",
        "terminate_containers": True,
    }
    snapshot = json.loads((tmp_path / "artifacts/training-live.json").read_text())
    assert snapshot["status"] == "stopped"
    assert snapshot["backend"] == "modal"
    assert snapshot["execution"] == {"function_call_id": "fc-offline-test"}
    assert not (tmp_path / ".env").exists()
    assert not (output / "modal-execution.json").exists()
