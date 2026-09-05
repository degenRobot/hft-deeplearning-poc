import json

import pytest

from market_gate.training_service import SnapshotWriter, TrainingService


def test_single_writer_and_dead_worker_detection(tmp_path):
    service = TrainingService(tmp_path)
    assert service.snapshot()["status"] == "idle"
    writer = SnapshotWriter(service.path, "first", "modal")
    assert service.snapshot()["status"] == "running"
    with pytest.raises(ValueError, match="another training"):
        SnapshotWriter(service.path, "second", "local")
    with pytest.raises(ValueError, match="another training"):
        service.start()
    writer.close()
    assert service.snapshot()["status"] == "failed"
    assert "exited" in service.snapshot()["error"]


def test_snapshot_completion_and_finite_serialization(tmp_path):
    service = TrainingService(tmp_path)
    writer = SnapshotWriter(service.path, "first", "local")
    writer.accept({"kind": "dataset", "dataset": {"symbol": "BTCUSDT"}})
    writer.accept({"kind": "completed", "evaluation": {"adapted": 0.1}})
    writer.close()
    result = service.snapshot()
    assert result["status"] == "completed"
    assert result["evaluation"] == {"adapted": 0.1}
    assert json.loads(service.path.read_text()) == result
    with pytest.raises(ValueError, match="no local run"):
        service.stop()


def test_missing_dataset_does_not_start_process(tmp_path):
    service = TrainingService(tmp_path)
    with pytest.raises(FileNotFoundError, match="Selected public recording is unavailable"):
        service.start()
    assert service.process is None


def test_failed_write_preserves_last_valid_snapshot(tmp_path):
    writer = SnapshotWriter(tmp_path / "live.json", "first", "local")
    previous = writer.path.read_bytes()
    writer.state["invalid"] = float("nan")
    with pytest.raises(ValueError):
        writer.publish()
    assert writer.path.read_bytes() == previous
    writer.close()


def test_training_api_rejects_hostile_origin_without_starting(monkeypatch):
    import asyncio

    from conftest import asgi_request

    from market_gate.api import create_app

    app = create_app()

    def unexpected():
        raise AssertionError("hostile request started a worker")

    monkeypatch.setattr(app.state.training_service, "start", unexpected)
    status, _ = asyncio.run(
        asgi_request(app, "POST", "/training/live/start", origin="https://evil.example")
    )
    assert status == 403


def test_new_run_does_not_show_previous_completed_snapshot(tmp_path, monkeypatch):
    service = TrainingService(tmp_path)
    writer = SnapshotWriter(service.path, "old", "local")
    writer.finish("completed")
    writer.close()

    (tmp_path / "data").mkdir()
    (tmp_path / "data/training-public.jsonl").write_text("public recording")

    class StartingProcess:
        def __init__(self, *args, **kwargs):
            pass

        def poll(self):
            return None

    monkeypatch.setattr("market_gate.training_service.subprocess.Popen", StartingProcess)
    started = service.start()
    assert started["run_id"] != "old"
    assert service.snapshot() == started
    with pytest.raises(ValueError, match="another training"):
        service.start()
    writer = SnapshotWriter(service.path, started["run_id"], "local")
    assert service.snapshot()["run_id"] == started["run_id"]
    assert service.pending is None
    writer.close()


def test_modal_start_forwards_validated_options_without_credentials_in_arguments(
    tmp_path, monkeypatch
):
    from market_gate.training_options import TrainingOptions

    service = TrainingService(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/training-public.jsonl").write_text("public data")
    commands = []

    class Process:
        def __init__(self, command, **kwargs):
            commands.append(command)

        def poll(self):
            return None

    monkeypatch.setattr(
        "market_gate.training_service.credential_status",
        lambda _: {"configured": True, "available": True},
    )
    monkeypatch.setattr("market_gate.training_service.subprocess.Popen", Process)
    result = service.start(TrainingOptions(hidden_1=1024, hidden_2=512, epochs=2), "modal")
    assert result["backend"] == "modal"
    command = commands[0]
    assert "--run" in command and "--env-file" in command
    assert command[command.index("--hidden-1") + 1] == "1024"
    assert command[command.index("--hidden-2") + 1] == "512"
    assert not any("TOKEN_SECRET" in value for value in command)
