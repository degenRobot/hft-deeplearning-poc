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
    with pytest.raises(FileNotFoundError, match="Record public data"):
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
