import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from conftest import asgi_request

from market_gate.api import create_app
from market_gate.training_progress import MODAL_PRICING, training_progress
from market_gate.training_service import SnapshotWriter


def test_real_step_progress_waits_for_artifacts_and_freezes_at_completion(tmp_path):
    writer = SnapshotWriter(tmp_path / "run.json", "run", "modal")
    assert writer.state["progress"]["percent"] is None
    writer.accept({"kind": "dataset", "dataset": {"epochs": 2, "rl_examples": 15}})
    writer.accept({"kind": "step", "step": {"step": 1, "phase": "supervised"}})
    assert writer.state["progress"]["percent"] == 5
    writer.accept({"kind": "step", "step": {"step": 17, "phase": "rl"}})
    assert writer.state["progress"]["percent"] == 99
    assert writer.state["progress"]["stage"] == "finalizing"
    writer.accept({"kind": "completed", "evaluation": {}})
    completed = writer.state["progress"].copy()
    assert completed["percent"] == 100
    writer.publish()
    assert writer.state["progress"] == completed
    writer.close()


def test_estimate_counts_only_observed_remote_time_at_public_rates():
    now = datetime.now(UTC)
    state = dict(status="running", started_at=(now - timedelta(seconds=90)).isoformat())
    assert training_progress(state, now)["compute_estimate_usd"] is None
    state["execution_started_at"] = (now - timedelta(seconds=60)).isoformat()
    result = training_progress(state, now)
    assert result["remote_elapsed_seconds"] == 60
    assert result["elapsed_seconds"] == 90
    assert result["compute_estimate_usd"] == pytest.approx(0.0018384)
    assert MODAL_PRICING["source_url"] == "https://modal.com/pricing"


@pytest.mark.parametrize(
    "payload",
    [{"seconds": True}, {"seconds": 29}, {"seconds": 1801}, {"seconds": 30, "path": "/tmp/x"}],
)
def test_capture_api_rejects_invalid_requests_before_launch(monkeypatch, payload):
    app = create_app()
    monkeypatch.setattr(
        app.state.training_service.datasets, "start", lambda *_: pytest.fail("started")
    )
    assert asyncio.run(asgi_request(app, "POST", "/training/data/start", payload))[0] == 422


def test_capture_api_allows_bounded_request_and_blocks_foreign_origin(monkeypatch):
    app = create_app()
    calls = []
    monkeypatch.setattr(
        app.state.training_service.datasets,
        "start",
        lambda seconds: calls.append(seconds) or {"ok": True},
    )
    assert (
        asyncio.run(
            asgi_request(
                app, "POST", "/training/data/start", {"seconds": 30}, origin="https://evil.example"
            )
        )[0]
        == 403
    )
    assert calls == []
    assert (
        asyncio.run(
            asgi_request(
                app, "POST", "/training/data/start", {"seconds": 30}, origin="http://localhost:3010"
            )
        )[0]
        == 200
    )
    assert calls == [30]
