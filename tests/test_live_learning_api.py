"""Local control API and runtime own one cancellable in-memory learner."""

import asyncio

import numpy as np
import pytest
from conftest import ROOT, asgi_request

from market_gate.api import create_app
from market_gate.config import LabConfig
from market_gate.runtime import MarketRuntime


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    async def idle_feed(self, engine):
        engine.feed_status = "running"
        await asyncio.Event().wait()

    monkeypatch.setattr(MarketRuntime, "_run_feed", idle_feed)


def test_learning_mutation_rejects_hostile_origin_before_changing_state():
    async def scenario():
        app = create_app()
        original = app.state.runtime.learning.status()
        status, body = await asgi_request(
            app, "PATCH", "/learning", {"enabled": True}, "https://evil.example"
        )
        assert status == 403
        assert body["detail"] == "mutation origin is not allowed"
        assert app.state.runtime.learning.status() == original

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "payload",
    [
        {"enabled": "true"},
        {"reset": 1},
        {"interval_seconds": True},
        {"interval_seconds": 4},
        {"interval_seconds": 61},
        {"interval_seconds": 5.5},
        {"unknown": "/private/secret"},
        {"enabled": True, "interval_seconds": 0},
    ],
)
def test_learning_invalid_requests_are_atomic(payload):
    async def scenario():
        app = create_app()
        status, original = await asgi_request(app, "GET", "/learning")
        assert status == 200
        status, body = await asgi_request(app, "PATCH", "/learning", payload)
        assert status == 422
        assert "/private/secret" not in body["detail"]
        status, current = await asgi_request(app, "GET", "/learning")
        assert status == 200
        assert current == original

    asyncio.run(scenario())


def test_learning_request_body_size_and_content_type_guards():
    async def scenario():
        app = create_app()
        status, _ = await asgi_request(app, "PATCH", "/learning")
        assert status == 415
        status, _ = await asgi_request(app, "PATCH", "/learning", {"unknown": "x" * 5000})
        assert status == 413
        assert not app.state.runtime.learning.enabled

    asyncio.run(scenario())


def test_learning_enable_pause_reset_and_continuous_snapshot():
    async def scenario():
        app = create_app()
        async with app.router.lifespan_context(app):
            runtime = app.state.runtime
            initial = runtime.learning.status()
            status, active = await asgi_request(
                app,
                "PATCH",
                "/learning",
                {"enabled": True, "interval_seconds": 5},
                "http://localhost:3000",
            )
            assert status == 200
            assert active["enabled"] is True
            assert active["interval_seconds"] == 5
            assert active["stage"] == "warming_up"
            status, snapshot = await asgi_request(app, "GET", "/learning/training")
            assert status == 200
            assert snapshot["continuous"] is True
            assert snapshot["status"] == "running"
            assert snapshot["run_id"] == runtime.engine.run_id
            assert snapshot["backend"] == "local"
            assert snapshot["evaluation"] is None
            assert snapshot["dataset"] is None
            assert snapshot["latest"] is None
            assert snapshot["history"] == []
            assert snapshot["learning"]["enabled"] is True
            assert snapshot["can_stop"] is False  # Independent from the batch subprocess.
            assert "holdout" not in snapshot

            # An in-memory learned head must pause intact, then reset to the loaded base.
            runtime.engine.gate.w3 += 0.01
            runtime.engine.gate.model_version += "-live-rl-1"
            runtime.learning.updates = 1
            learned_hash = runtime.learning.status()["head_sha256"]
            assert learned_hash != initial["head_sha256"]
            status, paused = await asgi_request(app, "PATCH", "/learning", {"enabled": False})
            assert status == 200
            assert paused["enabled"] is False
            assert paused["stage"] == "paused"
            assert paused["head_sha256"] == learned_hash
            _, snapshot = await asgi_request(app, "GET", "/learning/training")
            assert snapshot["status"] == "stopped"
            status, reset = await asgi_request(app, "PATCH", "/learning", {"reset": True})
            assert status == 200
            assert reset["enabled"] is False
            assert reset["updates"] == 0
            assert reset["head_sha256"] == initial["head_sha256"]
            assert reset["model_version"] == initial["model_version"]
            _, snapshot = await asgi_request(app, "GET", "/learning/training")
            assert snapshot["status"] == "idle"
            assert snapshot["history"] == []
            assert snapshot["evaluation"] is None

    asyncio.run(scenario())


def test_learning_api_rejects_enable_when_gate_is_not_neural():
    async def scenario():
        app = create_app()
        async with app.router.lifespan_context(app):
            status, _ = await asgi_request(app, "PATCH", "/config", {"gate_mode": "static"})
            assert status == 200
            status, body = await asgi_request(app, "PATCH", "/learning", {"enabled": True})
            assert status == 422
            assert "neural" in body["detail"]
            assert not app.state.runtime.learning.enabled

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["reset", "configure"])
def test_runtime_restart_replaces_learner_and_restores_loaded_base(operation):
    async def scenario():
        runtime = MarketRuntime(
            LabConfig(), ROOT / "models/gate-demo.npz", ROOT / "fixtures/replay.jsonl"
        )
        await runtime.start()
        try:
            await asyncio.sleep(0)
            old_engine = runtime.engine
            old_learner = runtime.learning
            old_task = runtime.learning_task
            base_head = old_engine.gate.w3.copy()
            base_version = old_engine.gate.model_version
            old_learner.configure({"enabled": True})
            old_engine.gate.w3 += 0.01
            old_engine.gate.model_version += "-live-rl-1"
            old_learner.updates = 1
            if operation == "reset":
                await runtime.reset()
            else:
                await runtime.configure({"flow_window_trades": 5})
            assert old_task.done() and old_task.cancelled()
            assert runtime.learning is not old_learner
            assert runtime.engine is not old_engine
            assert runtime.learning.engine is runtime.engine
            assert runtime.learning_task is not old_task
            assert not runtime.learning_task.done()
            assert not runtime.learning.enabled
            assert runtime.learning.updates == 0
            assert runtime.learning.pending is None
            assert runtime.engine.gate.model_version == base_version
            np.testing.assert_array_equal(runtime.engine.gate.w3, base_head)
        finally:
            latest = runtime.learning_task
            await runtime.stop()
            assert latest.done() and latest.cancelled()
            assert runtime.learning_task is None
            assert runtime.feed_task is None

    asyncio.run(scenario())


def test_concurrent_restarts_keep_one_learning_owner_and_stop_drains_it(monkeypatch):
    original = MarketRuntime._run_learning
    active = set()
    max_active = 0
    exited = []

    async def observed(self, learner):
        nonlocal max_active
        active.add(id(learner))
        max_active = max(max_active, len(active))
        try:
            await original(self, learner)
        finally:
            active.remove(id(learner))
            exited.append(id(learner))

    monkeypatch.setattr(MarketRuntime, "_run_learning", observed)

    async def scenario():
        runtime = MarketRuntime(
            LabConfig(), ROOT / "models/gate-demo.npz", ROOT / "fixtures/replay.jsonl"
        )
        await runtime.start()
        await asyncio.sleep(0)
        first = runtime.learning_task
        try:
            await asyncio.gather(
                runtime.reset(), runtime.configure({"flow_window_trades": 5}), runtime.reset()
            )
            await asyncio.sleep(0)
            assert first.done() and first.cancelled()
            assert max_active == 1
            assert active == {id(runtime.learning)}
            assert runtime.learning_task is not None and not runtime.learning_task.done()
            assert runtime.feed_generation == 4
        finally:
            task = runtime.learning_task
            learner = runtime.learning
            await runtime.stop()
            assert task.cancelled()
            assert runtime.learning_task is None
            assert id(learner) in exited
            assert not active
            await runtime.stop()  # Idempotent teardown leaves no learner task behind.

    asyncio.run(scenario())
