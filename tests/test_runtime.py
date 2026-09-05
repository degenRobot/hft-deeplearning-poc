import asyncio
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.requests import Request

from market_gate.api import create_app
from market_gate.config import LabConfig
from market_gate.runtime import MarketRuntime

ROOT = Path(__file__).parents[1]
runtime_for = partial(MarketRuntime, LabConfig(), ROOT / "models" / "gate-demo.npz")


def test_replay_deadlines_do_not_accumulate_sleep_and_processing_drift(monkeypatch):
    from market_gate import runtime as runtime_module
    from market_gate.contracts import BookEvent
    from market_gate.feeds.replay import replay_schedule

    clock = [1_000.0]
    events = [BookEvent("replay", "BTCUSDT", 10, 10, 1, 99, 1, 101, 1)]
    market = MarketRuntime(
        LabConfig(gate_mode="uniform"),
        ROOT / "models/gate-demo.npz",
        ROOT / "fixtures/replay.jsonl",
    )
    monkeypatch.setattr(runtime_module, "ReplayFeed", lambda _: events)
    monkeypatch.setattr(
        runtime_module,
        "replay_schedule",
        lambda rows, anchor, cycles: replay_schedule(rows, anchor, 200),
    )
    monkeypatch.setattr(
        runtime_module,
        "time",
        SimpleNamespace(time=lambda: clock[0], monotonic=lambda: clock[0]),
    )

    async def oversleep(seconds):
        assert seconds > 0
        clock[0] += seconds + 0.025

    monkeypatch.setattr(runtime_module.asyncio, "sleep", oversleep)
    process = market.engine.process
    lateness = []

    def process_with_work(event, arrival_ts_ms):
        lateness.append(arrival_ts_ms - event.event_ts_ms)
        process(event, arrival_ts_ms=arrival_ts_ms)
        clock[0] += 0.010

    monkeypatch.setattr(market.engine, "process", process_with_work)
    asyncio.run(market._run_feed(market.engine))
    assert len(lateness) == 200
    # The old relative sleeps accumulate nearly seven seconds of artificial age.
    assert max(lateness) <= 26
    snapshot = market.engine.snapshot(now_ms=int(clock[0] * 1000))
    assert snapshot["health"]["ready"] is True
    assert snapshot["health"]["book_age_ms"] <= 36


def test_runtime_reset_and_concurrent_lifecycle_ownership() -> None:
    async def scenario() -> None:
        runtime = runtime_for(ROOT / "fixtures" / "replay.jsonl")
        await runtime.start()
        assert runtime.feed_generation == 1
        assert runtime.engine.feed_generation == 1
        assert runtime.active_feed_tasks == 1
        reset = await runtime.reset()
        assert reset["feed_generation"] == 2
        await asyncio.gather(runtime.configure({"flow_window_trades": 5}), runtime.reset())
        assert runtime.feed_generation == 4
        assert runtime.engine.feed_generation == 4
        assert runtime.active_feed_tasks == 1
        assert runtime.engine.recent_trades.maxlen == 5
        await runtime.stop()
        assert runtime.active_feed_tasks == 0

    asyncio.run(scenario())


def test_post_reset_returns_current_run_and_generation() -> None:
    async def scenario() -> None:
        app = create_app(ROOT / "configs" / "demo.toml")
        reset_endpoint = next(route.endpoint for route in app.routes if route.path == "/reset")
        async with app.router.lifespan_context(app):
            result = await reset_endpoint(Request({"type": "http", "headers": []}))
            assert result["feed_generation"] == 2
            assert result["run_id"] == app.state.runtime.engine.run_id
            assert app.state.runtime.active_feed_tasks == 1

    asyncio.run(scenario())


def test_failed_feed_is_terminal_then_reset_and_stop_recover_cleanly() -> None:
    async def scenario() -> None:
        runtime = runtime_for(ROOT / "fixtures" / "missing.jsonl")
        await runtime.start()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert runtime.engine.feed_status == "failed"
        assert runtime.feed_error == "FileNotFoundError"
        assert runtime.snapshot()["health"]["feed_error"] == "FileNotFoundError"
        assert runtime.active_feed_tasks == 0
        assert runtime.feed_task is None

        runtime.fixture_path = ROOT / "fixtures" / "replay.jsonl"
        reset = await runtime.reset()
        assert reset["feed_generation"] == 2
        assert runtime.active_feed_tasks == 1
        await runtime.stop()
        assert runtime.active_feed_tasks == 0

    asyncio.run(scenario())


def test_feed_that_returns_normally_is_reported_as_failed(tmp_path: Path) -> None:
    async def scenario() -> None:
        empty_fixture = tmp_path / "empty.jsonl"
        empty_fixture.write_text("")
        runtime = runtime_for(empty_fixture)
        await runtime.start()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert runtime.engine.feed_status == "failed"
        assert runtime.feed_error == "FeedEndedError"
        assert runtime.feed_task is None
        assert runtime.active_feed_tasks == 0
        reset = await runtime.reset()
        assert reset["feed_generation"] == 2
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert runtime.engine.feed_status == "failed"
        assert runtime.feed_error == "FeedEndedError"
        assert runtime.feed_task is None
        await runtime.stop()

    asyncio.run(scenario())


def test_configure_returns_run_identity_and_invalid_patch_preserves_active_run() -> None:
    async def scenario() -> None:
        runtime = MarketRuntime(
            LabConfig(), ROOT / "models" / "gate-demo.npz", ROOT / "fixtures" / "replay.jsonl"
        )
        await runtime.start()
        try:
            receipt = await runtime.configure({"gate_mode": "uniform"})
            assert receipt["run_id"] == runtime.engine.run_id
            assert receipt["feed_generation"] == runtime.feed_generation
            assert receipt["gate_mode"] == "uniform"
            assert "run_id" not in runtime.config.public()
            previous_engine, previous_task = runtime.engine, runtime.feed_task
            with pytest.raises(ValueError, match="BTCUSDT"):
                await runtime.configure({"symbol": "ETHUSDT", "gate_mode": "static"})
            assert runtime.engine is previous_engine
            assert runtime.feed_task is previous_task
            assert runtime.feed_generation == receipt["feed_generation"]
            assert runtime.config.gate_mode == "uniform"
        finally:
            await runtime.stop()

    asyncio.run(scenario())
