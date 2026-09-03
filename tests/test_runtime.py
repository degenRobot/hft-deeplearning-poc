import asyncio
from pathlib import Path

from starlette.requests import Request

from market_gate.api import create_app
from market_gate.config import LabConfig
from market_gate.runtime import MarketRuntime

ROOT = Path(__file__).parents[1]


def test_runtime_reset_and_concurrent_lifecycle_ownership() -> None:
    async def scenario() -> None:
        runtime = MarketRuntime(
            LabConfig(), ROOT / "models" / "gate-demo.npz", ROOT / "fixtures" / "replay.jsonl"
        )
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
        runtime = MarketRuntime(
            LabConfig(), ROOT / "models" / "gate-demo.npz", ROOT / "fixtures" / "missing.jsonl"
        )
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
