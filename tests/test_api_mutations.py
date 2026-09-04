import asyncio

from conftest import ROOT, asgi_request

from market_gate.api import create_app


def test_mutation_origin_guard_blocks_hostile_and_allows_local_or_cli() -> None:
    async def scenario() -> None:
        app = create_app(ROOT / "configs" / "demo.toml")
        async with app.router.lifespan_context(app):
            runtime = app.state.runtime
            status, _ = await asgi_request(
                app, "PATCH", "/config", {"flow_window_trades": 5}, "https://evil.example"
            )
            assert status == 403
            assert runtime.feed_generation == 1
            assert runtime.config.flow_window_trades == 64

            status, body = await asgi_request(
                app, "PATCH", "/config", {"flow_window_trades": 5}, "http://localhost:3000"
            )
            assert status == 200
            assert body["flow_window_trades"] == 5
            assert runtime.feed_generation == 2

            status, _ = await asgi_request(app, "POST", "/reset", origin="https://evil.example")
            assert status == 403
            assert runtime.feed_generation == 2

            status, body = await asgi_request(app, "POST", "/reset")
            assert status == 200
            assert body["feed_generation"] == 3

    asyncio.run(scenario())
