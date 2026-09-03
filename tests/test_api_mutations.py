import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path

from market_gate.api import create_app

ROOT = Path(__file__).parents[1]


async def asgi_request(
    app: Callable[..., Awaitable[None]],
    method: str,
    path: str,
    body: dict[str, object] | None = None,
    origin: str | None = None,
) -> tuple[int, dict[str, object]]:
    raw_body = json.dumps(body).encode() if body is not None else b""
    headers = [(b"host", b"testserver")]
    if body is not None:
        headers.extend(
            [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(raw_body)).encode()),
            ]
        )
    if origin is not None:
        headers.append((b"origin", origin.encode()))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
    }
    sent: list[dict[str, object]] = []
    delivered = False

    async def receive() -> dict[str, object]:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": raw_body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    await app(scope, receive, send)
    status = next(message["status"] for message in sent if message["type"] == "http.response.start")
    response = b"".join(
        message.get("body", b"") for message in sent if message["type"] == "http.response.body"
    )
    return int(status), json.loads(response)


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
