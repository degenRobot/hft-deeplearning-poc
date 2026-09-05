import json
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path

ROOT = Path(__file__).parents[1]


async def asgi_request(
    app: Callable[..., Awaitable[None]],
    method: str,
    path: str,
    body: Mapping[str, object] | None = None,
    origin: str | None = None,
) -> tuple[int, dict[str, object]]:
    raw_body = json.dumps(body).encode() if body is not None else b""
    headers = [(b"host", b"testserver")]
    if body is not None:
        headers += [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(raw_body)).encode()),
        ]
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
        if delivered:
            return {"type": "http.disconnect"}
        delivered = True
        return {"type": "http.request", "body": raw_body, "more_body": False}

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    await app(scope, receive, send)
    status = next(message["status"] for message in sent if message["type"] == "http.response.start")
    response = b"".join(
        message.get("body", b"") for message in sent if message["type"] == "http.response.body"
    )
    return int(status), json.loads(response)
