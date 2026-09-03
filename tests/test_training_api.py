import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path

from market_gate import api
from market_gate.api import create_app

ROOT = Path(__file__).parents[1]


async def request(app: Callable[..., Awaitable[None]], path: str) -> tuple[int, dict[str, object]]:
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"host", b"testserver")],
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
    }
    sent: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    await app(scope, receive, send)
    status = next(message["status"] for message in sent if message["type"] == "http.response.start")
    body = b"".join(
        message.get("body", b"") for message in sent if message["type"] == "http.response.body"
    )
    return int(status), json.loads(body)


def test_training_endpoint_returns_committed_receipt(monkeypatch, tmp_path: Path) -> None:
    receipt = {
        "schema_version": 1,
        "generated_at": "2026-09-04T00:00:00Z",
        "source": {},
        "dataset": {},
        "training": {},
        "limitations": [],
    }
    path = tmp_path / "training-demo.json"
    path.write_text(json.dumps(receipt))
    monkeypatch.setattr(api, "TRAINING_RECEIPT_PATH", path)

    status, body = asyncio.run(request(create_app(), "/training"))
    assert status == 200
    assert body == receipt


def test_training_endpoint_is_read_only_when_receipt_is_missing(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(api, "TRAINING_RECEIPT_PATH", tmp_path / "missing.json")
    status, body = asyncio.run(request(create_app(), "/training"))
    assert status == 404
    assert body["detail"] == "training receipt is not available yet"


def test_checked_in_training_receipt_matches_the_public_contract() -> None:
    receipt = json.loads((ROOT / "artifacts" / "training-demo.json").read_text())

    assert set(receipt) == {
        "schema_version",
        "generated_at",
        "source",
        "dataset",
        "training",
        "limitations",
    }
    assert receipt["source"]["event_counts"]["total"] == 1_377
    assert receipt["dataset"]["examples"] == (
        receipt["dataset"]["train_examples"] + receipt["dataset"]["validation_examples"]
    )
    assert receipt["training"]["model_path"] == "models/gate-binance-demo.npz"
