"""Thin read-only FastAPI surface over the single-owner market runtime."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from starlette.websockets import WebSocketDisconnect

from .config import load_config
from .runtime import MarketRuntime
from .training_credentials import save_credentials
from .training_options import TrainingOptions
from .training_service import TrainingService

ALLOWED_MUTATION_ORIGINS = {
    f"http://{host}:{port}" for host in ("localhost", "127.0.0.1") for port in (3000, 3010)
}
TRAINING_RECEIPT_PATH = Path(__file__).parents[2] / "artifacts" / "training-demo.json"


def require_allowed_mutation_origin(request: Request) -> None:
    """Originless local clients remain usable; browser-originated mutations are allowlisted."""
    origin = request.headers.get("origin")
    if origin is not None and origin not in ALLOWED_MUTATION_ORIGINS:
        raise HTTPException(status_code=403, detail="mutation origin is not allowed")


async def training_json(request: Request, *, allow_empty: bool = False) -> dict:
    """Small JSON requests with generic errors that never echo secret input."""
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 4096:
            raise HTTPException(status_code=413, detail="Training request is too large")
    if not body and allow_empty:
        return {}
    if request.headers.get("content-type", "").split(";")[0] != "application/json":
        raise HTTPException(status_code=415, detail="Send application/json")
    try:
        value = json.loads(body)
        if not isinstance(value, dict):
            raise ValueError
        return value
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(status_code=422, detail="Invalid training request") from None


def load_training_receipt(path: Path = TRAINING_RECEIPT_PATH) -> dict[str, object]:
    """Load the offline training receipt without triggering a recording or training run."""
    if not path.is_file():
        raise HTTPException(status_code=404, detail="training receipt is not available yet")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=503, detail="training receipt is invalid") from error
    if not isinstance(receipt, dict) or receipt.get("schema_version") != 1:
        raise HTTPException(status_code=503, detail="training receipt has an unsupported schema")
    return receipt


class ConfigPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str | None = None
    symbol: str | None = None
    gate_mode: str | None = None
    higher_level_influence: float | None = None
    gate_interval_ms: int | None = None
    expert_strength: float | None = None
    base_spread_bps: float | None = None
    max_inventory: float | None = None
    stale_after_ms: int | None = None
    flow_window_trades: int | None = None


def create_app(config_path: str | Path = "configs/demo.toml") -> FastAPI:
    config = load_config(config_path)
    root = Path(__file__).parents[2]
    runtime = MarketRuntime(
        config, root / "models" / "gate-demo.npz", root / "fixtures" / "replay.jsonl"
    )
    training_service = TrainingService(root)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await runtime.start()
        try:
            yield
        finally:
            training_service.close()
            await runtime.stop()

    app = FastAPI(title="Market Gate Lab", version="0.1.0", lifespan=lifespan)
    app.state.runtime = runtime
    app.state.training_service = training_service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=sorted(ALLOWED_MUTATION_ORIGINS),
        allow_credentials=False,
        allow_methods=["GET", "PATCH", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health")
    async def health() -> dict[str, object]:
        return runtime.snapshot()

    @app.get("/config")
    async def get_config() -> dict[str, object]:
        return runtime.config.public()

    @app.patch("/config")
    async def patch_config(patch: ConfigPatch, request: Request) -> dict[str, object]:
        require_allowed_mutation_origin(request)
        try:
            return await runtime.configure(patch.model_dump(exclude_none=True))
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/reset")
    async def reset(request: Request) -> dict[str, object]:
        require_allowed_mutation_origin(request)
        return await runtime.reset()

    @app.get("/ledger")
    async def ledger() -> list[dict[str, object]]:
        return runtime.engine.ledger_rows()

    @app.get("/training")
    async def training() -> dict[str, object]:
        return load_training_receipt(TRAINING_RECEIPT_PATH)

    @app.get("/learning")
    async def learning_status() -> dict:
        return runtime.learning.status()

    @app.patch("/learning")
    async def configure_learning(request: Request) -> dict:
        require_allowed_mutation_origin(request)
        payload = await training_json(request)
        try:
            return runtime.learning.configure(payload)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from None

    @app.get("/learning/training")
    async def learning_training() -> dict:
        return runtime.learning.training_snapshot()

    @app.get("/training/live")
    async def live_training() -> dict:
        return training_service.snapshot() | {
            "can_stop": training_service.process is not None
            and training_service.process.poll() is None
        }

    @app.get("/training/settings")
    async def training_settings() -> dict:
        return training_service.settings()

    @app.get("/training/data")
    async def public_training_data() -> dict:
        return await asyncio.to_thread(training_service.datasets.snapshot)

    @app.post("/training/data/start")
    async def start_public_capture(request: Request) -> dict:
        require_allowed_mutation_origin(request)
        payload = await training_json(request)
        if (
            set(payload) != {"seconds"}
            or type(payload["seconds"]) is not int
            or not 30 <= payload["seconds"] <= 1800
        ):
            raise HTTPException(status_code=422, detail="Choose 30 to 1800 whole capture seconds")
        try:
            return await asyncio.to_thread(training_service.datasets.start, payload["seconds"])
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from None

    @app.post("/training/data/history")
    async def start_public_history(request: Request) -> dict:
        require_allowed_mutation_origin(request)
        payload = await training_json(request)
        if set(payload) != {"symbol", "start", "end"} or any(
            not isinstance(value, str) for value in payload.values()
        ):
            raise HTTPException(status_code=400, detail="Provide symbol, start and end as strings")
        try:
            return await asyncio.to_thread(training_service.datasets.start_history, **payload)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from None

    @app.post("/training/data/stop")
    async def stop_public_capture(request: Request) -> dict:
        require_allowed_mutation_origin(request)
        await training_json(request, allow_empty=True)
        try:
            return await asyncio.to_thread(training_service.datasets.stop)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from None

    @app.post("/training/credentials")
    async def training_credentials(request: Request) -> dict:
        require_allowed_mutation_origin(request)
        payload = await training_json(request)
        try:
            status = save_credentials(training_service.env_path, payload)
            return {"modal": status}
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from None
        except OSError:
            raise HTTPException(status_code=500, detail="Could not save the project .env") from None

    @app.post("/training/live/start")
    async def start_training(request: Request) -> dict:
        require_allowed_mutation_origin(request)
        payload = await training_json(request, allow_empty=True)
        backend = payload.pop("backend", "local")
        try:
            if not isinstance(backend, str) or backend not in {"local", "modal"}:
                raise ValueError
            options = TrainingOptions(**payload).validate()
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=422, detail="Invalid model or training settings"
            ) from None
        try:
            return training_service.start(options, backend)
        except FileNotFoundError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/training/live/stop")
    async def stop_training(request: Request) -> dict:
        require_allowed_mutation_origin(request)
        try:
            return await asyncio.to_thread(training_service.stop)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.websocket("/ws/market")
    async def market_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                await websocket.send_json(runtime.snapshot())
                await asyncio.sleep(0.1)
        except WebSocketDisconnect:
            return

    return app


app = create_app()
