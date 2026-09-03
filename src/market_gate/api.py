"""Thin read-only FastAPI surface over the single-owner market runtime."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from starlette.websockets import WebSocketDisconnect

from .config import load_config
from .runtime import MarketRuntime


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

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await runtime.start()
        try:
            yield
        finally:
            await runtime.stop()

    app = FastAPI(title="Market Gate Lab", version="0.1.0", lifespan=lifespan)
    app.state.runtime = runtime
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=False,
        allow_methods=["GET", "PATCH", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health")
    async def health() -> dict[str, object]:
        return runtime.engine.snapshot()

    @app.get("/config")
    async def get_config() -> dict[str, object]:
        return runtime.config.public()

    @app.patch("/config")
    async def patch_config(patch: ConfigPatch) -> dict[str, object]:
        try:
            return await runtime.configure(patch.model_dump(exclude_none=True))
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/reset")
    async def reset() -> dict[str, object]:
        return await runtime.reset()

    @app.get("/ledger")
    async def ledger() -> list[dict[str, object]]:
        return runtime.engine.ledger_rows()

    @app.websocket("/ws/market")
    async def market_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                await websocket.send_json(runtime.engine.snapshot())
                await asyncio.sleep(0.1)
        except WebSocketDisconnect:
            return

    return app


app = create_app()
