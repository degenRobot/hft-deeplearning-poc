"""Read-only FastAPI surface for configuration, health, and market snapshots."""

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect

from .config import load_config
from .engine import MarketEngine
from .feeds.binance import BinancePublicFeed
from .feeds.replay import ReplayFeed, replay_schedule


class ConfigPatch(BaseModel):
    source: str | None = None
    symbol: str | None = None
    gate_mode: str | None = None
    higher_level_influence: float | None = None
    gate_interval_ms: int | None = None
    expert_strength: float | None = None
    base_spread_bps: float | None = None
    max_inventory: float | None = None
    stale_after_ms: int | None = None


def create_app(config_path: str | Path = "configs/demo.toml") -> FastAPI:
    config = load_config(config_path)
    root = Path(__file__).parents[2]
    model_path = root / "models" / "gate-demo.npz"
    engine = MarketEngine(config, model_path)
    fixture = root / "fixtures" / "replay.jsonl"
    feed_task: asyncio.Task[None] | None = None

    def fresh_engine() -> MarketEngine:
        replacement = MarketEngine(config, model_path)
        replacement.run_id = uuid.uuid4().hex[:12]
        return replacement

    async def stop_feed() -> None:
        nonlocal feed_task
        if feed_task is not None:
            feed_task.cancel()
            try:
                await feed_task
            except asyncio.CancelledError:
                pass
            feed_task = None

    def start_feed() -> None:
        nonlocal feed_task
        feed_task = asyncio.create_task(run_feed())

    async def run_feed() -> None:
        """Continuously own one public/replay feed; config changes replace this task."""
        nonlocal engine
        if config.source == "replay":
            engine.feed_status = "replay_loading"
            events = list(ReplayFeed(fixture))
            for scheduled in replay_schedule(events, int(time.time() * 1000), cycles=None):
                if config.source != "replay":
                    return
                if scheduled.delay_ms:
                    await asyncio.sleep(scheduled.delay_ms / 1_000)
                engine.process(scheduled.event, arrival_ts_ms=int(time.time() * 1000))
                engine.feed_status = "running"
        else:

            def update_feed_status(status: str) -> None:
                engine.feed_status = status
                engine.reconnects = feed.reconnects

            feed = BinancePublicFeed(config.symbol, update_feed_status)
            async for event in feed.events():
                if config.source != "binance":
                    return
                engine.reconnects = feed.reconnects
                engine.process(event)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        nonlocal engine, feed_task
        await stop_feed()
        engine = fresh_engine()
        start_feed()
        try:
            yield
        finally:
            if feed_task is not None:
                feed_task.cancel()
                try:
                    await feed_task
                except asyncio.CancelledError:
                    pass

    app = FastAPI(title="Market Gate Lab", version="0.1.0", lifespan=lifespan)
    # The companion Next development server is a separate origin during local work.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=False,
        allow_methods=["GET", "PATCH"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health")
    def health() -> dict[str, object]:
        return engine.snapshot()

    @app.get("/config")
    def get_config() -> dict[str, object]:
        return config.public()

    @app.patch("/config")
    async def patch_config(patch: ConfigPatch) -> dict[str, object]:
        nonlocal engine, feed_task
        values = patch.model_dump(exclude_none=True)
        try:
            config.patch(values)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        # Replacing the engine makes a source or symbol change explicit and avoids mixing feeds.
        await stop_feed()
        engine = fresh_engine()
        start_feed()
        return config.public()

    @app.get("/ledger")
    def ledger() -> list[dict[str, object]]:
        return engine.ledger_rows()

    @app.websocket("/ws/market")
    async def market_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                await websocket.send_json(engine.snapshot())
                await asyncio.sleep(0.1)
        except WebSocketDisconnect:
            return

    return app


app = create_app()
