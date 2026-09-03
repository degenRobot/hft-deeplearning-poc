"""One-owner lifecycle for the engine and its sole public/replay feed task."""

import asyncio
import time
import uuid
from pathlib import Path

from .config import LabConfig
from .engine import MarketEngine
from .feeds.binance import BinancePublicFeed
from .feeds.replay import ReplayFeed, replay_schedule


class MarketRuntime:
    """Serializes stop/configure/start so a restart cannot leave an orphan feed task."""

    def __init__(self, config: LabConfig, model_path: str | Path, fixture_path: str | Path) -> None:
        self.config = config
        self.model_path = Path(model_path)
        self.fixture_path = Path(fixture_path)
        self.engine = MarketEngine(config, self.model_path)
        self.feed_task: asyncio.Task[None] | None = None
        self.feed_generation = 0
        self._lock = asyncio.Lock()

    @property
    def active_feed_tasks(self) -> int:
        return int(self.feed_task is not None and not self.feed_task.done())

    async def start(self) -> None:
        async with self._lock:
            await self._restart_locked()

    async def stop(self) -> None:
        async with self._lock:
            await self._stop_locked()

    async def configure(self, values: dict[str, object]) -> dict[str, object]:
        async with self._lock:
            self.config.patch(values)
            await self._restart_locked()
            return self.config.public()

    async def reset(self) -> dict[str, object]:
        async with self._lock:
            await self._restart_locked()
            return {"run_id": self.engine.run_id, "feed_generation": self.feed_generation}

    async def _stop_locked(self) -> None:
        if self.feed_task is not None:
            self.feed_task.cancel()
            try:
                await self.feed_task
            except asyncio.CancelledError:
                pass
            self.feed_task = None

    async def _restart_locked(self) -> None:
        # Await cancellation before replacing the engine: the old task must never write new state.
        await self._stop_locked()
        self.feed_generation += 1
        self.engine = MarketEngine(self.config, self.model_path)
        self.engine.run_id = uuid.uuid4().hex[:12]
        self.engine.feed_generation = self.feed_generation
        self.feed_task = asyncio.create_task(self._run_feed(self.engine))

    async def _run_feed(self, engine: MarketEngine) -> None:
        if self.config.source == "replay":
            engine.feed_status = "replay_loading"
            events = list(ReplayFeed(self.fixture_path))
            for scheduled in replay_schedule(events, int(time.time() * 1000), cycles=None):
                if scheduled.delay_ms:
                    await asyncio.sleep(scheduled.delay_ms / 1_000)
                engine.process(scheduled.event, arrival_ts_ms=int(time.time() * 1000))
                engine.feed_status = "running"
        else:

            def update_feed_status(status: str) -> None:
                engine.feed_status = status
                engine.reconnects = feed.reconnects

            feed = BinancePublicFeed(self.config.symbol, update_feed_status)
            async for event in feed.events():
                engine.reconnects = feed.reconnects
                engine.process(event)
