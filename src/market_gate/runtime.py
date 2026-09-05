"""One-owner lifecycle for the engine and its sole public/replay feed task."""

import asyncio
import time
import uuid
from pathlib import Path

from .config import LabConfig
from .engine import MarketEngine
from .feeds.binance import BinancePublicFeed
from .feeds.replay import ReplayFeed, replay_schedule
from .live_learning import LiveLearner


class FeedEndedError(RuntimeError):
    """A producer returned without cancellation even though feeds are continuous."""


class MarketRuntime:
    """Serializes stop/configure/start so a restart cannot leave an orphan feed task."""

    def __init__(self, config: LabConfig, model_path: str | Path, fixture_path: str | Path) -> None:
        self.config = config
        self.model_path = Path(model_path)
        self.fixture_path = Path(fixture_path)
        self.engine = MarketEngine(config, self.model_path)
        self.learning = LiveLearner(self.engine)
        self.learning_task: asyncio.Task[None] | None = None
        self.feed_task: asyncio.Task[None] | None = None
        self.feed_generation = 0
        self.feed_error: str | None = None
        self._lock = asyncio.Lock()

    @property
    def active_feed_tasks(self) -> int:
        return int(self.feed_task is not None and not self.feed_task.done())

    def snapshot(self) -> dict[str, object]:
        """Return engine state with runtime-owned feed failure telemetry."""
        snapshot = self.engine.snapshot()
        health = snapshot["health"]
        if isinstance(health, dict):
            health["feed_error"] = self.feed_error
        return snapshot

    async def start(self) -> None:
        await self.reset()

    async def stop(self) -> None:
        async with self._lock:
            await self._stop_locked()

    async def configure(self, values: dict[str, object]) -> dict[str, object]:
        async with self._lock:
            self.config.patch(values)
            await self._restart_locked()
            return self.config.public() | {
                "run_id": self.engine.run_id,
                "feed_generation": self.feed_generation,
            }

    async def reset(self) -> dict[str, object]:
        async with self._lock:
            await self._restart_locked()
            return {"run_id": self.engine.run_id, "feed_generation": self.feed_generation}

    async def _stop_locked(self) -> None:
        if self.learning_task is not None:
            self.learning_task.cancel()
            try:
                await self.learning_task
            except asyncio.CancelledError:
                pass
            self.learning_task = None
        if self.feed_task is not None:
            self.feed_task.cancel()
            try:
                await self.feed_task
            except asyncio.CancelledError:
                pass
            except Exception as error:
                # A task failure is telemetry, not a lifecycle failure.
                self._record_feed_failure(self.engine, error)
            self.feed_task = None

    async def _restart_locked(self) -> None:
        # Await cancellation before replacing the engine: the old task must never write new state.
        await self._stop_locked()
        self.feed_generation += 1
        self.feed_error = None
        self.engine = MarketEngine(self.config, self.model_path)
        self.engine.run_id = uuid.uuid4().hex[:12]
        self.engine.feed_generation = self.feed_generation
        self.learning = LiveLearner(self.engine)
        self.learning_task = asyncio.create_task(self._run_learning(self.learning))
        self.feed_task = asyncio.create_task(self._run_feed(self.engine))
        self.feed_task.add_done_callback(
            lambda task, engine=self.engine: self._consume_task_result(task, engine)
        )

    async def _run_learning(self, learner: LiveLearner) -> None:
        while True:
            try:
                learner.tick()
            except Exception as error:
                learner.enabled = False
                learner.pending = None
                learner.stage = "failed"
                learner.error = f"Learning stopped: {type(error).__name__}"
            await asyncio.sleep(0.2)

    def _record_feed_failure(self, engine: MarketEngine, error: Exception) -> None:
        engine.feed_status = "failed"
        # Keep browser telemetry useful without exposing local paths or payloads.
        if engine is self.engine:
            self.feed_error = type(error).__name__

    def _consume_task_result(self, task: asyncio.Task[None], engine: MarketEngine) -> None:
        """Consume unexpected task exceptions so a broken feed cannot poison later resets."""
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as error:
            self._record_feed_failure(engine, error)
        else:
            self._record_feed_failure(engine, FeedEndedError("feed task ended"))
        finally:
            if task is self.feed_task:
                self.feed_task = None

    async def _run_feed(self, engine: MarketEngine) -> None:
        try:
            if self.config.source == "replay":
                engine.feed_status = "replay_loading"
                events = list(ReplayFeed(self.fixture_path))
                anchor_ts_ms = int(time.time() * 1000)
                anchor_clock = time.monotonic()
                for scheduled in replay_schedule(events, anchor_ts_ms, cycles=None):
                    # Fixed deadlines absorb processing time and sleep overshoot instead
                    # of accumulating them until healthy replay books appear stale.
                    deadline = anchor_clock + (scheduled.event.event_ts_ms - anchor_ts_ms) / 1_000
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        await asyncio.sleep(remaining)
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
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._record_feed_failure(engine, error)
            raise
