"""Bounded public recording capture and immutable selection for future training runs."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

SYMBOL = "BTCUSDT"
MAX_EVENTS = 500_000
MAX_BYTES = 100_000_000
BOOK_INTERVAL_MS = 100
ACTIVE = {"running", "stopping", "validating"}
CAPTURE_ID = re.compile(r"[0-9a-f]{32}\Z")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _duration(seconds: int) -> int:
    if type(seconds) is not int or not 30 <= seconds <= 1800:
        raise ValueError("Capture duration must be an integer from 30 to 1800 seconds")
    return seconds


def _capture_path(root: Path, capture_id: str) -> Path:
    if not isinstance(capture_id, str) or CAPTURE_ID.fullmatch(capture_id) is None:
        raise ValueError("Invalid capture ID")
    return root / "data/training-captures" / f"{capture_id}.jsonl"


def _write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(state, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def inspect_recording(root: Path, path: Path, dataset_id: str) -> dict:
    """Check the exact causal training split, not a duration or event-count proxy."""
    from .contracts import BookEvent
    from .training import build_examples, build_frame_dataset, load_recording
    from .training_lab import split_lab_examples

    result = {
        "id": dataset_id,
        "label": "Built-in public recording"
        if dataset_id == "builtin"
        else "Captured public recording",
        "path": str(path.relative_to(root)),
        "source": "binance_public",
        "symbol": SYMBOL,
        "event_count": 0,
        "book_count": 0,
        "trade_count": 0,
        "candle_count": 0,
        "source_mode": "recorded market replay",
        "bytes": 0,
        "first_event_ts_ms": None,
        "last_event_ts_ms": None,
        "sha256": None,
        "frame_count": 0,
        "training_ready": False,
        "error": None,
    }
    if not path.is_file() or path.is_symlink():
        result["error"] = "Recording is unavailable"
        return result
    result["bytes"] = path.stat().st_size
    if result["bytes"] > MAX_BYTES:
        result["error"] = "Recording exceeds 100 MB"
        return result
    result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        with path.open(encoding="utf-8") as recording:
            first = next((json.loads(line) for line in recording if line.strip()), {})
        if isinstance(first, dict) and first.get("kind") == "candle":
            from .historical import load_candle_frames

            frames, metadata = load_candle_frames(path)
            if metadata["event_count"] > MAX_EVENTS:
                raise ValueError("Historical recording exceeds the event limit")
            result.update(
                label="Historical Binance 1s candles",
                source="binance_historical_candles",
                source_mode="historical_candles_1s",
                symbol=metadata["symbol"],
                event_count=metadata["event_count"],
                candle_count=metadata["candle_count"],
                first_event_ts_ms=metadata.get("first_event_ts_ms", int(frames.close_ts_ms[0])),
                last_event_ts_ms=metadata.get("last_event_ts_ms", int(frames.close_ts_ms[-1])),
                feature_names=metadata["feature_names"],
                limitations=metadata.get("limitations", []),
            )
        else:
            events = load_recording(path)
            if len(events) > MAX_EVENTS or any(
                e.symbol != SYMBOL or e.venue != "binance" for e in events
            ):
                raise ValueError(
                    "Capture must contain at most 500000 public Binance BTCUSDT events"
                )
            result.update(
                event_count=len(events),
                book_count=sum(isinstance(e, BookEvent) for e in events),
                trade_count=sum(not isinstance(e, BookEvent) for e in events),
                first_event_ts_ms=min(e.event_ts_ms for e in events),
                last_event_ts_ms=max(e.event_ts_ms for e in events),
            )
            frames = build_frame_dataset(events)
        result["frame_count"] = len(frames.values)
        examples = build_examples(frames.values, frames.mids, 30, 5, close_ts_ms=frames.close_ts_ms)
        split_lab_examples(examples, len(frames.values), 15)
        result["training_ready"] = True
    except (ValueError, KeyError, TypeError, IndexError, UnicodeError) as error:
        # Split failures are useful explanations; malformed-file errors can contain paths.
        message = str(error)
        result["error"] = (
            message
            if message.startswith("insufficient contiguous")
            else "Recording does not contain enough valid contiguous public data"
        )
    return result


class PublicDatasetService:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.path = self.root / "artifacts/training-data.json"
        self.lock_path = self.path.with_suffix(".lock")
        self.process: subprocess.Popen | None = None
        self._builtin: dict | None = None
        self._mutex = threading.RLock()

    def _default_selected(self) -> dict:
        if self._builtin is None:
            self._builtin = inspect_recording(
                self.root, self.root / "data/training-public.jsonl", "builtin"
            )
        return self._builtin.copy()

    def _state(self) -> dict:
        try:
            state = json.loads(self.path.read_text())
            if state.get("schema_version") != 1 or not isinstance(state.get("capture"), dict):
                raise ValueError("Invalid capture state")
            if state["capture"].get("status") not in ACTIVE | {
                "idle",
                "completed",
                "incomplete",
                "stopped",
                "failed",
            }:
                raise ValueError("Invalid capture status")
            selected = state["selected"]
            expected = (
                "data/training-public.jsonl"
                if selected["id"] == "builtin"
                else str(_capture_path(self.root, selected["id"]).relative_to(self.root))
            )
            if selected["path"] != expected:
                raise ValueError("Invalid selection path")
            return state
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return {
                "schema_version": 1,
                "selected": self._default_selected(),
                "capture": {"id": None, "status": "idle", "can_stop": False},
            }

    def _acquire(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.lock_path.open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise ValueError("Another public data capture is active") from None
        return lock

    def snapshot(self) -> dict:
        with self._mutex:
            state = self._state()
            capture = state["capture"]
            if capture["status"] in ACTIVE:
                try:
                    lock = self._acquire()
                except ValueError:
                    pass
                else:
                    lock.close()
                    capture.update(
                        status="failed", error="Capture worker exited without completion"
                    )
            capture["can_stop"] = (
                self.process is not None
                and self.process.poll() is None
                and capture["status"] in ACTIVE
            )
            return state

    def selected_recording(self) -> Path:
        selected = self._state()["selected"]
        path = self.root / selected["path"]
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError("Selected public recording is unavailable")
        return path

    def start(self, seconds: int) -> dict:
        return self._start(_duration(seconds))

    def start_history(self, symbol: str, start: str, end: str) -> dict:
        from .historical import validate_history_request

        request = validate_history_request(symbol, start, end)
        return self._start(request["seconds"], history=request)

    def _start(self, seconds: int, *, history: dict | None = None) -> dict:
        with self._mutex:
            if self.process is not None and self.process.poll() is None:
                raise ValueError("Another public data capture is active")
            lock = self._acquire()
            try:
                state = self._state()
                capture_id = uuid.uuid4().hex
                state["capture"] = {
                    "id": capture_id,
                    "status": "running",
                    "mode": "historical" if history else "live",
                    "requested_seconds": seconds,
                    "elapsed_seconds": 0.0,
                    "progress": 0.0,
                    "events": 0,
                    "book_count": 0,
                    "trade_count": 0,
                    "candle_count": 0,
                    "bytes": 0,
                    "first_event_ts_ms": None,
                    "last_event_ts_ms": None,
                    "started_at": _now(),
                    "ended_at": None,
                    "updated_at": _now(),
                    "error": None,
                    "can_stop": True,
                    "symbol": history["symbol"] if history else SYMBOL,
                    "source": "binance_historical_candles" if history else "binance_public",
                    "sha256": None,
                }
                if history:
                    state["capture"].update(start=history["start"], end=history["end"])
                _write_state(self.path, state)
                command = [
                    sys.executable,
                    str(self.root / "scripts/capture_training_data.py"),
                    "--id",
                    capture_id,
                    "--seconds",
                    str(seconds),
                    "--lock-fd",
                    str(lock.fileno()),
                ]
                if history:
                    command += [
                        "--mode",
                        "historical",
                        "--symbol",
                        history["symbol"],
                        "--start",
                        history["start"],
                        "--end",
                        history["end"],
                    ]
                with (self.path.parent / f"training-data-{capture_id}.log").open("x") as log:
                    self.process = subprocess.Popen(
                        command, cwd=self.root, stdout=log, stderr=log, pass_fds=(lock.fileno(),)
                    )
            except BaseException:
                if "state" in locals() and "capture_id" in locals():
                    state["capture"].update(
                        status="failed", error="Capture worker failed to start", can_stop=False
                    )
                    _write_state(self.path, state)
                raise
            finally:
                lock.close()
            return self.snapshot()

    def stop(self) -> dict:
        with self._mutex:
            if self.process is None or self.process.poll() is not None:
                raise ValueError("No capture owned by this server is active")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Keep observing its lock and progress; never report an unconfirmed stop.
                pass
            return self.snapshot()

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.stop()


async def capture_dataset(
    root: Path,
    capture_id: str,
    seconds: int,
    recorder,
    *,
    lock_fd: int | None = None,
    heartbeat_seconds: float = 0.5,
    history: dict | None = None,
) -> dict:
    """Run either public recorder under the shared capture owner and validation lifecycle."""
    if history is None:
        _duration(seconds)
    else:
        from .historical import validate_history_request

        history = validate_history_request(history["symbol"], history["start"], history["end"])
        if seconds != history["seconds"]:
            raise ValueError("Historical duration does not match the requested interval")
    root = Path(root).resolve()
    output = _capture_path(root, capture_id)
    service = PublicDatasetService(root)
    if lock_fd is None:
        lock = service._acquire()
    else:
        lock = os.fdopen(lock_fd, "a")
        if os.fstat(lock.fileno()).st_ino != service.lock_path.stat().st_ino:
            lock.close()
            raise ValueError("Invalid capture lock")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = service._state()
    capture = state["capture"]
    if capture.get("id") != capture_id or capture.get("status") != "running":
        lock.close()
        raise ValueError("Capture request does not match the current state")
    if history is not None and any(
        capture.get(key) != history[key] for key in ("symbol", "start", "end")
    ):
        lock.close()
        raise ValueError("Historical request does not match the current state")
    started = time.monotonic()
    stop_heartbeat = threading.Event()
    mutex = threading.RLock()

    def publish():
        with mutex:
            elapsed = time.monotonic() - started
            capture.update(
                elapsed_seconds=round(elapsed, 3),
                progress=1.0
                if capture["status"] == "completed"
                else min(capture.get("progress", 0.0) if history else elapsed / seconds, 0.99),
                updated_at=_now(),
            )
            _write_state(service.path, state)

    def progress(receipt):
        with mutex:
            capture.update(
                events=receipt["total"],
                book_count=receipt["book"],
                trade_count=receipt["trade"],
                candle_count=receipt.get("candle", 0),
                bytes=receipt["bytes"],
                first_event_ts_ms=receipt["first_event_ts_ms"],
                last_event_ts_ms=receipt["last_event_ts_ms"],
                feed_status=receipt["feed_status"],
                reconnects=receipt["reconnects"],
            )
            if history:
                capture["progress"] = max(0.0, min(float(receipt["progress"]), 0.99))
                capture["coverage_fraction"] = receipt.get("coverage_fraction", 0.0)
                capture["missing_candle_count"] = receipt.get("missing_candle_count", seconds)

    def heartbeat():
        while not stop_heartbeat.wait(heartbeat_seconds):
            publish()

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    status, error = "completed", None
    try:
        try:
            result = await recorder(
                output,
                history["symbol"] if history else SYMBOL,
                seconds,
                MAX_EVENTS,
                BOOK_INTERVAL_MS,
                MAX_BYTES,
                progress_callback=progress,
            )
            capture["stop_reason"] = result["stop_reason"]
            if history and result["stop_reason"] == "source_exhausted":
                status, error = (
                    "incomplete",
                    (
                        "Binance returned no further candles before the requested end. "
                        "The previous dataset remains selected; try another UTC range."
                    ),
                )
        except asyncio.CancelledError:
            status = "stopped"
        except Exception:
            status, error = "failed", "Public capture failed; see the local capture log"
        capture["status"] = (
            "validating" if status == "completed" else "stopping" if status == "stopped" else status
        )
        publish()
        # Parsing and split construction can take time. Keep the event loop responsive
        # to Stop, but retain the capture lock until the read-only validator finishes.
        validation = asyncio.create_task(
            asyncio.to_thread(inspect_recording, root, output, capture_id)
        )
        metadata = None
        while True:
            try:
                metadata = await asyncio.shield(validation)
                break
            except asyncio.CancelledError:
                status = "stopped"
                capture["status"] = "stopping"
                publish()
            except Exception:
                if status != "stopped":
                    status, error = "failed", "Public capture validation failed"
                break
        if metadata is not None:
            capture.update(
                events=metadata["event_count"],
                book_count=metadata["book_count"],
                trade_count=metadata["trade_count"],
                candle_count=metadata.get("candle_count", 0),
                bytes=metadata["bytes"],
                first_event_ts_ms=metadata["first_event_ts_ms"],
                last_event_ts_ms=metadata["last_event_ts_ms"],
                sha256=metadata["sha256"],
                frame_count=metadata["frame_count"],
                training_ready=metadata["training_ready"],
                path=metadata["path"],
            )
            if status == "completed":
                if metadata["training_ready"]:
                    state["selected"] = metadata
                else:
                    status, error = "incomplete", metadata["error"]
        capture.update(status=status, error=error, ended_at=_now(), can_stop=False)
        # Drain the state writer before publishing the terminal snapshot or unlocking.
        stop_heartbeat.set()
        thread.join()
        publish()
        return state
    finally:
        stop_heartbeat.set()
        thread.join()
        lock.close()
