"""Local process control and an atomic snapshot shared by local and Modal training."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path


def empty_snapshot() -> dict:
    return dict(
        schema_version=1,
        run_id=None,
        status="idle",
        backend="local",
        updated_at=None,
        dataset=None,
        latest=None,
        history=[],
        evaluation=None,
        error=None,
    )


class SnapshotWriter:
    """Hold one OS lock for the entire run; publish only complete JSON snapshots."""

    def __init__(self, path: Path, run_id: str, backend: str):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = path.with_suffix(".lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise ValueError("another training run is active") from None
        self.state = empty_snapshot() | dict(run_id=run_id, status="running", backend=backend)
        self.mutex = threading.RLock()
        self.publish()

    def publish(self):
        with self.mutex:
            self.state["updated_at"] = datetime.now(UTC).isoformat()
            temporary = self.path.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(self.state, allow_nan=False), encoding="utf-8")
            temporary.replace(self.path)

    def accept(self, event: dict):
        if event["kind"] == "dataset":
            self.state["dataset"] = event["dataset"]
        elif event["kind"] == "step":
            self.state["latest"] = event["step"]
            self.state["history"].append(event["step"])
            self.state["history"] = self.state["history"][-400:]
        elif event["kind"] == "completed":
            self.state.update(status="completed", evaluation=event["evaluation"])
        self.publish()

    def finish(self, status: str, error: str | None = None):
        self.state.update(status=status, error=error)
        self.publish()

    def close(self):
        self.lock.close()


class TrainingService:
    def __init__(self, root: Path):
        self.root = root
        self.path = root / "artifacts/training-live.json"
        self.process: subprocess.Popen | None = None
        self.pending: dict | None = None

    def snapshot(self) -> dict:
        if self.pending is not None:
            published = None
            try:
                published = json.loads(self.path.read_text())
            except (OSError, ValueError):
                pass
            if published and published.get("run_id") == self.pending["run_id"]:
                self.pending = None
            elif self.process is not None and self.process.poll() is None:
                return self.pending.copy()
            else:
                return self.pending | dict(status="failed", error="Training worker failed to start")
        if not self.path.exists():
            return empty_snapshot()
        try:
            result = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return empty_snapshot() | dict(status="failed", error="Training snapshot unavailable")
        if result.get("status") == "running":
            # An OS lock disappears even when a worker is killed before its final write.
            with self.path.with_suffix(".lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    result.update(
                        status="failed", error="Training worker exited without completion"
                    )
        return result

    def start(self) -> dict:
        if self.process is not None and self.process.poll() is None:
            raise ValueError("another training run is active")
        if self.snapshot()["status"] == "running":
            raise ValueError("another training run is active")
        recording = self.root / "data/training-public.jsonl"
        if not recording.is_file():
            raise FileNotFoundError("Record public data to data/training-public.jsonl first")
        run_id = uuid.uuid4().hex
        output = self.root / "artifacts/training-runs" / run_id
        output.parent.mkdir(parents=True, exist_ok=True)
        with (output.parent / f"{run_id}.log").open("x") as log:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    str(self.root / "scripts/run_training_lab.py"),
                    "--input",
                    str(recording),
                    "--output",
                    str(output),
                    "--live-state",
                    str(self.path),
                    "--pace",
                    "0.25",
                ],
                cwd=self.root,
                stdout=log,
                stderr=log,
            )
        # The worker owns the lock and state; never overwrite a concurrent worker's snapshot.
        self.pending = empty_snapshot() | dict(
            run_id=run_id, status="running", updated_at=datetime.now(UTC).isoformat()
        )
        return self.pending.copy()

    def stop(self):
        if self.process is None or self.process.poll() is not None:
            raise ValueError("no local run owned by this server is active")
        self.process.terminate()
        self.process.wait(timeout=10)
        return self.snapshot()

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.stop()
