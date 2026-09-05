#!/usr/bin/env python3
"""One ephemeral CPU training run; stream real steps and retrieve experimental artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from market_gate.training_service import SnapshotWriter  # noqa: E402

SOURCE_FILES = (
    "__init__.py",
    "training_lab.py",
    "training_options.py",
    "training.py",
    "contracts.py",
    "experts.py",
    "features.py",
    "gate.py",
)
CAPS = dict(cpu=(2, 2), memory=(2048, 2048), timeout=600, retries=0, max_containers=1)


class TrainingCancelled(Exception):
    """Unlike KeyboardInterrupt, this is not suppressed by Modal's app context."""


def remote_training(recording_bytes: bytes, expected_sha: str, options: dict):
    import time

    import modal

    from market_gate.training_lab import train_lab

    yield {"kind": "execution", "function_call_id": modal.current_function_call_id()}

    if len(recording_bytes) > 100_000_000:
        raise ValueError("recording exceeds 100 MB")
    if hashlib.sha256(recording_bytes).hexdigest() != expected_sha:
        raise ValueError("recording hash mismatch")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        recording = root / "recording.jsonl"
        recording.write_bytes(recording_bytes)
        output = root / "run"
        for event in train_lab(recording, output, **options):
            yield event
            if event["kind"] == "step":
                time.sleep(0.15)
        # Transport only the known small model/receipt outputs, never a directory archive.
        files = {}
        for path in output.iterdir():
            if path.is_file() and path.suffix in {".npz", ".json"}:
                if path.stat().st_size > 16_000_000:
                    raise ValueError("artifact exceeds 16 MB")
                files[path.name] = path.read_bytes()
        yield {
            "kind": "artifacts",
            "files": files,
            "function_call_id": modal.current_function_call_id(),
        }


def load_credentials(path: Path):
    """Read only the two named credentials locally; never evaluate a shell dotenv file."""
    from market_gate.training_credentials import read_credentials

    os.environ.update(read_credentials(path))
    if not all(os.environ.get(key) for key in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")):
        raise ValueError("MODAL_TOKEN_ID and MODAL_TOKEN_SECRET are required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--input", type=Path, default=ROOT / "data/training-public.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--hidden-1", type=int, default=64)
    parser.add_argument("--hidden-2", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-rl-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    from market_gate.training_options import TrainingOptions

    options = TrainingOptions(
        hidden_1=args.hidden_1,
        hidden_2=args.hidden_2,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_rl_steps=args.max_rl_steps,
        seed=args.seed,
    ).validate()
    if not args.input.is_file() or args.input.stat().st_size > 100_000_000:
        parser.error("input must be a public recording of at most 100 MB")
    if args.output.exists():
        parser.error("output must be a new directory")
    data = args.input.read_bytes()
    input_sha = hashlib.sha256(data).hexdigest()
    source_hashes = {
        name: hashlib.sha256((ROOT / "src/market_gate" / name).read_bytes()).hexdigest()
        for name in SOURCE_FILES
    }
    plan = dict(
        input_sha256=input_sha,
        input_bytes=len(data),
        sources=source_hashes,
        resources=CAPS,
        **options.to_dict(),
        note="Execution timeout excludes image build/startup; limits are not a billing cap.",
    )
    if not args.run:
        print(json.dumps(plan, indent=2))
        return
    from market_gate.training import build_examples, build_frame_dataset, load_recording
    from market_gate.training_lab import split_lab_examples

    frames = build_frame_dataset(load_recording(args.input))
    rows = build_examples(frames.values, frames.mids, 30, 5, close_ts_ms=frames.close_ts_ms)
    split_lab_examples(rows, len(frames.values), options.max_rl_steps)
    # Upload only reviewed committed module files. No source directory, .env or auto mounts.
    for name in SOURCE_FILES:
        relative = f"src/market_gate/{name}"
        committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
        if hashlib.sha256(committed).hexdigest() != source_hashes[name]:
            raise ValueError(f"commit reviewed source before cloud run: {relative}")
    load_credentials(args.env_file)
    import modal

    writer = SnapshotWriter(ROOT / "artifacts/training-live.json", args.output.name, "modal")
    args.output.mkdir(parents=True, exist_ok=False)
    heartbeat_stop = threading.Event()

    def heartbeat():
        # Snapshot freshness reports the local monitor's liveness, including image startup.
        while not heartbeat_stop.wait(2):
            writer.publish()

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    call_id = None

    def stop_signal(*_):
        raise TrainingCancelled

    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    try:
        with tempfile.TemporaryDirectory(prefix="training-source-") as temporary:
            staging = Path(temporary)
            for name in SOURCE_FILES:
                (staging / name).write_bytes(
                    subprocess.check_output(
                        ["git", "show", f"HEAD:src/market_gate/{name}"], cwd=ROOT
                    )
                )
            image = (
                modal.Image.debian_slim(python_version="3.12")
                .pip_install("numpy==2.5.2", "torch==2.14.0")
                .env({"PYTHONPATH": "/training-src"})
            )
            for name in SOURCE_FILES:
                image = image.add_local_file(staging / name, f"/training-src/market_gate/{name}")
            app = modal.App("market-gate-training-example", include_source=False)
            # Modal generators reject even retries=0; omission means no retry policy.
            generator_caps = {key: value for key, value in CAPS.items() if key != "retries"}
            remote = app.function(image=image, serialized=True, **generator_caps)(remote_training)
            with modal.enable_output(), app.run():
                completion = None
                artifact_count = 0
                for event in remote.remote_gen(data, input_sha, options.to_dict()):
                    if event["kind"] == "execution":
                        call_id = event["function_call_id"]
                        writer.state["execution"] = {"function_call_id": call_id}
                        writer.publish()
                    elif event["kind"] == "artifacts":
                        for name, content in event["files"].items():
                            if (
                                Path(name).name != name
                                or Path(name).suffix not in {".npz", ".json"}
                                or len(content) > 16_000_000
                            ):
                                raise ValueError("invalid returned artifact")
                            with (args.output / name).open("xb") as handle:
                                handle.write(content)
                            artifact_count += 1
                        plan["function_call_id"] = event["function_call_id"]
                    elif event["kind"] == "completed":
                        completion = event
                    else:
                        writer.accept(event)
                if completion is None or artifact_count < 3:
                    raise ValueError("remote training did not return completed model artifacts")
                receipt = json.loads((args.output / "training-lab.json").read_text())
                if receipt["dataset"]["sha256"] != input_sha:
                    raise ValueError("returned receipt does not match the public recording")
                artifact_hashes = {}
                for phase in ("supervised", "adapted"):
                    filename = receipt["artifacts"][phase]["path"]
                    if Path(filename).name != filename:
                        raise ValueError("invalid artifact receipt path")
                    path = args.output / filename
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    if digest != receipt["artifacts"][phase]["sha256"]:
                        raise ValueError("returned model does not match its receipt")
                    artifact_hashes[path.name] = digest
                plan["artifact_sha256"] = artifact_hashes
                plan["app_id"] = app.app_id
                plan["evaluation"] = completion["evaluation"]
                plan["status"] = "completed"
                (args.output / "modal-execution.json").write_text(json.dumps(plan, indent=2))
                writer.accept(completion)
        print(
            json.dumps(
                {
                    "status": "completed",
                    "output": str(args.output),
                    "evaluation": plan["evaluation"],
                }
            )
        )
    except TrainingCancelled:
        if call_id:
            modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)
        writer.finish("stopped")
    except BaseException:
        writer.finish("failed", "Modal run failed; see the local command log")
        raise
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=3)
        writer.close()


if __name__ == "__main__":
    main()
