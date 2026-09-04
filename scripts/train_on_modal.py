#!/usr/bin/env python3
# ruff: noqa: E501
"""Plan, or explicitly run, one bounded CPU-only Modal parity smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

INPUT = Path("data/binance-btcusdt-sample.jsonl")
MODEL = Path("models/gate-binance-demo.npz")
CAPS = {"cpu": 2, "memory_mib": 2048, "timeout_seconds": 300, "retries": 0}
# fmt: off
LIMITATIONS = ["This is a CPU-only reproducibility smoke, not a performance benchmark or trading signal.", "Only caller-provided bytes from a tracked public repository file are sent to Modal.", "The remote model remains ephemeral; this command never replaces a committed model."]
def parse(argv: list[str] | None, root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="invoke the ephemeral Modal CPU job")
    parser.add_argument("--input", type=Path, default=root / INPUT)
    parser.add_argument("--expected-model", type=Path, default=root / MODEL)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-receipt", type=Path, default=root / "artifacts/modal-parity.json")
    return parser.parse_args(argv)


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def redact(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return f"<external>/{path.name or 'path'}"


def plan(root: Path, input_path: Path, model_path: Path, epochs: int, seed: int) -> dict[str, Any]:
    if epochs < 1 or not input_path.is_file() or not model_path.is_file():
        raise ValueError("epochs must be positive and input/model must be files")
    root, input_path, model_path = root.resolve(), input_path.resolve(), model_path.resolve()
    input_sha256, expected_model_sha256 = sha(input_path), sha(model_path)
    config = {"input_sha256": input_sha256, "expected_model_sha256": expected_model_sha256, "epochs": epochs, "seed": seed}
    return {"root": root, "input_path": input_path, "model_path": model_path, "input_sha256": input_sha256, "expected_model_sha256": expected_model_sha256, "epochs": epochs, "seed": seed, "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def public(request: Mapping[str, Any]) -> dict[str, object]:
    return {"input": {"path": redact(request["input_path"], request["root"]), "sha256": request["input_sha256"]}, "expected_model": {"path": redact(request["model_path"], request["root"]), "sha256": request["expected_model_sha256"]}, "epochs": request["epochs"], "seed": request["seed"], "config_sha256": request["config_sha256"], "remote": CAPS}


def git(root: Path, *command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(root), *command], capture_output=True, text=True)


def preflight(request: Mapping[str, Any]) -> str:
    root = request["root"]
    if git(root, "diff", "--quiet", "HEAD", "--").returncode:
        raise ValueError("--run requires every tracked repository file to exactly match HEAD")
    # add_local_dir includes ignored files, so no untracked source byte may exist.
    if git(root, "status", "--porcelain", "--ignored", "--untracked-files=all", "--", "src").stdout:
        raise ValueError("--run requires src to contain only files tracked in HEAD")
    for label, path in (("input", request["input_path"]), ("expected model", request["model_path"])):
        try:
            relative = path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"--run requires a tracked repository {label}") from error
        if git(root, "ls-files", "--error-unmatch", "--", str(relative)).returncode:
            raise ValueError(f"--run requires a tracked repository {label}")
        if git(root, "diff", "--quiet", "HEAD", "--", str(relative)).returncode:
            raise ValueError(f"--run requires {label} bytes that exactly match HEAD")
    commit = git(root, "rev-parse", "HEAD").stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("repository did not return a full commit SHA")
    return commit


def _train_remote(
    recording_bytes: bytes, expected_input_sha256: str, epochs: int, seed: int
) -> dict[str, Any]:
    import time

    sys.path.insert(0, "/repo/src")
    from market_gate.training import train_recording

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="market-gate-parity-") as directory:
        workdir = Path(directory)
        recording, model, receipt = workdir / "recording.jsonl", workdir / "remote-model.npz", workdir / "training.json"
        recording.write_bytes(recording_bytes)
        input_sha256 = hashlib.sha256(recording_bytes).hexdigest()
        if input_sha256 != expected_input_sha256:
            raise ValueError("received bytes do not match the planned sample hash")
        training = train_recording(recording, model, receipt, epochs=epochs, seed=seed)["training"]
        return {"input_sha256": input_sha256, "model_sha256": sha(model), "loss_metrics": {key: training[key] for key in ("first_train_loss", "last_train_loss", "validation_loss")}, "duration_seconds": round(time.monotonic() - started, 3)}


def run_remote(request: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import modal
    except ImportError as error:
        raise RuntimeError(
            "run with: uv run --with modal==1.5.2 python scripts/train_on_modal.py --run"
        ) from error
    image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.5.2", "torch==2.14.0").add_local_dir(request["root"] / "src", remote_path="/repo/src")
    app = modal.App("market-gate-parity-smoke")
    remote = app.function(image=image, cpu=2, memory=2048, timeout=300, retries=0)(_train_remote)
    with app.run():
        return remote.remote(request["input_path"].read_bytes(), request["input_sha256"], request["epochs"], request["seed"])


def value(remote: Mapping[str, object], key: str, *, number: bool = False) -> str | float:
    result = remote.get(key)
    if number:
        if not isinstance(result, (int, float)) or isinstance(result, bool) or not math.isfinite(result):
            raise ValueError(f"remote result has invalid {key}")
        return float(result)
    if not isinstance(result, str) or not re.fullmatch(r"[0-9a-f]{64}", result):
        raise ValueError(f"remote result has invalid {key}")
    return result


def receipt(
    request: Mapping[str, Any], commit: str, remote: Mapping[str, object]
) -> dict[str, object]:
    input_sha256, model_sha256 = value(remote, "input_sha256"), value(remote, "model_sha256")
    if input_sha256 != request["input_sha256"]:
        raise ValueError("remote input hash differs from the planned public sample")
    duration = value(remote, "duration_seconds", number=True)
    if duration < 0:
        raise ValueError("remote duration must not be negative")
    losses = remote.get("loss_metrics")
    if not isinstance(losses, Mapping):
        raise ValueError("remote result has no loss metrics")
    metrics = {key: value(losses, key, number=True) for key in ("first_train_loss", "last_train_loss", "validation_loss")}
    return {"schema_version": 1, "generated_at": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"), "repo_commit": commit, "input": {"path": redact(request["input_path"], request["root"]), "sha256": request["input_sha256"]}, "expected_model": {"path": redact(request["model_path"], request["root"]), "sha256": request["expected_model_sha256"]}, "remote_model": {"sha256": model_sha256}, "config": {"seed": request["seed"], "epochs": request["epochs"], "sha256": request["config_sha256"]}, "loss_metrics": metrics, "duration_seconds": duration, "parity": model_sha256 == request["expected_model_sha256"], "limitations": LIMITATIONS}


def write_receipt(path: Path, request: Mapping[str, Any], result: Mapping[str, object]) -> None:
    path = path.resolve()
    if path in {request["input_path"], request["model_path"]}:
        raise ValueError("output receipt must be separate from the input and expected model")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None, *, root: Path | None = None) -> int:
    root = root or Path(__file__).resolve().parents[1]
    options = parse(argv, root)
    request = plan(root, options.input, options.expected_model, options.epochs, options.seed)
    if not options.run:
        print(json.dumps({"mode": "dry-run", "modal_contact": False, "writes": False, "plan": public(request)}, indent=2))
        return 0
    result = receipt(request, preflight(request), run_remote(request))
    write_receipt(options.output_receipt, request, result)
    print(json.dumps(result, indent=2))
    return 0 if result["parity"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
