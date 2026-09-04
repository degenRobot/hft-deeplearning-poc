#!/usr/bin/env python3
"""Plan, or explicitly run, one bounded CPU-only Modal parity smoke."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path("data/binance-btcusdt-sample.jsonl")
DEFAULT_MODEL = Path("models/gate-binance-demo.npz")
CAPS = {"cpu": 2, "memory": 2048, "timeout": 300, "retries": 0}


def args(argv: list[str] | None, root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="invoke the ephemeral Modal CPU job")
    parser.add_argument("--input", type=Path, default=root / DEFAULT_INPUT)
    parser.add_argument("--expected-model", type=Path, default=root / DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-receipt",
        type=Path,
        default=root / "artifacts" / "modal-parity.json",
        help="local JSON receipt written only after --run",
    )
    return parser.parse_args(argv)


def digest(path: Path) -> str:
    import hashlib

    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def git(root: Path, *command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(root), *command], capture_output=True, text=True)


def plan(
    root: Path, input_path: Path, expected_model: Path, epochs: int, seed: int
) -> dict[str, Any]:
    if epochs < 1 or not input_path.is_file() or not expected_model.is_file():
        raise ValueError("epochs must be positive and input/model must be files")
    input_path, expected_model = input_path.resolve(), expected_model.resolve()
    hashes = {"input": digest(input_path), "expected_model": digest(expected_model)}
    config = {**hashes, "epochs": epochs, "seed": seed}
    import hashlib

    return {
        "root": root.resolve(),
        "input_path": input_path,
        "model_path": expected_model,
        "epochs": epochs,
        "seed": seed,
        **hashes,
        "config": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
    }


def require_clean_head_inputs(request: dict[str, Any]) -> None:
    root = request["root"]
    if git(root, "diff", "--quiet", "HEAD", "--").returncode:
        raise ValueError("--run requires every tracked repository file to exactly match HEAD")
    # add_local_dir would upload ignored scratch files too; reject every non-HEAD src file.
    if git(root, "status", "--porcelain", "--ignored", "--untracked-files=all", "--", "src").stdout:
        raise ValueError("--run requires src to contain only files tracked in HEAD")
    for label, path in (
        ("input", request["input_path"]),
        ("expected model", request["model_path"]),
    ):
        try:
            relative = path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"--run requires a tracked repository {label}") from error
        if git(root, "ls-files", "--error-unmatch", "--", str(relative)).returncode:
            raise ValueError(f"--run requires a tracked repository {label}")
        if git(root, "diff", "--quiet", "HEAD", "--", str(relative)).returncode:
            raise ValueError(f"--run requires {label} bytes that exactly match HEAD")


def _train_remote(
    recording_bytes: bytes,
    expected_input_sha256: str,
    epochs: int,
    seed: int,
    repo_commit: str,
) -> dict[str, Any]:
    """Run inside the ephemeral image and return hashes and metrics only."""
    import hashlib
    import sys
    import tempfile
    import time
    from pathlib import Path

    sys.path.insert(0, "/repo/src")
    from market_gate.training import train_recording

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="market-gate-parity-") as directory:
        workdir = Path(directory)
        recording_path = workdir / "recording.jsonl"
        model_path = workdir / "remote-model.npz"
        receipt_path = workdir / "remote-training.json"
        recording_path.write_bytes(recording_bytes)
        input_sha256 = hashlib.sha256(recording_bytes).hexdigest()
        if input_sha256 != expected_input_sha256:
            raise ValueError("received bytes do not match the planned sample hash")
        training = train_recording(
            recording_path, model_path, receipt_path, epochs=epochs, seed=seed
        )
        metrics = training["training"]
        return {
            "input_sha256": input_sha256,
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "loss_metrics": {
                "first_train_loss": metrics["first_train_loss"],
                "last_train_loss": metrics["last_train_loss"],
                "validation_loss": metrics["validation_loss"],
            },
            "duration_seconds": round(time.monotonic() - started, 3),
            "repo_commit": repo_commit,
        }


def run_remote(request: dict[str, Any], commit: str) -> dict[str, Any]:
    try:
        import modal
    except ImportError as error:
        raise RuntimeError(
            "run with the pinned dependency: uv run --with modal==1.5.2 "
            "python scripts/train_on_modal.py --run"
        ) from error

    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("numpy==2.5.2", "torch==2.14.0")
        .add_local_dir(request["root"] / "src", remote_path="/repo/src")
    )
    app = modal.App("market-gate-parity-smoke")
    train_remote = app.function(image=image, **CAPS)(_train_remote)

    with app.run():
        return train_remote.remote(
            request["input_path"].read_bytes(),
            request["input"],
            request["epochs"],
            request["seed"],
            commit,
        )


def main(argv: list[str] | None = None, *, root: Path | None = None) -> int:
    repo_root = root or Path(__file__).resolve().parents[1]
    options = args(argv, repo_root)
    request = plan(repo_root, options.input, options.expected_model, options.epochs, options.seed)
    public = {
        key: value for key, value in request.items() if not key.endswith("path") and key != "root"
    }
    if not options.run:
        print(
            json.dumps(
                {"mode": "dry-run", "modal_contact": False, "writes": False, "plan": public},
                indent=2,
            )
        )
        return 0
    require_clean_head_inputs(request)
    commit = git(repo_root, "rev-parse", "HEAD").stdout.strip()
    remote = run_remote(request, commit)
    if remote["input_sha256"] != request["input"]:
        raise ValueError("remote input hash differs from the planned public sample")
    receipt = {
        "repo_commit": commit,
        "input_sha256": request["input"],
        "expected_model_sha256": request["expected_model"],
        "remote_model_sha256": remote["model_sha256"],
        "config_sha256": request["config"],
        "parity": remote["model_sha256"] == request["expected_model"],
        "loss_metrics": remote["loss_metrics"],
        "duration_seconds": remote["duration_seconds"],
    }
    output = options.output_receipt.resolve()
    if output in {request["input_path"], request["model_path"]}:
        raise ValueError("output receipt must be separate from input and expected model")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["parity"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
