#!/usr/bin/env python3
"""Optionally run one bounded CPU parity smoke against Modal.

Without ``--run`` this command only hashes and prints a launch plan.  It does
not import or contact Modal, and it does not write a receipt.  ``--run`` opens
one ephemeral Modal app; the remote model is hashed then discarded.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from market_gate.modal_parity import (
    DEFAULT_EXPECTED_MODEL,
    DEFAULT_INPUT,
    ParityPlan,
    build_plan,
    build_receipt,
    validate_output_receipt_path,
    write_receipt_atomic,
)


def parse_args(argv: list[str] | None = None, *, root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="invoke the ephemeral Modal CPU job")
    parser.add_argument("--input", type=Path, default=root / DEFAULT_INPUT)
    parser.add_argument("--expected-model", type=Path, default=root / DEFAULT_EXPECTED_MODEL)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-receipt",
        type=Path,
        default=root / "artifacts" / "modal-parity.json",
        help="local JSON receipt written only after --run",
    )
    return parser.parse_args(argv)


def current_commit(root: Path) -> str:
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("repository did not return a full commit SHA")
    return commit


def require_clean_head_inputs(plan: ParityPlan) -> None:
    """Bind the remote bytes and parity target to the stated repository commit."""
    repository_differs_from_head = subprocess.run(
        ["git", "-C", str(plan.root), "diff", "--quiet", "HEAD", "--"], check=False
    )
    if repository_differs_from_head.returncode != 0:
        raise ValueError("--run requires every tracked repository file to exactly match HEAD")

    # Modal mounts the whole source directory, including ignored files. Refuse any
    # file that is not part of HEAD so local scratch files or secrets cannot ride
    # along with an otherwise clean commit.
    for ignored_args in ((), ("--ignored",)):
        untracked = subprocess.run(
            [
                "git",
                "-C",
                str(plan.root),
                "ls-files",
                "--others",
                *ignored_args,
                "--exclude-standard",
                "--",
                "src",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if untracked.returncode != 0:
            raise RuntimeError("could not inspect the source directory before --run")
        if untracked.stdout.strip():
            raise ValueError("--run requires src to contain only files tracked in HEAD")

    for label, path in (("input", plan.input_path), ("expected model", plan.expected_model_path)):
        try:
            relative = path.relative_to(plan.root)
        except ValueError as error:
            raise ValueError(f"--run requires a tracked repository {label}") from error
        tracked = subprocess.run(
            ["git", "-C", str(plan.root), "ls-files", "--error-unmatch", "--", str(relative)],
            capture_output=True,
            text=True,
            check=False,
        )
        if tracked.returncode != 0:
            raise ValueError(f"--run requires a tracked repository {label}")
        differs_from_head = subprocess.run(
            ["git", "-C", str(plan.root), "diff", "--quiet", "HEAD", "--", str(relative)],
            check=False,
        )
        if differs_from_head.returncode != 0:
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


def run_remote(plan: ParityPlan, repo_commit: str) -> dict[str, Any]:
    """Import Modal only for an explicitly authorized remote invocation."""
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
        .add_local_dir(plan.root / "src", remote_path="/repo/src")
    )
    app = modal.App("market-gate-parity-smoke")
    train_remote = app.function(image=image, cpu=2, memory=2048, timeout=300, retries=0)(
        _train_remote
    )

    with app.run():
        return train_remote.remote(
            plan.input_path.read_bytes(), plan.input_sha256, plan.epochs, plan.seed, repo_commit
        )


def main(argv: list[str] | None = None, *, root: Path | None = None) -> int:
    repo_root = root or Path(__file__).resolve().parents[1]
    args = parse_args(argv, root=repo_root)
    plan = build_plan(
        repo_root, args.input, args.expected_model, epochs=args.epochs, seed=args.seed
    )
    if not args.run:
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "modal_contact": False,
                    "writes": False,
                    "plan": plan.public_dict(),
                },
                indent=2,
            )
        )
        return 0

    require_clean_head_inputs(plan)
    repo_commit = current_commit(repo_root)
    receipt = build_receipt(plan, repo_commit=repo_commit, remote=run_remote(plan, repo_commit))
    if args.output_receipt:
        write_receipt_atomic(validate_output_receipt_path(args.output_receipt, plan), receipt)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["parity"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
