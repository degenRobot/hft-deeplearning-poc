"""Pure planning and receipt helpers for the optional Modal parity smoke."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path("data/binance-btcusdt-sample.jsonl")
DEFAULT_EXPECTED_MODEL = Path("models/gate-binance-demo.npz")
LIMITATIONS = [
    "This is a CPU-only reproducibility smoke, not a performance benchmark or trading signal.",
    "Only caller-provided bytes from a tracked public repository file are sent to Modal.",
    "The remote model remains ephemeral; this command never replaces a committed model.",
]


@dataclass(frozen=True)
class ParityPlan:
    """A fully hashed, side-effect-free request for one remote smoke run."""

    root: Path
    input_path: Path
    expected_model_path: Path
    input_sha256: str
    expected_model_sha256: str
    epochs: int
    seed: int
    config_sha256: str

    def public_dict(self) -> dict[str, object]:
        return {
            "input": {"path": redact_path(self.input_path, self.root), "sha256": self.input_sha256},
            "expected_model": {
                "path": redact_path(self.expected_model_path, self.root),
                "sha256": self.expected_model_sha256,
            },
            "epochs": self.epochs,
            "seed": self.seed,
            "config_sha256": self.config_sha256,
            "remote": {"cpu": 2, "memory_mib": 2048, "timeout_seconds": 300, "retries": 0},
        }


def sha256_file(path: Path) -> str:
    """Hash a regular file without loading its full contents into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def redact_path(path: Path, root: Path) -> str:
    """Keep repository-relative paths and remove local directory details otherwise."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return f"<external>/{path.name or 'path'}"


def build_plan(
    root: Path, input_path: Path, expected_model_path: Path, *, epochs: int, seed: int
) -> ParityPlan:
    """Validate local inputs and bind all mutable launch values into one digest."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    for label, path in (("input", input_path), ("expected model", expected_model_path)):
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {redact_path(path, root)}")
    resolved_root = root.resolve()
    resolved_input = input_path.resolve()
    resolved_expected = expected_model_path.resolve()
    input_sha256 = sha256_file(resolved_input)
    expected_sha256 = sha256_file(resolved_expected)
    bound = {
        "input_sha256": input_sha256,
        "expected_model_sha256": expected_sha256,
        "epochs": epochs,
        "seed": seed,
    }
    config_sha256 = hashlib.sha256(
        json.dumps(bound, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ParityPlan(
        root=resolved_root,
        input_path=resolved_input,
        expected_model_path=resolved_expected,
        input_sha256=input_sha256,
        expected_model_sha256=expected_sha256,
        epochs=epochs,
        seed=seed,
        config_sha256=config_sha256,
    )


def build_receipt(
    plan: ParityPlan, *, repo_commit: str, remote: Mapping[str, Any]
) -> dict[str, object]:
    """Validate the small remote response and create a redacted local receipt."""
    if not re.fullmatch(r"[0-9a-f]{40}", repo_commit):
        raise ValueError("receipt requires a full repository commit SHA")
    remote_model_sha256 = _sha256_value(remote, "model_sha256")
    remote_input_sha256 = _sha256_value(remote, "input_sha256")
    if remote_input_sha256 != plan.input_sha256:
        raise ValueError("remote input hash differs from the planned public sample")
    duration_seconds = _number(remote, "duration_seconds")
    if duration_seconds < 0:
        raise ValueError("remote duration must not be negative")
    losses = remote.get("loss_metrics")
    if not isinstance(losses, Mapping):
        raise ValueError("remote result has no loss metrics")
    loss_metrics = {
        key: _number(losses, key)
        for key in ("first_train_loss", "last_train_loss", "validation_loss")
    }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
        "repo_commit": repo_commit,
        "input": {"path": redact_path(plan.input_path, plan.root), "sha256": plan.input_sha256},
        "expected_model": {
            "path": redact_path(plan.expected_model_path, plan.root),
            "sha256": plan.expected_model_sha256,
        },
        "remote_model": {"sha256": remote_model_sha256},
        "config": {
            "seed": plan.seed,
            "epochs": plan.epochs,
            "sha256": plan.config_sha256,
        },
        "loss_metrics": loss_metrics,
        "duration_seconds": duration_seconds,
        "parity": remote_model_sha256 == plan.expected_model_sha256,
        "limitations": LIMITATIONS,
    }


def validate_output_receipt_path(path: Path, plan: ParityPlan) -> Path:
    """Refuse paths that could overwrite either committed binary input."""
    resolved = path.resolve()
    if resolved in {plan.input_path, plan.expected_model_path}:
        raise ValueError("output receipt must be separate from the input and expected model")
    return resolved


def write_receipt_atomic(path: Path, receipt: Mapping[str, object]) -> None:
    """Write a receipt atomically after callers have explicitly requested it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _sha256_value(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"remote result has invalid {key}")
    return value


def _number(mapping: Mapping[str, Any], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"remote result has invalid {key}")
    return float(value)
