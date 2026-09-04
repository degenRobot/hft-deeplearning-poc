from __future__ import annotations

import hashlib
import importlib.util
import subprocess
from pathlib import Path

import pytest

from market_gate.modal_parity import (
    build_plan,
    build_receipt,
    redact_path,
    validate_output_receipt_path,
)


def _remote(plan, *, model_sha256: str | None = None) -> dict[str, object]:
    return {
        "input_sha256": plan.input_sha256,
        "model_sha256": model_sha256 or plan.expected_model_sha256,
        "duration_seconds": 1.25,
        "loss_metrics": {
            "first_train_loss": -0.3,
            "last_train_loss": -0.5,
            "validation_loss": -0.2,
        },
    }


def _write(path: Path, contents: bytes) -> Path:
    path.write_bytes(contents)
    return path


def _load_script():
    script = Path(__file__).parents[1] / "scripts" / "train_on_modal.py"
    spec = importlib.util.spec_from_file_location("train_on_modal_test", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_hash_binds_sample_model_and_config(tmp_path: Path) -> None:
    input_path = _write(tmp_path / "sample.jsonl", b"public bytes")
    model_path = _write(tmp_path / "expected.npz", b"expected model")
    first = build_plan(tmp_path, input_path, model_path, epochs=20, seed=7)
    different_seed = build_plan(tmp_path, input_path, model_path, epochs=20, seed=8)
    _write(input_path, b"changed public bytes")
    different_input = build_plan(tmp_path, input_path, model_path, epochs=20, seed=7)

    assert first.config_sha256 != different_seed.config_sha256
    assert first.config_sha256 != different_input.config_sha256
    assert first.public_dict()["input"]["path"] == "sample.jsonl"  # type: ignore[index]


def test_external_paths_are_redacted_in_receipt(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    external_dir = tmp_path / "private"
    external_dir.mkdir()
    external = _write(external_dir / "sample.jsonl", b"public bytes")
    model = _write(root / "expected.npz", b"model")
    plan = build_plan(root, external, model, epochs=20, seed=7)
    receipt = build_receipt(plan, repo_commit="a" * 40, remote=_remote(plan))

    assert redact_path(external, root) == "<external>/sample.jsonl"
    assert receipt["input"]["path"] == "<external>/sample.jsonl"  # type: ignore[index]


def test_mismatch_is_explicit_and_output_must_be_separate(tmp_path: Path) -> None:
    input_path = _write(tmp_path / "sample.jsonl", b"public bytes")
    model_path = _write(tmp_path / "expected.npz", b"expected model")
    plan = build_plan(tmp_path, input_path, model_path, epochs=20, seed=7)
    mismatch = hashlib.sha256(b"other model").hexdigest()
    receipt = build_receipt(plan, repo_commit="b" * 40, remote=_remote(plan, model_sha256=mismatch))

    assert receipt["parity"] is False
    with pytest.raises(ValueError, match="separate"):
        validate_output_receipt_path(model_path, plan)
    assert validate_output_receipt_path(tmp_path / "receipt.json", plan).name == "receipt.json"


def test_default_dry_run_never_calls_remote_or_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True)
    (root / "models").mkdir()
    _write(root / "data" / "binance-btcusdt-sample.jsonl", b"public bytes")
    _write(root / "models" / "gate-binance-demo.npz", b"expected model")
    module = _load_script()

    def remote_called(*_args: object) -> None:
        raise AssertionError("remote called")

    monkeypatch.setattr(module, "run_remote", remote_called)
    output = root / "artifacts" / "modal-parity.json"

    assert module.main([], root=root) == 0
    assert not output.exists()


def test_run_gate_requires_both_inputs_to_match_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True)
    (root / "models").mkdir()
    input_path = _write(root / "data" / "sample.jsonl", b"public bytes")
    model_path = _write(root / "models" / "expected.npz", b"expected model")
    trainer_path = _write(root / "trainer.py", b"head source")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True
    )
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
    plan = build_plan(root, input_path, model_path, epochs=20, seed=7)
    module = _load_script()

    module.require_clean_head_inputs(plan)
    mismatch = hashlib.sha256(b"other model").hexdigest()
    monkeypatch.setattr(
        module,
        "run_remote",
        lambda launch_plan, _commit: _remote(launch_plan, model_sha256=mismatch),
    )
    output = root / "artifacts" / "mismatch.json"
    assert (
        module.main(
            [
                "--run",
                "--input",
                str(input_path),
                "--expected-model",
                str(model_path),
                "--output-receipt",
                str(output),
            ],
            root=root,
        )
        == 1
    )
    assert '"parity": false' in output.read_text(encoding="utf-8")

    _write(trainer_path, b"dirty source")
    with pytest.raises(ValueError, match="every tracked repository file"):
        module.require_clean_head_inputs(plan)
    _write(trainer_path, b"head source")
    _write(model_path, b"changed model")
    with pytest.raises(ValueError, match="exactly match HEAD"):
        module.require_clean_head_inputs(plan)
