from __future__ import annotations

import builtins
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


def script():
    path = Path(__file__).parents[1] / "scripts" / "train_on_modal.py"
    spec = importlib.util.spec_from_file_location("modal_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "repo"
    for directory in ("data", "models", "src"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    sample, model = root / "data/sample.jsonl", root / "models/gate.npz"
    (root / "src/gate.py").write_text("safe = True\n")
    sample.write_bytes(b"public sample")
    model.write_bytes(b"expected model")
    (root / ".gitignore").write_text("src/private.env\n")
    for command in (
        ("init", "-q"),
        ("config", "user.email", "test@example.invalid"),
        ("config", "user.name", "Test"),
        ("add", "."),
        ("commit", "-qm", "fixture"),
    ):
        subprocess.run(["git", "-C", str(root), *command], check=True)
    return root, sample, model


def remote(request, **changes):
    value = {
        "input_sha256": request["input_sha256"],
        "model_sha256": request["expected_model_sha256"],
        "duration_seconds": 1.25,
        "loss_metrics": {
            "first_train_loss": -0.3,
            "last_train_loss": -0.5,
            "validation_loss": -0.2,
        },
    }
    value.update(changes)
    return value


def test_plan_is_redacted_and_binds_hashes_config_and_caps(tmp_path: Path) -> None:
    root, sample, model = repo(tmp_path)
    module = script()
    request = module.plan(root, sample, model, 20, 7)
    changed = module.plan(root, sample, model, 20, 8)
    sample.write_bytes(b"changed sample")
    changed_sample = module.plan(root, sample, model, 20, 7)
    display = module.public(request)
    assert display["input"] == {"path": "data/sample.jsonl", "sha256": request["input_sha256"]}
    assert display["expected_model"]["path"] == "models/gate.npz"
    assert request["config_sha256"] != changed["config_sha256"]
    assert request["config_sha256"] != changed_sample["config_sha256"]
    assert display["remote"] == {"cpu": 2, "memory": 2048, "timeout": 300, "retries": 0}


def test_dry_run_never_calls_remote_or_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, sample, model = repo(tmp_path)
    module = script()
    importer = builtins.__import__

    def no_modal(name: str, *args: object, **kwargs: object) -> object:
        if name == "modal":
            raise AssertionError("Modal imported")
        return importer(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_modal)
    monkeypatch.setattr(module, "run_remote", lambda *_: pytest.fail("remote called"))
    assert module.main(["--input", str(sample), "--expected-model", str(model)], root=root) == 0
    assert not (root / "artifacts/modal-parity.json").exists()


def test_preflight_rejects_dirty_inputs_and_source_uploads(tmp_path: Path) -> None:
    root, sample, model = repo(tmp_path)
    module, request = script(), script().plan(root, sample, model, 20, 7)
    module.preflight(request)
    cases = [
        (root / "src/gate.py", "dirty", "every tracked"),
        (model, "changed", "every tracked"),
        (root / "src/private.env", "secret", "only files tracked"),
        (root / "src/scratch.py", "scratch", "only files tracked"),
    ]
    for path, contents, message in cases:
        path.write_text(contents)
        with pytest.raises(ValueError, match=message):
            module.preflight(request)
        subprocess.run(
            ["git", "-C", str(root), "checkout", "--", "src/gate.py", "models/gate.npz"],
            check=False,
        )
        if path not in {root / "src/gate.py", model}:
            path.unlink()
    external = tmp_path / "external.jsonl"
    external.write_text("no")
    with pytest.raises(ValueError, match="tracked repository input"):
        module.preflight(module.plan(root, external, model, 20, 7))
    untracked_model = root / "models/untracked.npz"
    untracked_model.write_text("no")
    with pytest.raises(ValueError, match="tracked repository expected model"):
        module.preflight(module.plan(root, sample, untracked_model, 20, 7))


def test_preflight_requires_a_full_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, sample, model = repo(tmp_path)
    module, request = script(), script().plan(root, sample, model, 20, 7)
    original = module.git

    def abbreviated(root: Path, *command: str) -> SimpleNamespace:
        return SimpleNamespace(
            returncode=0,
            stdout="short" if command == ("rev-parse", "HEAD") else original(root, *command).stdout,
        )

    monkeypatch.setattr(module, "git", abbreviated)
    with pytest.raises(RuntimeError, match="full commit"):
        module.preflight(request)


@pytest.mark.parametrize(
    "change",
    [
        {"input_sha256": "f" * 64},
        {"model_sha256": "nope"},
        {"duration_seconds": float("inf")},
        {"loss_metrics": {}},
    ],
)
def test_receipt_rejects_invalid_remote_values(tmp_path: Path, change: dict[str, object]) -> None:
    root, sample, model = repo(tmp_path)
    module, request = script(), script().plan(root, sample, model, 20, 7)
    with pytest.raises(ValueError):
        module.receipt(request, "a" * 40, remote(request, **change))


def test_mismatch_writes_atomic_separate_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, sample, model = repo(tmp_path)
    module = script()
    monkeypatch.setattr(
        module, "run_remote", lambda request: remote(request, model_sha256="0" * 64)
    )
    output = root / "artifacts/receipt.json"
    assert (
        module.main(
            [
                "--run",
                "--input",
                str(sample),
                "--expected-model",
                str(model),
                "--output-receipt",
                str(output),
            ],
            root=root,
        )
        == 1
    )
    assert (
        (saved := json.loads(output.read_text()))["parity"] is False
        and saved["schema_version"] == 1
        and saved["limitations"]
        and saved["generated_at"].endswith("Z")
    )
    request = module.plan(root, sample, model, 20, 7)
    with pytest.raises(ValueError, match="separate"):
        module.write_receipt(model, request, {})
