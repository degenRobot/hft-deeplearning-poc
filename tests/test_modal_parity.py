from __future__ import annotations

import hashlib
import importlib.util
import subprocess
from pathlib import Path

import pytest


def load_script():
    path = Path(__file__).parents[1] / "scripts" / "train_on_modal.py"
    spec = importlib.util.spec_from_file_location("modal_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True)
    (root / "models").mkdir()
    (root / "src").mkdir()
    source, sample, model = (
        root / "src" / "gate.py",
        root / "data" / "sample.jsonl",
        root / "models" / "gate.npz",
    )
    source.write_text("safe = True\n")
    sample.write_bytes(b"public sample")
    model.write_bytes(b"expected model")
    (root / ".gitignore").write_text("src/private.env\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True
    )
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
    return root, sample, model


def test_dry_run_never_calls_remote_or_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, sample, model = repo(tmp_path)
    module = load_script()
    monkeypatch.setattr(module, "run_remote", lambda *_: pytest.fail("remote called"))
    assert module.main(["--input", str(sample), "--expected-model", str(model)], root=root) == 0
    assert not (root / "artifacts" / "modal-parity.json").exists()


def test_run_rejects_untracked_source_uploads(tmp_path: Path) -> None:
    root, sample, model = repo(tmp_path)
    module = load_script()
    request = module.plan(root, sample, model, 20, 7)
    module.require_clean_head_inputs(request)
    (root / "src" / "private.env").write_text("secret")
    with pytest.raises(ValueError, match="only files tracked"):
        module.require_clean_head_inputs(request)
    (root / "src" / "private.env").unlink()
    (root / "src" / "scratch.py").write_text("scratch")
    with pytest.raises(ValueError, match="only files tracked"):
        module.require_clean_head_inputs(request)


def test_run_writes_a_mismatch_receipt_and_checks_remote_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, sample, model = repo(tmp_path)
    module = load_script()
    remote = {
        "input_sha256": hashlib.sha256(b"public sample").hexdigest(),
        "model_sha256": "0" * 64,
        "loss_metrics": {},
        "duration_seconds": 0,
    }
    monkeypatch.setattr(module, "run_remote", lambda *_: remote)
    output = root / "artifacts" / "receipt.json"
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
    assert '"parity": false' in output.read_text()
    remote["input_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="remote input hash"):
        module.main(["--run", "--input", str(sample), "--expected-model", str(model)], root=root)
