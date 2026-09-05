from __future__ import annotations

import builtins
import importlib.util
import json
import subprocess
import sys
from contextlib import nullcontext
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
        "environment": {
            "python": "3.12.11",
            "numpy": "2.5.2",
            "torch": "2.14.0",
            "platform": "Linux-test",
            "machine": "x86_64",
            "torch_threads": 2,
            "torch_interop_threads": 2,
            "deterministic_algorithms": False,
        },
        "execution": {"app_id": "ap-test", "function_call_id": "fc-test"},
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
    assert display["remote"] == {
        "cpu": (2, 2),
        "memory": (2048, 2048),
        "timeout": 300,
        "retries": 0,
    }
    assert "not a billing cap" in " ".join(display["limitations"])


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
    module.run_remote = lambda _: pytest.fail("remote called before preflight completed")
    launch = ["--run", "--input", str(sample), "--expected-model", str(model)]
    cases = [
        (root / "src/gate.py", "dirty", "every tracked"),
        (model, "changed", "every tracked"),
        (root / "src/private.env", "secret", "only files tracked"),
        (root / "src/scratch.py", "scratch", "only files tracked"),
    ]
    for path, contents, message in cases:
        path.write_text(contents)
        with pytest.raises(ValueError, match=message):
            module.main(launch, root=root)
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


@pytest.mark.parametrize(
    "destination",
    ["data/sample.jsonl", "models/gate.npz", "src/new.json", ".git/new.json", "existing.json"],
)
def test_invalid_output_is_rejected_before_remote(tmp_path: Path, destination: str) -> None:
    root, sample, model = repo(tmp_path)
    (root / "existing.json").write_text("previous evidence")
    module = script()
    module.run_remote = lambda _: pytest.fail("remote called with invalid output")
    with pytest.raises(ValueError):
        module.main(
            [
                "--run",
                "--input",
                str(sample),
                "--expected-model",
                str(model),
                "--output-receipt",
                str(root / destination),
            ],
            root=root,
        )
    assert (root / "existing.json").read_text() == "previous evidence"


def test_unwritable_output_is_rejected_before_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, sample, model = repo(tmp_path)
    module = script()
    module.run_remote = lambda _: pytest.fail("remote called before writable probe")

    def denied(**kwargs):
        raise PermissionError("destination denied")

    monkeypatch.setattr(module.tempfile, "TemporaryFile", denied)
    with pytest.raises(PermissionError, match="destination denied"):
        module.main(["--run", "--input", str(sample), "--expected-model", str(model)], root=root)


def test_dangling_symlink_and_tracked_missing_output_are_rejected(tmp_path: Path) -> None:
    root, sample, model = repo(tmp_path)
    module, request = script(), script().plan(root, sample, model, 20, 7)
    symlink = root / "link.json"
    symlink.symlink_to(root / "absent.json")
    with pytest.raises(ValueError, match="overwrite"):
        module.output_preflight(symlink, request)
    (root / ".gitignore").unlink()
    with pytest.raises(ValueError, match="tracked"):
        module.output_preflight(root / ".gitignore", request)


def test_atomic_publication_cannot_overwrite_a_racing_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, sample, model = repo(tmp_path)
    module, request = script(), script().plan(root, sample, model, 20, 7)
    output = root / "receipt.json"
    link = module.os.link

    def racing_link(source, destination):
        output.write_text("other receipt")
        link(source, destination)

    monkeypatch.setattr(module.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        module.write_receipt(output, request, {"complete": True})
    assert output.read_text() == "other receipt"
    assert not list(root.glob(".receipt.json.*"))


@pytest.mark.parametrize(
    "section,key,bad",
    [
        ("environment", "python", ""),
        ("environment", "numpy", []),
        ("environment", "platform", "host\nsecret"),
        ("environment", "torch_threads", True),
        ("environment", "torch_interop_threads", 0),
        ("environment", "deterministic_algorithms", 1),
        ("execution", "app_id", "https://private.invalid/token"),
        ("execution", "function_call_id", 4),
    ],
)
def test_receipt_rejects_malformed_metadata(
    tmp_path: Path, section: str, key: str, bad: object
) -> None:
    root, sample, model = repo(tmp_path)
    module, request = script(), script().plan(root, sample, model, 20, 7)
    response = remote(request)
    response[section][key] = bad
    with pytest.raises(ValueError):
        module.receipt(request, "a" * 40, response)


def test_identifiers_can_be_explicitly_unavailable_but_metadata_cannot_disappear(
    tmp_path: Path,
) -> None:
    root, sample, model = repo(tmp_path)
    module, request = script(), script().plan(root, sample, model, 20, 7)
    response = remote(request, execution={"app_id": None, "function_call_id": None})
    saved = module.receipt(request, "a" * 40, response)
    assert saved["execution"] == {"app_id": None, "function_call_id": None}
    assert saved["environment"] == response["environment"]
    assert saved["parity"] is True
    for section in ("environment", "execution"):
        with pytest.raises(ValueError):
            module.receipt(request, "a" * 40, remote(request, **{section: None}))


def test_modal_invocation_uses_limits_and_observed_app_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, sample, model = repo(tmp_path)
    module, request = script(), script().plan(root, sample, model, 20, 7)
    observed = {}

    class Image:
        @classmethod
        def debian_slim(cls, **kwargs):
            return cls()

        def pip_install(self, *packages):
            observed["packages"] = packages
            return self

        def add_local_dir(self, path, **kwargs):
            observed["source"] = path
            return self

    class App:
        app_id = "ap-observed"

        def __init__(self, name):
            pass

        def function(self, **kwargs):
            observed.update(kwargs)
            return lambda function: SimpleNamespace(remote=lambda *args: remote(request))

        def run(self):
            return nullcontext()

    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(Image=Image, App=App))
    result = module.run_remote(request)
    assert observed["cpu"] == (2, 2)
    assert observed["memory"] == (2048, 2048)
    assert observed["timeout"] == 300 and observed["retries"] == 0
    assert observed["source"] == root / "src"
    assert observed["packages"] == ("numpy==2.5.2", "torch==2.14.0")
    assert result["execution"]["app_id"] == "ap-observed"


def test_remote_collects_actual_environment_and_call_id(monkeypatch: pytest.MonkeyPatch) -> None:
    module = script()
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setitem(
        sys.modules, "modal", SimpleNamespace(current_function_call_id=lambda: None)
    )
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(__version__="observed-numpy"))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="observed-torch",
            get_num_threads=lambda: 3,
            get_num_interop_threads=lambda: 4,
            are_deterministic_algorithms_enabled=lambda: True,
        ),
    )

    def training(recording, model, receipt, **kwargs):
        assert recording.read_bytes() == b"sample"
        model.write_bytes(b"ephemeral model")
        return {
            "training": {"first_train_loss": -0.1, "last_train_loss": -0.2, "validation_loss": -0.3}
        }

    monkeypatch.setitem(
        sys.modules, "market_gate.training", SimpleNamespace(train_recording=training)
    )
    response = module._train_remote(b"sample", module.hashlib.sha256(b"sample").hexdigest(), 20, 7)
    assert response["environment"]["numpy"] == "observed-numpy"
    assert response["environment"]["torch"] == "observed-torch"
    assert response["environment"]["torch_threads"] == 3
    assert response["environment"]["torch_interop_threads"] == 4
    assert response["environment"]["deterministic_algorithms"] is True
    assert response["environment"]["python"] and response["environment"]["platform"]
    assert response["execution"] == {"function_call_id": None}
