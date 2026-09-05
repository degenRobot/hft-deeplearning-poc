"""Offline cloud-package checks: no credentials, imports or compute on Modal."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def test_exact_modal_source_package_imports_without_web_runtime(tmp_path):
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location(
        "training_cloud", root / "scripts/train_lab_on_modal.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    package = tmp_path / "market_gate"
    package.mkdir()
    for name in module.SOURCE_FILES:
        (package / name).write_bytes((root / "src/market_gate" / name).read_bytes())
    script = (
        "import sys; from market_gate.training_lab import train_lab; "
        "assert 'market_gate.api' not in sys.modules; "
        "assert 'fastapi' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=os.environ | {"PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert set(p.name for p in package.iterdir() if p.is_file()) == set(module.SOURCE_FILES)
