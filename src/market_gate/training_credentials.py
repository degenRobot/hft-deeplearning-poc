"""Local-only dotenv storage. Public status never contains credential values."""

import importlib.util
import os
import re
import subprocess
import tempfile
import threading
from pathlib import Path

KEYS = ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")
TOKEN = re.compile(r"[A-Za-z0-9_-]{8,256}\Z")
_lock = threading.Lock()


def project_env(root: Path) -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    return (Path(result.stdout.strip()).parent if result.returncode == 0 else root) / ".env"


def read_credentials(path: Path) -> dict[str, str]:
    values = {}
    if path.is_symlink():
        raise ValueError("The project .env must be a regular file")
    if path.is_file():
        for line in path.read_text().splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() in KEYS:
                values[key.strip()] = value.strip().strip("\"'")
    return values


def credential_status(path: Path) -> dict:
    values = read_credentials(path)
    return {
        "configured": all(values.get(key) or os.environ.get(key) for key in KEYS),
        "available": importlib.util.find_spec("modal") is not None,
    }


def save_credentials(path: Path, payload: object) -> dict:
    if (
        not isinstance(payload, dict)
        or set(payload) != {"token_id", "token_secret"}
        or any(not isinstance(v, str) or TOKEN.fullmatch(v) is None for v in payload.values())
    ):
        raise ValueError("Enter a valid Modal token ID and secret; whitespace is not allowed")
    with _lock:
        if path.is_symlink():
            raise ValueError("The project .env must be a regular file")
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", ".env"],
            cwd=path.parent,
            capture_output=True,
        )
        if tracked.returncode == 0:
            raise ValueError("Remove .env from Git tracking before saving credentials")
        original = path.read_text() if path.exists() else ""
        lines = [
            line for line in original.splitlines() if line.partition("=")[0].strip() not in KEYS
        ]
        lines.extend(
            [
                f"MODAL_TOKEN_ID={payload['token_id']}",
                f"MODAL_TOKEN_SECRET={payload['token_secret']}",
            ]
        )
        fd, temporary = tempfile.mkstemp(prefix=".modal-env-", dir=path.parent)
        try:
            with os.fdopen(fd, "w") as handle:
                os.fchmod(handle.fileno(), 0o600)
                handle.write("\n".join(lines) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return credential_status(path)
