"""A trained 41-parameter shadow expert with NumPy-only runtime inference."""

from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np

from .experts import EXPERT_IDS

DEFAULT_TINY_EXPERT_PATH = Path(__file__).resolve().parents[2] / "models/tiny-expert-demo.npz"
SCHEMA = "tiny-expert-npz-v1"


class TinyExpert:
    """Bounded rule scores → 8 tanh neurons → one tanh direction score.

    This observer has no authority over weights, quotes or risk decisions.
    """

    parameter_count = 41

    def __init__(self, path: str | Path):
        path = Path(path)
        if path.stat().st_size > 65_536:
            raise ValueError("tiny expert artifact exceeds 64 KiB")
        with ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist()) > 65_536:
                raise ValueError("tiny expert arrays exceed 64 KiB")
        with np.load(path, allow_pickle=False) as artifact:
            if artifact["schema_version"].item() != SCHEMA:
                raise ValueError("unsupported tiny expert schema")
            if artifact["input_order"].tolist() != list(EXPERT_IDS):
                raise ValueError("tiny expert input order mismatch")
            self.model_version = str(artifact["model_version"].item())
            if not self.model_version or len(self.model_version) > 100:
                raise ValueError("invalid tiny expert version")
            for name, shape in (("w1", (3, 8)), ("b1", (8,)), ("w2", (8, 1)), ("b2", (1,))):
                value = artifact[name]
                if value.shape != shape or value.dtype.kind != "f" or not np.isfinite(value).all():
                    raise ValueError("invalid tiny expert weights")
                setattr(self, name, value.astype(np.float64))

    def predict(self, inputs: list[float]) -> float:
        values = np.asarray(inputs, dtype=np.float64)
        if values.shape != (3,) or not np.isfinite(values).all() or (np.abs(values) > 1).any():
            raise ValueError("tiny expert requires three finite rule scores in [-1, 1]")
        with np.errstate(over="raise", invalid="raise"):
            hidden = np.tanh(values @ self.w1 + self.b1)
            score = float(np.tanh(hidden @ self.w2 + self.b2).item())
        if not np.isfinite(score):
            raise ValueError("tiny expert output must be finite")
        return score


def load_tiny_expert(path: str | Path | None = DEFAULT_TINY_EXPERT_PATH) -> TinyExpert | None:
    if path is None:
        return None
    try:
        return TinyExpert(path)
    except (OSError, ValueError, KeyError, EOFError, BadZipFile):
        return None
