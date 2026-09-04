"""Small, reproducible helpers for the recorded-data gate training demo.

The live application never calls this module. It is deliberately a separate,
offline path: record a short public sample, train locally, then let the API
serve the resulting receipt.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .contracts import BookEvent, TradeEvent
from .experts import EXPERT_IDS, microprice_pressure, short_reversion, trade_flow_impulse
from .features import FeatureBuilder

FEATURE_NAMES = [
    "return_1s",
    "return_5s",
    "realized_volatility",
    "spread_bps",
    "book_imbalance",
    "microprice_displacement",
    "signed_flow_ratio",
    "trade_arrivals",
    "quote_updates",
    "fair_value_distance",
]
EXPERT_NAMES = EXPERT_IDS


@dataclass(frozen=True)
class TrainingExample:
    """One causal feature window and labels that start after the window ends."""

    start_frame: int
    target_frame: int
    features: np.ndarray
    utilities: np.ndarray
    start_ts_ms: int = 0
    input_end_ts_ms: int = 0
    target_ts_ms: int = 0


@dataclass(frozen=True)
class FrameDataset:
    values: np.ndarray
    mids: np.ndarray
    close_ts_ms: np.ndarray


def event_to_record(event: BookEvent | TradeEvent) -> dict[str, object]:
    """Return a stable, envelope-free JSONL record for a public market event."""
    return asdict(event) | {"kind": "book" if isinstance(event, BookEvent) else "trade"}


def _record_to_event(raw: dict[str, object]) -> BookEvent | TradeEvent:
    if not isinstance(raw, dict):
        raise ValueError("record must be an object")

    def number(key: str, *, integer: bool = False, positive: bool = False) -> int | float:
        value = raw[key]
        valid_type = type(value) is int if integer else type(value) in (int, float)
        if not valid_type or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite {'integer' if integer else 'number'}")
        if value < 0 or (positive and value == 0):
            raise ValueError(f"{key} must be {'positive' if positive else 'nonnegative'}")
        return int(value) if integer else float(value)

    for key in ("venue", "symbol"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    common = {
        "venue": raw["venue"],
        "symbol": raw["symbol"],
        "event_ts_ms": number("event_ts_ms", integer=True),
        "receive_ts_ms": number("receive_ts_ms", integer=True),
    }
    kind = raw["kind"]
    if kind == "book":
        bid, ask = number("bid_price", positive=True), number("ask_price", positive=True)
        if bid > ask:
            raise ValueError("book quotes must not be crossed")
        return BookEvent(
            **common,
            update_id=number("update_id", integer=True),
            bid_price=bid,
            ask_price=ask,
            bid_size=number("bid_size"),
            ask_size=number("ask_size"),
        )
    if kind != "trade":
        raise ValueError("kind must be book or trade")
    aggressor = raw["aggressor"]
    if aggressor not in ("buy", "sell"):
        raise ValueError("aggressor must be buy or sell")
    return TradeEvent(
        **common,
        trade_id=number("trade_id", integer=True),
        price=number("price", positive=True),
        size=number("size"),
        aggressor=aggressor,
    )


def should_keep_book(event_ts_ms: int, last_kept_ts_ms: int | None, interval_ms: int) -> bool:
    """Downsample book updates while preserving the first update and all trades."""
    return last_kept_ts_ms is None or event_ts_ms - last_kept_ts_ms >= interval_ms


def write_recording(
    path: Path, events: list[BookEvent | TradeEvent], book_interval_ms: int
) -> dict[str, int]:
    """Write normalized public events, downsampling only book updates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"book": 0, "trade": 0, "total": 0}
    last_book_ts_ms: int | None = None
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            if isinstance(event, BookEvent):
                if not should_keep_book(event.event_ts_ms, last_book_ts_ms, book_interval_ms):
                    continue
                last_book_ts_ms = event.event_ts_ms
            counts["book" if isinstance(event, BookEvent) else "trade"] += 1
            handle.write(json.dumps(event_to_record(event), sort_keys=True, separators=(",", ":")))
            handle.write("\n")
            counts["total"] += 1
    return counts


def load_recording(path: Path) -> list[BookEvent | TradeEvent]:
    """Read a normalized JSONL recording and reject malformed public records."""
    if not path.is_file():
        raise ValueError(f"recording does not exist: {path}")

    events: list[BookEvent | TradeEvent] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            events.append(_record_to_event(json.loads(line)))
        except (KeyError, TypeError, ValueError, OverflowError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid recording line {line_number}: {error}") from error

    if not events:
        raise ValueError(f"recording is empty: {path}")
    if len({(event.venue, event.symbol) for event in events}) != 1:
        raise ValueError("recording must contain one venue and symbol")
    return sorted(events, key=lambda e: (e.event_ts_ms, e.receive_ts_ms, type(e).__name__))


def book_metrics(book: BookEvent) -> tuple[float, float, float, float]:
    """Calculate the four quote-derived inputs used when a second is closed."""
    mid = (book.bid_price + book.ask_price) / 2
    spread_bps = (book.ask_price - book.bid_price) / mid * 10_000 if mid else 0.0
    size_total = book.bid_size + book.ask_size
    if not size_total:
        return mid, spread_bps, 0.0, mid
    imbalance = (book.bid_size - book.ask_size) / size_total
    microprice = (book.ask_price * book.bid_size + book.bid_price * book.ask_size) / size_total
    return mid, spread_bps, imbalance, microprice


def build_frame_dataset(events: list[BookEvent | TradeEvent]) -> FrameDataset:
    """Close observed event-time seconds; retain gaps and discard the unclosed final second."""
    builder = FeatureBuilder()
    last_book: BookEvent | None = None
    values: list[tuple[float, ...]] = []
    mids: list[float] = []
    timestamps: list[int] = []

    for event in events:
        prior_second = builder.current_second
        if last_book is not None:
            mid, spread_bps, imbalance, microprice = book_metrics(last_book)
            frame = builder.close_before(event.event_ts_ms, mid, spread_bps, imbalance, microprice)
            if frame is not None:
                values.append(frame.values)
                mids.append(mid)
                timestamps.append(frame.close_ts_ms)
        if prior_second is not None and event.event_ts_ms // 1000 - prior_second > 1:
            # Start a fresh history segment; pre-gap returns/fair values and quotes
            # must not be inherited as though the absent seconds were observed.
            builder = FeatureBuilder()
            last_book = None
        if last_book is None and isinstance(event, TradeEvent):
            # A later book must never be used to close an earlier trade-only second.
            # Initial and post-gap trades are discarded until a fresh book arrives.
            continue
        builder.begin_second(event.event_ts_ms)
        if isinstance(event, BookEvent):
            last_book = event
            builder.observe_book()
        else:
            builder.observe_trade(event.size, event.aggressor)

    if len(values) < 2:
        raise ValueError("recording needs quote updates across at least two seconds")
    return FrameDataset(
        np.asarray(values, dtype=np.float32),
        np.asarray(mids, dtype=np.float64),
        np.asarray(timestamps, dtype=np.int64),
    )


def build_frames(events: list[BookEvent | TradeEvent]) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility helper; use build_frame_dataset for timestamp-aware experiments."""
    dataset = build_frame_dataset(events)
    return dataset.values, dataset.mids


def build_examples(
    frames: np.ndarray,
    mids: np.ndarray,
    lookback_frames: int,
    horizon_frames: int,
    *,
    close_ts_ms: np.ndarray | None = None,
) -> list[TrainingExample]:
    """Create causal windows; each utility uses a future mid after the input window."""
    if (
        type(lookback_frames) is not int
        or type(horizon_frames) is not int
        or lookback_frames < 1
        or horizon_frames < 1
    ):
        raise ValueError("lookback_frames and horizon_frames must be positive")
    if len(frames) != len(mids):
        raise ValueError("frame and mid counts differ")

    if not np.all(np.isfinite(frames)) or not np.all(np.isfinite(mids)) or np.any(mids <= 0):
        raise ValueError("frames and positive mids must be finite")
    if frames.ndim != 2 or frames.shape[1] != len(FEATURE_NAMES):
        raise ValueError("frames must have ten features")
    # Synthetic callers may use an explicitly regular ordinal clock. Recorded-data
    # training and evaluation always pass the actual closed-frame timestamps.
    timestamps = (
        np.arange(len(frames), dtype=np.int64) * 1000 if close_ts_ms is None else close_ts_ms
    )
    if len(timestamps) != len(frames) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("frame timestamps must match frames and strictly increase")
    examples: list[TrainingExample] = []
    for start in range(len(frames) - lookback_frames - horizon_frames + 1):
        end = start + lookback_frames
        target = end + horizon_frames - 1
        if np.any(np.diff(timestamps[start : target + 1]) != 1000):
            continue
        future_return_bps = float((mids[target] / mids[end - 1] - 1.0) * 10_000)
        last = frames[end - 1]
        # Approximate the three runtime experts from one closed feature frame,
        # then reward a proxy score only when it agrees with the later move.
        proxy_scores = np.asarray(
            [
                microprice_pressure(1.0, float(last[4]), 1.0 + float(last[5])),
                trade_flow_impulse(float(last[6]), 1.0, int(last[7])),
                short_reversion(1.0 + float(last[9]), 1.0),
            ],
            dtype=np.float32,
        )
        utilities = proxy_scores * future_return_bps
        examples.append(
            TrainingExample(
                start,
                target,
                frames[start:end].reshape(-1),
                utilities,
                int(timestamps[start]),
                int(timestamps[end - 1]),
                int(timestamps[target]),
            )
        )
    if not examples:
        raise ValueError(
            "insufficient contiguous frames: need lookback_frames + horizon_frames "
            f"({lookback_frames + horizon_frames})"
        )
    return examples


def chronological_split(
    examples: list[TrainingExample], frame_count: int, validation_fraction: float = 0.2
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """Split on a frame boundary, leaving an embargo so the two sets never overlap."""
    if not examples:
        raise ValueError("no examples available for chronological split")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if any(example.target_frame >= frame_count for example in examples):
        raise ValueError("example target is outside the available frames")
    # Freeze the original candidate-start boundary before excluding gapped windows.
    # Rejected data must never move held-out observations back into training time.
    span = examples[0].target_frame - examples[0].start_frame + 1
    if any(e.target_frame - e.start_frame + 1 != span for e in examples):
        raise ValueError("examples must share one lookback and horizon")
    boundary = int((frame_count - span + 1) * (1 - validation_fraction))
    train = [example for example in examples if example.target_frame < boundary]
    validation = [example for example in examples if example.start_frame >= boundary]
    if not train or not validation:
        raise ValueError(
            "insufficient frames for a chronological train/validation split; record a longer sample"
        )
    return train, validation


def _array(examples: list[TrainingExample]) -> tuple[np.ndarray, np.ndarray]:
    return np.stack([e.features for e in examples]), np.stack([e.utilities for e in examples])


def _save_gate_model(model: object, destination: Path) -> int:
    """Export the small Torch MLP to the portable NPZ format used by the runtime."""
    from torch import nn

    destination.parent.mkdir(parents=True, exist_ok=True)
    layers = [layer for layer in model if isinstance(layer, nn.Linear)]  # type: ignore[union-attr]
    if len(layers) != 3:
        raise ValueError("expected three linear gate layers")
    first, second, third = layers
    with destination.open("xb") as handle:
        np.savez(
            handle,
            schema_version=np.array("gate-npz-v1"),
            w1=first.weight.detach().cpu().numpy().T,
            b1=first.bias.detach().cpu().numpy(),
            w2=second.weight.detach().cpu().numpy().T,
            b2=second.bias.detach().cpu().numpy(),
            w3=third.weight.detach().cpu().numpy().T,
            b3=third.bias.detach().cpu().numpy(),
        )
    return int(sum(parameter.numel() for parameter in model.parameters()))  # type: ignore[union-attr]


def _utc_timestamp(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).isoformat().replace("+00:00", "Z")


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return f"<external>/{path.name}"


def train_recording(
    recording_path: Path,
    model_path: Path,
    receipt_path: Path,
    *,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    seed: int = 7,
    lookback_frames: int = 30,
    horizon_frames: int = 5,
) -> dict[str, object]:
    """Train the illustrative gate from a recording and write a transparent receipt."""
    if lookback_frames != 30:
        raise ValueError("runtime gate export requires lookback_frames=30 (30 x 10 features)")
    if len({path.resolve() for path in (recording_path, model_path, receipt_path)}) != 3:
        raise ValueError("recording, model and receipt paths must be separate")
    if model_path.exists() or receipt_path.exists():
        raise ValueError("model and receipt outputs must be new files; refusing to overwrite")
    if model_path.suffix != ".npz":
        raise ValueError("model output must have a .npz suffix")
    if (
        type(epochs) is not int
        or epochs < 1
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("epochs and learning_rate must be positive")
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError("install the training extra before running this script") from error

    torch.manual_seed(seed)
    np.random.seed(seed)
    events = load_recording(recording_path)
    dataset = build_frame_dataset(events)
    frames, mids = dataset.values, dataset.mids
    examples = build_examples(
        frames,
        mids,
        lookback_frames,
        horizon_frames,
        close_ts_ms=dataset.close_ts_ms,
    )
    train_examples, validation_examples = chronological_split(examples, len(frames))
    x_train, y_train = _array(train_examples)
    x_validation, y_validation = _array(validation_examples)

    model = nn.Sequential(
        nn.Linear(x_train.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, len(EXPERT_NAMES)),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_features = torch.from_numpy(x_train)
    train_utilities = torch.from_numpy(y_train)
    validation_features = torch.from_numpy(x_validation)
    validation_utilities = torch.from_numpy(y_validation)

    losses: list[float] = []
    for _ in range(epochs):
        weights = torch.softmax(model(train_features), dim=1)
        loss = -(weights * train_utilities).sum(dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        validation_weights = torch.softmax(model(validation_features), dim=1)
        validation_loss = float(
            (-(validation_weights * validation_utilities).sum(dim=1).mean()).detach()
        )

    if not np.all(np.isfinite(validation_weights.numpy())) or not math.isfinite(validation_loss):
        raise ValueError("training produced non-finite weights or loss; no model exported")
    parameter_count = _save_gate_model(model, model_path)
    counts = {
        "book": sum(isinstance(event, BookEvent) for event in events),
        "trade": sum(isinstance(event, TradeEvent) for event in events),
        "total": len(events),
    }
    root = Path.cwd()
    receipt: dict[str, object] = {
        "schema_version": 2,
        "generated_at": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
        "source": {
            "name": "Binance public WebSocket sample",
            "symbol": events[0].symbol,
            "venue": events[0].venue,
            "url": "https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams",
            "recording_path": _relative_path(recording_path, root),
            "started_at": _utc_timestamp(min(event.event_ts_ms for event in events)),
            "ended_at": _utc_timestamp(max(event.event_ts_ms for event in events)),
            "duration_seconds": round((events[-1].event_ts_ms - events[0].event_ts_ms) / 1000, 3),
            "event_counts": counts,
        },
        "dataset": {
            "feature_names": FEATURE_NAMES,
            "frame_seconds": 1,
            "lookback_frames": lookback_frames,
            "horizon_frames": horizon_frames,
            "frames": len(frames),
            "examples": len(train_examples) + len(validation_examples),
            "train_examples": len(train_examples),
            "validation_examples": len(validation_examples),
        },
        "training": {
            "seed": seed,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "parameter_count": parameter_count,
            "first_train_loss": losses[0],
            "last_train_loss": losses[-1],
            "validation_loss": validation_loss,
            "model_path": _relative_path(model_path, root),
        },
        "limitations": [
            "A short public sample is a teaching artifact, not a trading signal or backtest.",
            "The labels use one-second proxies for the live microprice, flow, and reversion "
            "experts; flow and reversion do not reproduce the live engine's rolling state.",
            "The exported model is an offline example and is not the gate used by the live demo.",
            "Chronological splitting keeps validation later than training; it does not establish "
            "robustness.",
            f"{len(examples) - len(train_examples) - len(validation_examples)} overlapping "
            "boundary examples were excluded from both sets.",
        ],
    }
    from .evaluation import compare_utilities

    receipt["dataset"].update(
        dataset_evidence(
            dataset, examples, train_examples, validation_examples, lookback_frames, horizon_frames
        )
    )
    receipt["evaluation"] = compare_utilities(validation_examples, validation_weights.numpy())
    receipt["provenance"] = provenance(
        recording_path,
        model_path,
        {
            "epochs": epochs,
            "seed": seed,
            "learning_rate": learning_rate,
            "lookback_frames": lookback_frames,
            "horizon_frames": horizon_frames,
            "validation_fraction": 0.2,
            "clock": "event_ts_ms",
            "gap_policy": "exclude_crossing_windows",
            "feature_history": "reset_after_gap",
            "label": "offline_expert_proxy_future_mid_return_bps_v2",
        },
    )
    receipt["limitations"].extend(
        [
            "Offline event-time ordering differs from the live Binance receive-time clock.",
            "Windows crossing absent seconds are excluded; the final open second is not used.",
            "Overlapping held-out windows are not independent or a generalisation test.",
        ]
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, indent=2) + "\n")
    return receipt


def dataset_evidence(dataset, examples, train, validation, lookback_frames, horizon_frames):
    """Report exclusions and split bounds on the actual event-time clock."""
    candidate_count = max(0, len(dataset.values) - lookback_frames - horizon_frames + 1)
    return {
        "clock": "event_ts_ms",
        "gap_policy": "exclude_crossing_windows",
        "feature_history": "reset_after_gap; discard pre-book trades; full 30-frame model input",
        "candidate_examples": candidate_count,
        "gap_excluded_examples": candidate_count - len(examples),
        "boundary_excluded_examples": len(examples) - len(train) - len(validation),
        "missing_seconds": int(np.sum(np.diff(dataset.close_ts_ms) // 1000 - 1)),
        "first_frame_close_ts_ms": int(dataset.close_ts_ms[0]),
        "last_frame_close_ts_ms": int(dataset.close_ts_ms[-1]),
        "train_last_target_ts_ms": max(e.target_ts_ms for e in train),
        "validation_first_start_ts_ms": min(e.start_ts_ms for e in validation),
        "validation_last_target_ts_ms": max(e.target_ts_ms for e in validation),
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(recording_path: Path, model_path: Path, config: dict) -> dict:
    sources = {
        name: file_sha256(Path(__file__).parent / name)
        for name in (
            "training.py",
            "evaluation.py",
            "features.py",
            "experts.py",
            "contracts.py",
            "gate.py",
        )
    }
    packages = {}
    for name in ("numpy", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    return {
        "input_sha256": file_sha256(recording_path),
        "model_sha256": file_sha256(model_path),
        "source_files": sources,
        "source_sha256": digest(sources),
        "config": config,
        "config_sha256": digest(config),
        "packages": packages,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
