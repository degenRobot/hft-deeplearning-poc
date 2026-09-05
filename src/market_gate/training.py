"""Small, reproducible helpers for the recorded-data gate training demo.

The live application never calls this module. It is deliberately a separate,
offline path: record a short public sample, train locally, then let the API
serve the resulting receipt.
"""

from __future__ import annotations

import json
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


def event_to_record(event: BookEvent | TradeEvent) -> dict[str, object]:
    """Return a stable, envelope-free JSONL record for a public market event."""
    return asdict(event) | {"kind": "book" if isinstance(event, BookEvent) else "trade"}


def _record_to_event(raw: dict[str, object]) -> BookEvent | TradeEvent:
    common = {
        "venue": str(raw["venue"]),
        "symbol": str(raw["symbol"]),
        "event_ts_ms": int(raw["event_ts_ms"]),
        "receive_ts_ms": int(raw["receive_ts_ms"]),
    }
    kind = raw["kind"]
    if kind == "book":
        return BookEvent(
            **common,
            update_id=int(raw["update_id"]),
            **{key: float(raw[key]) for key in ("bid_price", "bid_size", "ask_price", "ask_size")},
        )
    if kind != "trade":
        raise ValueError("kind must be book or trade")
    aggressor = str(raw["aggressor"])
    if aggressor not in {"buy", "sell"}:
        raise ValueError("aggressor must be buy or sell")
    return TradeEvent(
        **common,
        trade_id=int(raw["trade_id"]),
        **{key: float(raw[key]) for key in ("price", "size")},
        aggressor=aggressor,  # type: ignore[arg-type]
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
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid recording line {line_number}: {error}") from error

    if not events:
        raise ValueError(f"recording is empty: {path}")
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


def build_frames(events: list[BookEvent | TradeEvent]) -> tuple[np.ndarray, np.ndarray]:
    """Build one-second frames using only observations available at each close."""
    builder = FeatureBuilder()
    last_book: BookEvent | None = None
    values: list[tuple[float, ...]] = []
    mids: list[float] = []

    for event in events:
        if last_book is not None:
            mid, spread_bps, imbalance, microprice = book_metrics(last_book)
            frame = builder.close_before(event.event_ts_ms, mid, spread_bps, imbalance, microprice)
            if frame is not None:
                values.append(frame.values)
                mids.append(mid)
        builder.begin_second(event.event_ts_ms)
        if isinstance(event, BookEvent):
            last_book = event
            builder.observe_book()
        else:
            builder.observe_trade(event.size, event.aggressor)

    if len(values) < 2:
        raise ValueError("recording needs quote updates across at least two seconds")
    return np.asarray(values, dtype=np.float32), np.asarray(mids, dtype=np.float32)


def build_examples(
    frames: np.ndarray, mids: np.ndarray, lookback_frames: int, horizon_frames: int
) -> list[TrainingExample]:
    """Create causal windows; each utility uses a future mid after the input window."""
    if lookback_frames < 1 or horizon_frames < 1:
        raise ValueError("lookback_frames and horizon_frames must be positive")
    if len(frames) != len(mids):
        raise ValueError("frame and mid counts differ")

    examples: list[TrainingExample] = []
    for start in range(len(frames) - lookback_frames - horizon_frames + 1):
        end = start + lookback_frames
        target = end + horizon_frames - 1
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
        examples.append(TrainingExample(start, target, frames[start:end].reshape(-1), utilities))
    if not examples:
        raise ValueError(
            "insufficient frames: need more than lookback_frames + horizon_frames "
            f"({lookback_frames + horizon_frames})"
        )
    return examples


def chronological_split(
    examples: list[TrainingExample], frame_count: int, validation_fraction: float = 0.2
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """Split on a frame boundary, leaving an embargo so the two sets never overlap."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if any(example.target_frame >= frame_count for example in examples):
        raise ValueError("example target is outside the available frames")
    # Choose the boundary from viable input starts, then embargo every window
    # that would touch both sides. A frame-count split alone can leave no room
    # for a complete validation window in a deliberately short recording.
    last_start = max(example.start_frame for example in examples)
    boundary = int((last_start + 1) * (1 - validation_fraction))
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
    np.savez(
        destination,
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
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError("install the training extra before running this script") from error

    torch.manual_seed(seed)
    np.random.seed(seed)
    events = load_recording(recording_path)
    frames, mids = build_frames(events)
    examples = build_examples(frames, mids, lookback_frames, horizon_frames)
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

    parameter_count = _save_gate_model(model, model_path)
    counts = {
        "book": sum(isinstance(event, BookEvent) for event in events),
        "trade": sum(isinstance(event, TradeEvent) for event in events),
        "total": len(events),
    }
    root = Path.cwd()
    receipt: dict[str, object] = {
        "schema_version": 1,
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
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt
