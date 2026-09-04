"""Train the small gate from one recorded public Binance sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from market_gate.training import train_recording


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=root / "data" / "binance-btcusdt-sample.jsonl"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lookback-frames", type=int, default=30)
    parser.add_argument("--horizon-frames", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = train_recording(
        args.input,
        args.output,
        args.receipt,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        lookback_frames=args.lookback_frames,
        horizon_frames=args.horizon_frames,
    )
    print(json.dumps(receipt["training"], indent=2))


if __name__ == "__main__":
    main()
