"""Download a bounded public candle sample for approximate offline training."""

import argparse
import asyncio
import json
from pathlib import Path

from market_gate.historical import SUPPORTED_SYMBOLS, fetch_history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", choices=SUPPORTED_SYMBOLS, required=True)
    parser.add_argument("--start", required=True, help="Inclusive ISO UTC timestamp, whole seconds")
    parser.add_argument("--end", required=True, help="Exclusive ISO UTC timestamp, whole seconds")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = asyncio.run(fetch_history(args.output, args.symbol, args.start, args.end))
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
