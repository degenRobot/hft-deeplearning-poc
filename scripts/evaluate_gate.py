"""Print fixed-baseline evaluation of an existing offline gate without modifying files."""

import argparse
import json
from pathlib import Path

from market_gate.evaluation import evaluate_recording


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--horizon-frames", type=int, default=5)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_recording(args.input, args.model, horizon_frames=args.horizon_frames), indent=2
        )
    )


if __name__ == "__main__":
    main()
