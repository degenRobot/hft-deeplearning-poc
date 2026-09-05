#!/usr/bin/env python3
"""Run the bounded training example locally, publishing actual optimizer telemetry."""

import argparse
import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from market_gate.training_service import SnapshotWriter  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/training-public.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live-state", type=Path, default=Path("artifacts/training-live.json"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--max-rl-steps", type=int, default=120)
    parser.add_argument("--pace", type=float, default=0)
    args = parser.parse_args()
    if not 0 <= args.pace <= 1:
        parser.error("pace must be between 0 and 1 seconds")
    writer = SnapshotWriter(args.live_state, args.output.name, "local")

    def stopped(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stopped)
    try:
        from market_gate.training_lab import train_lab

        for event in train_lab(
            args.input, args.output, epochs=args.epochs, max_rl_steps=args.max_rl_steps
        ):
            writer.accept(event)
            if event["kind"] == "step":
                time.sleep(args.pace)
        print(
            json.dumps({"status": writer.state["status"], "evaluation": writer.state["evaluation"]})
        )
    except KeyboardInterrupt:
        writer.finish("stopped")
    except Exception as error:
        writer.finish("failed", "Training failed; see the local run log for details")
        raise error
    finally:
        writer.close()


if __name__ == "__main__":
    main()
