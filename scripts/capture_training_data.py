#!/usr/bin/env python3
"""Create bounded immutable public live or historical data requested by the local UI."""

import argparse
import asyncio
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_gate.training_data import capture_dataset  # noqa: E402


async def run(args):
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    for number in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(number, task.cancel)
    try:
        # Install stop handling before importing the recorder's numeric dependencies.
        history = None
        if args.mode == "historical":
            from market_gate.historical import fetch_history, validate_history_request

            history = validate_history_request(args.symbol, args.start, args.end)

            async def record_events(output, symbol, *limits, progress_callback=None):
                return await fetch_history(
                    output,
                    symbol,
                    history["start"],
                    history["end"],
                    progress_callback=progress_callback,
                )
        else:
            from record_binance import record_events

        await capture_dataset(
            ROOT, args.id, args.seconds, record_events, lock_fd=args.lock_fd, history=history
        )
    finally:
        for number in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(number)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("live", "historical"), default="live")
    parser.add_argument("--symbol")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--id", required=True)
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--lock-fd", type=int, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
