#!/usr/bin/env python3
"""Capture a bounded immutable public BTCUSDT dataset requested by the local UI."""

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
        from record_binance import record_events

        await capture_dataset(ROOT, args.id, args.seconds, record_events, lock_fd=args.lock_fd)
    finally:
        for number in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(number)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True)
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--lock-fd", type=int, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
