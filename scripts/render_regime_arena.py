"""Re-render replay.html from an existing replay.json + summary.json (optional)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from market_gate.regime_arena import ArenaConfig, StepRecord, write_replay_html


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args()
    replay_path = args.artifact_dir / "replay.json"
    summary_path = args.artifact_dir / "summary.json"
    records = []
    for row in json.loads(replay_path.read_text())["steps"]:
        records.append(
            StepRecord(
                t=row["t"],
                regime=row["regime"],
                agent=tuple(row["agent"]),
                weights=row["weights"],
                policy="learned",
                reward=row["reward"],
                expert_cf=row["expert_cf"],
                features=[],
            )
        )
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    write_replay_html(args.artifact_dir / "replay.html", records, ArenaConfig(), summary)
    print("wrote", args.artifact_dir / "replay.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())