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
    replay = json.loads(replay_path.read_text())
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if summary.get("schema_version") != 2:
        parser.error("unsupported or missing summary schema: only v2 replay artifacts are supported")
    rows = replay.get("steps", [])
    if not rows or any(not row.get("walls") or "pellets" not in row or "predator" not in row for row in rows):
        parser.error("invalid v2 replay: each step requires walls, pellets, and predator")
    grid = len(rows[0]["walls"])
    if any(len(row["walls"]) != grid or any(len(col) != grid for col in row["walls"]) for row in rows):
        parser.error("invalid v2 replay: walls must be consistent square grids")
    records = []
    for row in rows:
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
                walls=row["walls"],
                pellets=row["pellets"],
                predator=row["predator"],
            )
        )
    write_replay_html(args.artifact_dir / "replay.html", records, ArenaConfig(grid=grid), summary)
    print("wrote", args.artifact_dir / "replay.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())