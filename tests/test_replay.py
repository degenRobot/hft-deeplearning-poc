import json
from pathlib import Path

from market_gate.config import LabConfig
from market_gate.engine import MarketEngine
from market_gate.feeds.replay import ReplayFeed

FIXTURE = Path(__file__).parents[1] / "fixtures" / "replay.jsonl"


def run_replay() -> str:
    engine = MarketEngine(LabConfig())
    for event in ReplayFeed(FIXTURE):
        engine.process(event)
    return json.dumps(engine.ledger_rows(), sort_keys=True, separators=(",", ":"))


def test_replay_decisions_are_byte_stable() -> None:
    assert run_replay() == run_replay()
