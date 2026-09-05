"""The independent verifier reconstructs nonzero clipped runtime updates exactly."""

import io
import json
import runpy
import shutil
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from market_gate.config import LabConfig
from market_gate.engine import MarketEngine
from market_gate.experts import EXPERT_IDS
from market_gate.live_learning import LiveLearner

ROOT = Path(__file__).parents[1]


def test_verifier_preserves_runtime_precision_for_nonzero_clipped_updates(tmp_path, monkeypatch):
    monkeypatch.setattr("market_gate.live_learning.time.time", lambda: 70.0)
    engine = MarketEngine(LabConfig(source="binance"), ROOT / "models/gate-demo.npz")
    engine.feed_status = "running"
    engine.book_valid = True
    engine.run_id = "generated-verifier-test"
    engine.frames = deque(np.random.default_rng(21).normal(0, 2, (30, 10)), maxlen=30)
    engine.decision = SimpleNamespace(scores=dict.fromkeys(EXPERT_IDS, 1.0))
    learner = LiveLearner(engine)
    initial_hash = learner.status()["head_sha256"]
    learner.configure({"enabled": True})

    def advance(now, mid):
        engine.frame_times = deque(range(now - 29_000, now + 1, 1000), maxlen=30)
        engine.last_receive_ts_ms = engine.last_book_receive_ts_ms = now
        engine.last_book_event_ts_ms = now
        engine.bid, engine.ask = mid - 0.01, mid + 0.01
        learner.tick(now)

    for index in range(3):
        selected = 100_000 + index * 10_000
        advance(selected, 100.0)
        sample = learner.pending
        assert sample is not None
        if index == 0:
            gradient = sample["before"].copy()
            gradient[sample["action"]] -= 1
            # The first reward saturates at +5: advantage=1 and clipping must run.
            norm = np.sqrt(np.sum(np.outer(sample["h2"], gradient) ** 2) + np.sum(gradient**2))
            assert norm > 1
        advance(selected + 5000, 101.0)
        assert learner.updates == index + 1
    snapshot = learner.training_snapshot()
    assert snapshot["learning"]["head_sha256"] != initial_hash
    assert all(step["layers"][2]["weight_delta_norm"] > 0 for step in snapshot["history"])
    payloads = {
        "/learning/training": snapshot,
        "/health": {
            "source": "binance",
            "symbol": "BTCUSDT",
            "health": {"ready": True, "events_processed": 0, "book_age_ms": 0},
            "gate": {"model_version": engine.gate.model_version},
        },
    }

    def urlopen(url, timeout):
        assert timeout == 5
        return io.BytesIO(json.dumps(payloads[url.removeprefix("http://127.0.0.1:8001")]).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    (tmp_path / "models").mkdir()
    shutil.copyfile(ROOT / "models/gate-demo.npz", tmp_path / "models/gate-demo.npz")
    receipt_path = tmp_path / "artifacts/training-example/live-rl-validation.json"
    receipt_path.parent.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(ROOT / "scripts/verify_live_learning.py"), run_name="__main__")
    receipt = json.loads(receipt_path.read_text())
    assert receipt["verified_steps"] == 3
    assert receipt["base_head_sha256"] == initial_hash
    assert receipt["adapted_head_sha256"] == learner.status()["head_sha256"]
    assert receipt["independent_head_reconstruction"] == "byte_identical"
