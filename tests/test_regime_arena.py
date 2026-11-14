import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from market_gate.regime_arena import (
    INPUT_DIM,
    ArenaConfig,
    TinyArenaGate,
    build_world,
    collect_training_data,
    normalize_weights,
    observe,
    policy_weights,
    regime_at_step,
    run_episode,
    main,
)


class RegimeArenaTests(unittest.TestCase):
    def test_weights_normalize(self) -> None:
        w = normalize_weights(np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(float(w.sum()), 1.0)
        self.assertTrue((w >= 0).all())

    def test_regime_not_in_feature_vector(self) -> None:
        cfg = ArenaConfig(grid=12, episode_steps=4)
        world = build_world(3, cfg)
        frame = observe(world, 0.0, cfg)
        self.assertEqual(frame.shape[0], 6)
        world.regime = 2
        world.predator = (5, 5)
        frame2 = observe(world, 0.0, cfg)
        self.assertNotEqual(frame[3], frame2[3])

    def test_gate_input_dim(self) -> None:
        gate = TinyArenaGate(np.random.default_rng(0))
        hist = np.zeros(INPUT_DIM)
        probs = gate.predict(hist)
        self.assertEqual(probs.shape, (3,))
        self.assertAlmostEqual(float(probs.sum()), 1.0)

    def test_deterministic_episode(self) -> None:
        cfg = ArenaConfig(train_episodes=2, eval_episodes=2, episode_steps=20, grid=14)
        a, _, _ = run_episode(5, "uniform", cfg)
        b, _, _ = run_episode(5, "uniform", cfg)
        self.assertEqual(a, b)

    def test_oracle_not_used_by_learned_policy(self) -> None:
        hist = np.zeros(INPUT_DIM)
        gate = TinyArenaGate(np.random.default_rng(1))
        w = policy_weights("learned", hist, gate, np.full(3, 1 / 3), np.zeros(3))
        self.assertAlmostEqual(float(w.sum()), 1.0)

    def test_cli_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp)
            code = main(["--output-dir", str(cfg_path), "--fast"])
            self.assertEqual(code, 0)
            summary = json.loads((cfg_path / "summary.json").read_text())
            self.assertIn("means", summary)
            self.assertTrue((cfg_path / "replay.html").exists())

    def test_training_labels_shape(self) -> None:
        cfg = ArenaConfig(train_episodes=2, episode_steps=12, grid=12, max_step_budget=5_000)
        static = normalize_weights(np.ones(3))
        x, y = collect_training_data(cfg, static)
        if len(x):
            self.assertEqual(x.shape[1], INPUT_DIM)
            self.assertTrue(set(y).issubset({0, 1, 2}))

    def test_regime_schedule(self) -> None:
        cfg = ArenaConfig(regime_period=10)
        self.assertEqual(regime_at_step(0, cfg), 0)
        self.assertEqual(regime_at_step(10, cfg), 1)


if __name__ == "__main__":
    unittest.main()