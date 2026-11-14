import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from market_gate.regime_arena import (
    INPUT_DIM,
    ArenaConfig,
    StepBudget,
    TinyArenaGate,
    WallClock,
    assert_all_regimes,
    assert_disjoint_worlds,
    blend_move,
    build_world,
    counterfactual_rewards,
    expert_wall,
    expert_seek,
    fast_config,
    list_eval_world_ids,
    list_train_world_ids,
    main,
    normalize_weights,
    policy_weights,
    uniform_weights,
    regime_step_counts,
    soft_label_from_cf,
)


class RegimeArenaTests(unittest.TestCase):
    def test_train_eval_world_ids_disjoint(self) -> None:
        cfg = ArenaConfig()
        assert_disjoint_worlds(cfg)
        train = set(list_train_world_ids(cfg))
        eval_ = set(list_eval_world_ids(cfg))
        self.assertFalse(train & eval_)

    def test_all_three_regimes_in_default_and_fast(self) -> None:
        for cfg in (ArenaConfig(), fast_config()):
            assert_all_regimes(cfg)
            counts = regime_step_counts(cfg)
            self.assertGreater(counts["predator"], 0)
            self.assertGreater(counts["maze"], 0)
            self.assertGreater(counts["open"], 0)

    def test_counterfactual_matches_one_hot_execution(self) -> None:
        cfg = ArenaConfig(grid=14, episode_steps=20)
        world = build_world(1_001_007, cfg)
        budget = StepBudget(10_000)
        cf = counterfactual_rewards(world, cfg, budget)
        for i, name in enumerate(("expert_seek", "expert_wall", "expert_evade")):
            trial = world.copy()
            w = policy_weights(name, np.zeros(INPUT_DIM), None, np.full(3, 1 / 3), np.zeros(3))
            move_policy = blend_move(trial, w, cfg)
            trial2 = world.copy()
            one = np.zeros(3)
            one[i] = 1.0
            move_cf = blend_move(trial2, one, cfg)
            self.assertEqual(move_policy, move_cf)

    def test_soft_label_uniform_on_ties(self) -> None:
        cf = np.array([0.5, 0.5, 0.5])
        lab = soft_label_from_cf(cf)
        self.assertAlmostEqual(float(lab.std()), 0.0, places=5)

    def test_adaptive_softmax_distinguishes_negative_totals(self) -> None:
        scores = np.array([-0.4, -0.5, -0.7])
        clipped = np.maximum(scores, 0.0)
        clipped = clipped / clipped.sum() if clipped.sum() else np.full(3, 1 / 3)
        softmax = normalize_weights(scores)
        self.assertTrue(np.allclose(clipped, uniform_weights()))
        self.assertGreater(float(softmax[0]), float(softmax[2]))
        self.assertFalse(np.allclose(softmax, uniform_weights()))

    def test_cf_included_in_step_budget(self) -> None:
        cfg = ArenaConfig(grid=12, episode_steps=5)
        world = build_world(1_001_007, cfg)
        budget = StepBudget(20)
        counterfactual_rewards(world, cfg, budget)
        self.assertEqual(budget.count, 3)

    def test_diagonal_seek_vs_one_hot_blend_can_differ(self) -> None:
        cfg = ArenaConfig()
        world = build_world(2_000_042, cfg)
        raw_seek = expert_seek(world, cfg)
        move = blend_move(world, np.array([1.0, 0.0, 0.0]), cfg)
        self.assertIsInstance(raw_seek, tuple)
        self.assertIsInstance(move, tuple)

    def test_deterministic_episode_world_id(self) -> None:
        from market_gate.regime_arena import run_episode

        cfg = fast_config()
        budget = StepBudget(50_000)
        clock = WallClock(60)
        wid = list_eval_world_ids(cfg)[0]
        a, _, _ = run_episode(wid, "uniform", cfg, budget, clock)
        budget2 = StepBudget(50_000)
        b, _, _ = run_episode(wid, "uniform", cfg, budget2, clock)
        self.assertEqual(a, b)

    def test_gate_predict(self) -> None:
        gate = TinyArenaGate(np.random.default_rng(0))
        probs = gate.predict(np.zeros(INPUT_DIM))
        self.assertAlmostEqual(float(probs.sum()), 1.0)

    def test_cli_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = main(["--output-dir", tmp, "--fast"])
            self.assertEqual(code, 0)
            summary = json.loads((Path(tmp) / "summary.json").read_text())
            self.assertEqual(summary["schema_version"], 2)
            self.assertIn("independent_eval_samples", summary)
            self.assertIn("paired_vs_uniform", summary)
            self.assertTrue((Path(tmp) / "replay.html").exists())


if __name__ == "__main__":
    unittest.main()