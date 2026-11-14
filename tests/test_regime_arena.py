import json
import tempfile
import unittest
from unittest.mock import patch
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
    expert_seek,
    fast_config,
    list_eval_world_ids,
    list_train_world_ids,
    main,
    step_world,
    summarize_fixture_advantages,
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
        world.walls.fill(0)
        world.walls[0, :] = world.walls[-1, :] = 1
        world.walls[:, 0] = world.walls[:, -1] = 1
        world.agent = (5, 5)
        world.pellets = {(6, 6)}
        world.predator = None
        world.walls[6, 5] = 1  # diagonal is open, axis projection is blocked
        self.assertEqual(expert_seek(world, cfg), (1, 1))
        self.assertEqual(blend_move(world, np.array([1., 0., 0.]), cfg), (0, 0))
        raw_reward = step_world(world.copy(), (1, 1), cfg, StepBudget(10))
        self.assertAlmostEqual(raw_reward, 0.99)
        cf = counterfactual_rewards(world, cfg, StepBudget(10))
        for i in range(3):
            trial = world.copy()
            one_hot = np.eye(3)[i]
            move = blend_move(trial, one_hot, cfg)
            reward = step_world(trial, move, cfg, StepBudget(10))
            self.assertAlmostEqual(cf[i], reward)
        self.assertAlmostEqual(cf[0], -0.01)

    def test_fixture_requires_each_expert_advantage(self) -> None:
        positive = {"open": [np.array([3., 0., 0.])],
                    "maze": [np.array([0., 3., 0.])],
                    "predator": [np.array([0., 0., 3.])]}
        result = summarize_fixture_advantages(positive)
        self.assertTrue(result["fixture_valid"])
        self.assertTrue(all(result["expert_has_advantage"].values()))
        # A winner in every regime is not sufficient if it is always the same expert.
        negative = {name: [np.array([3., 0., 0.])] for name in positive}
        result = summarize_fixture_advantages(negative)
        self.assertFalse(result["fixture_valid"])
        self.assertEqual(result["expert_has_advantage"], {"seek": True, "wall": False, "evade": False})
        positive["predator"] = []
        self.assertFalse(summarize_fixture_advantages(positive)["fixture_valid"])

    def test_non_finite_values_fail_closed(self) -> None:
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                normalize_weights(np.array([0., bad, 0.]))
        gate = TinyArenaGate(np.random.default_rng(0))
        gate.w1[0, 0] = np.nan
        with self.assertRaises(ValueError):
            gate.predict(np.zeros(INPUT_DIM))
        gate = TinyArenaGate(np.random.default_rng(0))
        with self.assertRaises(ValueError):
            gate.train_epoch(np.zeros((1, INPUT_DIM)), np.array([[np.nan, 0., 1.]]), .1)
        with patch("market_gate.regime_arena.np.outer", return_value=np.array([np.inf])):
            with self.assertRaisesRegex(ValueError, "gradient"):
                gate.train_epoch(np.zeros((1, INPUT_DIM)), np.array([[1., 0., 0.]]), .1)

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