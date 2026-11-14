"""Self-contained Regime Arena: tiny gate over three movement experts in a switching grid world."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

EXPERT_IDS = ("seek", "wall", "evade")
REGIME_NAMES = ("open", "maze", "predator")
HISTORY_LEN = 8
FEATURES_PER_FRAME = 6
INPUT_DIM = HISTORY_LEN * FEATURES_PER_FRAME
MAX_STEPS_BUDGET = 500_000
WALL_CLOCK_SECONDS = 90.0
SOFT_LABEL_TEMP = 0.08
ADVANTAGE_EPS = 0.002

PolicyName = Literal[
    "uniform",
    "expert_seek",
    "expert_wall",
    "expert_evade",
    "static",
    "adaptive",
    "learned",
    "oracle_regime_labeled",
]


def normalize_weights(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError("weights/logits must be three finite values")
    v = v - v.max()
    e = np.exp(np.clip(v, -40, 40))
    total = float(e.sum())
    if total <= 0 or not np.isfinite(total):
        raise ValueError("non-finite or zero weight normalization")
    return e / total


def uniform_weights() -> np.ndarray:
    return np.full(3, 1 / 3)


@dataclass
class ArenaConfig:
    grid: int = 24
    train_seeds: tuple[int, ...] = (1007, 1017, 1027)
    eval_seeds: tuple[int, ...] = (2001, 2099)
    train_episodes: int = 24
    eval_episodes: int = 16
    episode_steps: int = 111
    regime_period: int = 37
    gate_epochs: int = 12
    gate_lr: float = 0.08
    adaptive_eta: float = 0.35
    max_step_budget: int = MAX_STEPS_BUDGET


@dataclass
class StepRecord:
    t: int
    regime: int
    agent: tuple[int, int]
    weights: list[float]
    policy: str
    reward: float
    expert_cf: list[float]
    features: list[float]
    walls: list[list[int]]
    pellets: list[list[int]]
    predator: list[int] | None


class StepBudget:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.count = 0

    def add(self, n: int = 1) -> None:
        self.count += n
        if self.count > self.limit:
            raise RuntimeError(f"simulated transition budget exceeded ({self.count}>{self.limit})")


class WallClock:
    def __init__(self, seconds: float = WALL_CLOCK_SECONDS) -> None:
        self.deadline = time.perf_counter() + seconds

    def check(self) -> None:
        if time.perf_counter() > self.deadline:
            raise RuntimeError(f"wall clock exceeded ({WALL_CLOCK_SECONDS}s supervised cap)")


class TinyArenaGate:
    def __init__(self, rng: np.random.Generator) -> None:
        self.w1 = rng.normal(0, 0.15, (INPUT_DIM, 24))
        self.b1 = np.zeros(24)
        self.w2 = rng.normal(0, 0.15, (24, 12))
        self.b2 = np.zeros(12)
        self.w3 = rng.normal(0, 0.15, (12, 3))
        self.b3 = np.zeros(3)

    @property
    def parameter_count(self) -> int:
        return int(self.w1.size + self.b1.size + self.w2.size + self.b2.size + self.w3.size + self.b3.size)

    def forward(self, flat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = flat.reshape(-1).astype(np.float64)
        if x.size != INPUT_DIM:
            raise ValueError("gate expects flattened history")
        if not np.isfinite(x).all() or not all(np.isfinite(p).all() for p in (self.w1,self.b1,self.w2,self.b2,self.w3,self.b3)):
            raise ValueError("non-finite gate input or parameters")
        h1 = np.maximum(x @ self.w1 + self.b1, 0)
        h2 = np.maximum(h1 @ self.w2 + self.b2, 0)
        logits = h2 @ self.w3 + self.b3
        probs = normalize_weights(logits)
        return probs, h1, h2

    def predict(self, history: np.ndarray) -> np.ndarray:
        probs, _, _ = self.forward(history.reshape(-1))
        return probs

    def train_epoch(self, x: np.ndarray, y_soft: np.ndarray, lr: float) -> float:
        loss = 0.0
        for row, target in zip(x, y_soft, strict=True):
            target = np.asarray(target, dtype=np.float64)
            if not np.isfinite(target).all() or not np.isfinite(lr):
                raise ValueError("non-finite target or learning rate")
            target = target / max(float(target.sum()), 1e-12)
            probs, h1, h2 = self.forward(row)
            loss += -float(np.sum(target * np.log(np.maximum(probs, 1e-12))))
            grad = probs - target
            gw3 = np.outer(h2, grad)
            gb3 = grad
            gh2 = grad @ self.w3.T
            gh2 *= h2 > 0
            gw2 = np.outer(h1, gh2)
            gb2 = gh2
            gh1 = gh2 @ self.w2.T
            gh1 *= h1 > 0
            gw1 = np.outer(row.reshape(-1), gh1)
            gb1 = gh1
            if not all(np.isfinite(g).all() for g in (gw1, gb1, gw2, gb2, gw3, gb3)):
                raise ValueError("non-finite gate gradient")
            self.w3 -= lr * gw3
            self.b3 -= lr * gb3
            self.w2 -= lr * gw2
            self.b2 -= lr * gb2
            self.w1 -= lr * gw1
            self.b1 -= lr * gb1
            if not all(np.isfinite(p).all() for p in (self.w1,self.b1,self.w2,self.b2,self.w3,self.b3)):
                raise ValueError("non-finite gate parameters after update")
        return loss / max(len(x), 1)


@dataclass
class WorldState:
    walls: np.ndarray
    pellets: set[tuple[int, int]]
    agent: tuple[int, int]
    predator: tuple[int, int] | None
    regime: int
    step: int
    world_id: int
    base_walls: np.ndarray

    def copy(self) -> WorldState:
        return WorldState(
            self.walls.copy(),
            set(self.pellets),
            self.agent,
            self.predator,
            self.regime,
            self.step,
            self.world_id,
            self.base_walls.copy(),
        )


def world_id_train(seed: int, episode: int) -> int:
    return 1_000_000 + seed * 100 + episode


def world_id_eval(seed: int, episode: int) -> int:
    return 2_000_000 + seed * 100 + episode


def list_train_world_ids(cfg: ArenaConfig) -> list[int]:
    return [world_id_train(s, e) for s in cfg.train_seeds for e in range(cfg.train_episodes)]


def list_eval_world_ids(cfg: ArenaConfig) -> list[int]:
    return [world_id_eval(s, e) for s in cfg.eval_seeds for e in range(cfg.eval_episodes)]


def assert_disjoint_worlds(cfg: ArenaConfig) -> None:
    train = set(list_train_world_ids(cfg))
    eval_ = set(list_eval_world_ids(cfg))
    if train & eval_:
        raise ValueError(f"train/eval world id overlap: {sorted(train & eval_)[:5]}")


def _rng(world_id: int) -> np.random.Generator:
    return np.random.default_rng(world_id)


def build_world(world_id: int, cfg: ArenaConfig) -> WorldState:
    rng = _rng(world_id)
    g = cfg.grid
    walls = np.zeros((g, g), dtype=np.int8)
    walls[0, :] = walls[-1, :] = walls[:, 0] = walls[:, -1] = 1
    for _ in range(18 + (world_id % 7)):
        x, y = int(rng.integers(2, g - 2)), int(rng.integers(2, g - 2))
        walls[x, y] = 1
    pellets = set()
    for _ in range(10):
        x, y = int(rng.integers(1, g - 1)), int(rng.integers(1, g - 1))
        if walls[x, y] == 0:
            pellets.add((x, y))
    ax, ay = int(rng.integers(1, g - 1)), int(rng.integers(1, g - 1))
    while walls[ax, ay] or (ax, ay) in pellets:
        ax, ay = int(rng.integers(1, g - 1)), int(rng.integers(1, g - 1))
    px, py = int(rng.integers(1, g - 1)), int(rng.integers(1, g - 1))
    base = walls.copy()
    return WorldState(walls, pellets, (ax, ay), (px, py), 0, 0, world_id, base)


def regime_at_step(step: int, cfg: ArenaConfig) -> int:
    return (step // cfg.regime_period) % 3


def regime_step_counts(cfg: ArenaConfig) -> dict[str, int]:
    counts = {name: 0 for name in REGIME_NAMES}
    for t in range(cfg.episode_steps):
        counts[REGIME_NAMES[regime_at_step(t, cfg)]] += 1
    return counts


def assert_all_regimes(cfg: ArenaConfig) -> None:
    counts = regime_step_counts(cfg)
    missing = [k for k, v in counts.items() if v == 0]
    if missing:
        raise ValueError(f"episode does not visit regimes: missing {missing} in {counts}")


def apply_regime(world: WorldState, regime: int, cfg: ArenaConfig) -> None:
    world.regime = regime
    g = cfg.grid
    world.walls = world.base_walls.copy()
    if regime == 2:
        world.predator = world.predator or (g // 2, g // 2)
    else:
        world.predator = None
    if regime == 1:
        rng = _rng(world.world_id + regime * 997)
        for _ in range(14):
            x, y = int(rng.integers(2, g - 2)), int(rng.integers(2, g - 2))
            if (x, y) != world.agent:
                world.walls[x, y] = 1


def nearest_pellet(agent: tuple[int, int], pellets: set[tuple[int, int]]) -> tuple[int, int] | None:
    if not pellets:
        return None
    return min(pellets, key=lambda p: abs(p[0] - agent[0]) + abs(p[1] - agent[1]))


def min_wall_distance(walls: np.ndarray, agent: tuple[int, int]) -> float:
    x, y = agent
    dists = [0.0 if walls[x + dx, y + dy] else 1.0 for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    return float(min(dists))


def predator_distance(agent: tuple[int, int], predator: tuple[int, int] | None, grid: int) -> float:
    if predator is None:
        return 1.0
    manhattan = abs(agent[0] - predator[0]) + abs(agent[1] - predator[1])
    return float(min(1.0, manhattan / max(grid - 2, 1)))


def observe(world: WorldState, recent_rate: float, cfg: ArenaConfig) -> np.ndarray:
    ax, ay = world.agent
    target = nearest_pellet(world.agent, world.pellets)
    if target is None:
        pdx, pdy = 0.0, 0.0
    else:
        pdx = (target[0] - ax) / cfg.grid
        pdy = (target[1] - ay) / cfg.grid
    wall_neighbors = sum(
        1 for dx in (-1, 0, 1) for dy in (-1, 0, 1) if world.walls[ax + dx, ay + dy]
    )
    return np.array(
        [
            float(np.clip(pdx, -1, 1)),
            float(np.clip(pdy, -1, 1)),
            min_wall_distance(world.walls, world.agent),
            predator_distance(world.agent, world.predator, cfg.grid),
            float(np.clip(recent_rate, 0, 1)),
            wall_neighbors / 9.0,
        ],
        dtype=np.float64,
    )


def expert_seek(world: WorldState, cfg: ArenaConfig) -> tuple[int, int]:
    target = nearest_pellet(world.agent, world.pellets)
    if target is None:
        return 0, 0
    ax, ay = world.agent
    dx = int(math.copysign(1, target[0] - ax)) if target[0] != ax else 0
    dy = int(math.copysign(1, target[1] - ay)) if target[1] != ay else 0
    return _pick_move(world, dx, dy)


def expert_wall(world: WorldState, cfg: ArenaConfig) -> tuple[int, int]:
    ax, ay = world.agent
    for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        if world.walls[ax + dx, ay + dy]:
            side = (-dy, dx)
            if not world.walls[ax + side[0], ay + side[1]]:
                return side
    return expert_seek(world, cfg)


def expert_evade(world: WorldState, cfg: ArenaConfig) -> tuple[int, int]:
    if world.predator is None:
        return expert_seek(world, cfg)
    ax, ay = world.agent
    px, py = world.predator
    dx = int(math.copysign(1, ax - px)) if px != ax else 0
    dy = int(math.copysign(1, ay - py)) if py != ay else 0
    return _pick_move(world, dx, dy)


def _pick_move(world: WorldState, dx: int, dy: int) -> tuple[int, int]:
    ax, ay = world.agent
    if not world.walls[ax + dx, ay + dy]:
        return dx, dy
    if dx and not world.walls[ax + dx, ay]:
        return dx, 0
    if dy and not world.walls[ax, ay + dy]:
        return 0, dy
    return 0, 0


EXPERT_MOVES = (expert_seek, expert_wall, expert_evade)


def blend_move(world: WorldState, weights: np.ndarray, cfg: ArenaConfig) -> tuple[int, int]:
    moves = [fn(world, cfg) for fn in EXPERT_MOVES]
    vx = sum(weights[i] * moves[i][0] for i in range(3))
    vy = sum(weights[i] * moves[i][1] for i in range(3))
    if abs(vx) < 1e-9 and abs(vy) < 1e-9:
        return moves[int(np.argmax(weights))]
    dx = int(math.copysign(1, vx)) if abs(vx) >= abs(vy) and abs(vx) > 1e-9 else 0
    dy = int(math.copysign(1, vy)) if abs(vy) > abs(vx) and abs(vy) > 1e-9 else 0
    if dx == 0 and dy == 0:
        dx = int(math.copysign(1, vx)) if vx else 0
        dy = int(math.copysign(1, vy)) if vy else int(math.copysign(1, vx))
    return _pick_move(world, dx, dy)


def step_world(world: WorldState, move: tuple[int, int], cfg: ArenaConfig, budget: StepBudget) -> float:
    budget.add(1)
    ax, ay = world.agent
    nx, ny = ax + move[0], ay + move[1]
    reward = -0.01
    if not world.walls[nx, ny]:
        world.agent = (nx, ny)
    if world.agent in world.pellets:
        world.pellets.remove(world.agent)
        reward += 1.0
        px, py = int(_rng(world.world_id + world.step).integers(1, cfg.grid - 1)), int(
            _rng(world.world_id + world.step + 3).integers(1, cfg.grid - 1)
        )
        if not world.walls[px, py] and (px, py) != world.agent:
            world.pellets.add((px, py))
    if world.predator is not None:
        px, py = world.predator
        sx = int(math.copysign(1, world.agent[0] - px)) if world.agent[0] != px else 0
        sy = int(math.copysign(1, world.agent[1] - py)) if world.agent[1] != py else 0
        if not world.walls[px + sx, py + sy]:
            world.predator = (px + sx, py + sy)
        if world.predator == world.agent:
            reward -= 0.5
            world.agent = (cfg.grid // 2, cfg.grid // 2)
    world.step += 1
    return reward


def counterfactual_rewards(
    world: WorldState, cfg: ArenaConfig, budget: StepBudget
) -> np.ndarray:
    """Each expert evaluated through the same one-hot blend_move path as policies."""
    rewards = []
    for i in range(3):
        trial = world.copy()
        one_hot = np.zeros(3)
        one_hot[i] = 1.0
        move = blend_move(trial, one_hot, cfg)
        rewards.append(step_world(trial, move, cfg, budget))
    return np.array(rewards, dtype=np.float64)


def soft_label_from_cf(cf: np.ndarray) -> np.ndarray:
    return normalize_weights(cf / SOFT_LABEL_TEMP)


def policy_weights(
    name: PolicyName,
    history: np.ndarray,
    gate: TinyArenaGate | None,
    static: np.ndarray,
    adaptive_log: np.ndarray,
    regime: int | None = None,
) -> np.ndarray:
    if name == "uniform":
        return uniform_weights()
    if name == "expert_seek":
        return np.array([1.0, 0.0, 0.0])
    if name == "expert_wall":
        return np.array([0.0, 1.0, 0.0])
    if name == "expert_evade":
        return np.array([0.0, 0.0, 1.0])
    if name == "static":
        return static.copy()
    if name == "adaptive":
        return normalize_weights(adaptive_log)
    if name == "learned":
        if gate is None:
            raise ValueError("learned policy requires gate")
        return gate.predict(history)
    if name == "oracle_regime_labeled":
        if regime is None:
            raise ValueError("oracle needs regime")
        picks = (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
        return picks[regime]
    raise ValueError(f"unknown policy {name}")


def run_episode(
    world_id: int,
    policy: PolicyName,
    cfg: ArenaConfig,
    budget: StepBudget,
    clock: WallClock,
    *,
    gate: TinyArenaGate | None = None,
    static: np.ndarray | None = None,
    record: bool = False,
) -> tuple[float, list[StepRecord], int]:
    world = build_world(world_id, cfg)
    static = static if static is not None else uniform_weights()
    adaptive_log = np.zeros(3)
    history = np.zeros((HISTORY_LEN, FEATURES_PER_FRAME))
    recent: list[float] = []
    total_reward = 0.0
    records: list[StepRecord] = []
    decision_steps = 0
    prev_regime = -1

    for t in range(cfg.episode_steps):
        clock.check()
        if budget.count >= budget.limit:
            break
        regime = regime_at_step(t, cfg)
        if regime != prev_regime:
            apply_regime(world, regime, cfg)
            prev_regime = regime
        recent_rate = float(np.mean(recent[-10:])) if recent else 0.0
        frame = observe(world, recent_rate, cfg)
        history = np.roll(history, -1, axis=0)
        history[-1] = frame
        flat = history.reshape(-1)
        cf = counterfactual_rewards(world, cfg, budget)
        weights = policy_weights(
            policy,
            flat,
            gate,
            static,
            adaptive_log,
            regime if policy == "oracle_regime_labeled" else None,
        )
        move = blend_move(world, weights, cfg)
        reward = step_world(world, move, cfg, budget)
        total_reward += reward
        recent.append(1.0 if reward > 0.5 else 0.0)
        if policy == "adaptive":
            adaptive_log = adaptive_log + cfg.adaptive_eta * cf
        decision_steps += 1
        if record:
            records.append(
                StepRecord(
                    t=t,
                    regime=regime,
                    agent=world.agent,
                    weights=weights.tolist(),
                    policy=policy,
                    reward=reward,
                    expert_cf=cf.tolist(),
                    features=frame.tolist(),
                    walls=world.walls.astype(int).tolist(),
                    pellets=[list(p) for p in sorted(world.pellets)],
                    predator=list(world.predator) if world.predator else None,
                )
            )
    return total_reward / max(decision_steps, 1), records, decision_steps


def rollout_train_episode(
    world_id: int,
    cfg: ArenaConfig,
    budget: StepBudget,
    clock: WallClock,
    static: np.ndarray,
    *,
    collect_labels: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Roll out training world IDs; static fitting and collection may take different paths."""
    world = build_world(world_id, cfg)
    history = np.zeros((HISTORY_LEN, FEATURES_PER_FRAME))
    recent: list[float] = []
    prev_regime = -1
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    expert_returns = np.zeros(3)

    for t in range(cfg.episode_steps):
        clock.check()
        if budget.count >= budget.limit:
            break
        regime = regime_at_step(t, cfg)
        if regime != prev_regime:
            apply_regime(world, regime, cfg)
            prev_regime = regime
        recent_rate = float(np.mean(recent[-10:])) if recent else 0.0
        frame = observe(world, recent_rate, cfg)
        history = np.roll(history, -1, axis=0)
        history[-1] = frame
        cf = counterfactual_rewards(world, cfg, budget)
        expert_returns += cf
        if collect_labels:
            xs.append(history.reshape(-1).copy())
            ys.append(soft_label_from_cf(cf))
        weights = uniform_weights() * 0.5 + static * 0.5
        move = blend_move(world, weights, cfg)
        reward = step_world(world, move, cfg, budget)
        recent.append(1.0 if reward > 0.5 else 0.0)
    return xs, ys, expert_returns


def fit_static_blend(cfg: ArenaConfig, budget: StepBudget, clock: WallClock) -> np.ndarray:
    totals = np.zeros(3)
    for wid in list_train_world_ids(cfg):
        _, _, rets = rollout_train_episode(wid, cfg, budget, clock, uniform_weights(), collect_labels=False)
        totals += rets
    return normalize_weights(totals)


def collect_training_data(
    cfg: ArenaConfig, static: np.ndarray, budget: StepBudget, clock: WallClock
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for wid in list_train_world_ids(cfg):
        clock.check()
        if budget.count >= budget.limit // 2:
            break
        part_x, part_y, _ = rollout_train_episode(wid, cfg, budget, clock, static)
        xs.extend(part_x)
        ys.extend(part_y)
    return np.asarray(xs), np.asarray(ys)


def training_target_diagnostics(y_soft: np.ndarray) -> dict[str, object]:
    hard = np.argmax(y_soft, axis=1)
    counts = {EXPERT_IDS[i]: int(np.sum(hard == i)) for i in range(3)}
    entropy = float(-np.mean(np.sum(y_soft * np.log(np.maximum(y_soft, 1e-12)), axis=1)))
    return {
        "hard_label_counts": counts,
        "mean_label_entropy": entropy,
        "n_samples": int(len(y_soft)),
    }


def expert_regime_fixture_diagnostic(
    cfg: ArenaConfig, budget: StepBudget, clock: WallClock, *, sample_worlds: int = 6
) -> dict[str, object]:
    worlds = list_train_world_ids(cfg)[:sample_worlds]
    by_regime: dict[str, list[np.ndarray]] = {name: [] for name in REGIME_NAMES}
    for wid in worlds:
        world = build_world(wid, cfg)
        history = np.zeros((HISTORY_LEN, FEATURES_PER_FRAME))
        prev = -1
        for t in range(min(cfg.episode_steps, cfg.regime_period * 3)):
            clock.check()
            regime = regime_at_step(t, cfg)
            if regime != prev:
                apply_regime(world, regime, cfg)
                prev = regime
            cf = counterfactual_rewards(world, cfg, budget)
            by_regime[REGIME_NAMES[regime]].append(cf)
            move = blend_move(world, uniform_weights(), cfg)
            step_world(world, move, cfg, budget)
            _ = history  # history warm-up not required for CF diagnostic
    return summarize_fixture_advantages(by_regime)


def summarize_fixture_advantages(by_regime: dict[str, list[np.ndarray]]) -> dict[str, object]:
    """Require every expert to have positive mean advantage in at least one regime."""
    per_regime = {}
    per_expert = {name: False for name in EXPERT_IDS}
    all_present = True
    for name in REGIME_NAMES:
        if not by_regime[name]:
            all_present = False
            per_regime[name] = {"mean_cf": None, "best_expert_advantage": None}
            continue
        mean_cf = np.stack(by_regime[name]).mean(axis=0)
        if mean_cf.shape != (3,) or not np.isfinite(mean_cf).all():
            raise ValueError("non-finite or malformed fixture diagnostic")
        advantage = mean_cf - mean_cf.mean()
        for i, expert in enumerate(EXPERT_IDS):
            per_expert[expert] |= bool(advantage[i] >= ADVANTAGE_EPS)
        per_regime[name] = {"mean_cf": mean_cf.tolist(), "expert_advantage": advantage.tolist(),
                            "best_expert_advantage": float(advantage.max())}
    return {"per_regime": per_regime, "expert_has_advantage": per_expert,
            "fixture_valid": bool(all_present and all(per_expert.values()))}


def paired_bootstrap_diff(
    learned: dict[int, list[float]],
    baseline: dict[int, list[float]],
    n: int = 400,
) -> dict[str, float]:
    rng = np.random.default_rng(0)
    worlds = sorted(set(learned) & set(baseline))
    if not worlds:
        return {"mean_diff": 0.0, "low": 0.0, "high": 0.0}
    diffs = []
    for _ in range(n):
        pick = rng.choice(worlds, size=len(worlds), replace=True)
        sample = [float(np.mean(learned[w]) - np.mean(baseline[w])) for w in pick]
        diffs.append(float(np.mean(sample)))
    return {
        "mean_diff": float(np.mean([np.mean(learned[w]) - np.mean(baseline[w]) for w in worlds])),
        "low": float(np.percentile(diffs, 2.5)),
        "high": float(np.percentile(diffs, 97.5)),
    }


def post_switch_adaptation(records: list[StepRecord], window: int = 8) -> float | None:
    if len(records) < window + 2:
        return None
    switches = [i for i in range(1, len(records)) if records[i].regime != records[i - 1].regime]
    if not switches:
        return None
    deltas = []
    for idx in switches:
        before = np.array(records[max(0, idx - 1)].weights)
        after = np.array(records[min(len(records) - 1, idx + window)].weights)
        deltas.append(float(np.linalg.norm(after - before)))
    return float(np.mean(deltas)) if deltas else None


def run_benchmark(
    cfg: ArenaConfig,
    gate: TinyArenaGate,
    static: np.ndarray,
    budget: StepBudget,
    clock: WallClock,
) -> dict[str, object]:
    policies: list[PolicyName] = [
        "uniform",
        "expert_seek",
        "expert_wall",
        "expert_evade",
        "static",
        "adaptive",
        "learned",
        "oracle_regime_labeled",
    ]
    by_policy_world: dict[str, dict[int, list[float]]] = {p: {} for p in policies}
    sample_counts: dict[str, int] = {p: 0 for p in policies}
    adaptation: dict[str, float | None] = {}
    started = time.perf_counter()
    eval_ids = list_eval_world_ids(cfg)

    for policy in policies:
        for wid in eval_ids:
            clock.check()
            if budget.count >= budget.limit:
                break
            scores = []
            rec: list[StepRecord] = []
            score, records, _ = run_episode(
                wid,
                policy,
                cfg,
                budget,
                clock,
                gate=gate,
                static=static,
                record=(policy == "learned" and wid == eval_ids[0]),
            )
            scores.append(score)
            sample_counts[policy] += 1
            by_policy_world[policy].setdefault(wid, []).extend(scores)
            if records:
                rec = records
            if policy == "learned" and rec:
                adaptation[policy] = post_switch_adaptation(rec)

    means = {
        policy: float(np.mean([v for vals in by_policy_world[policy].values() for v in vals]))
        for policy in policies
    }
    world_means = {
        policy: {str(wid): float(np.mean(vals)) for wid, vals in by_policy_world[policy].items()}
        for policy in policies
    }
    learned = means["learned"]
    uni = means["uniform"]
    stat = means["static"]
    literal_threshold_met = learned > uni and learned > stat
    paired_uniform = paired_bootstrap_diff(by_policy_world["learned"], by_policy_world["uniform"])
    paired_static = paired_bootstrap_diff(by_policy_world["learned"], by_policy_world["static"])
    return {
        "means": means,
        "world_means": world_means,
        "eval_world_ids": eval_ids,
        "independent_eval_samples": sample_counts,
        "paired_vs_uniform": paired_uniform,
        "paired_vs_static": paired_static,
        "literal_threshold_met": literal_threshold_met,
        "post_switch_weight_delta_mean": adaptation,
        "simulated_transitions_total": budget.count,
        "runtime_seconds": time.perf_counter() - started,
        "gate_parameters": gate.parameter_count,
        "static_weights": static.tolist(),
    }


def write_replay_html(path: Path, records: list[StepRecord], cfg: ArenaConfig, summary: dict) -> None:
    payload = {
        "grid": cfg.grid,
        "records": [
            {
                "t": r.t,
                "regime": REGIME_NAMES[r.regime],
                "agent": list(r.agent),
                "weights": r.weights,
                "reward": r.reward,
                "expert_cf": r.expert_cf,
                "walls": r.walls,
                "pellets": r.pellets,
                "predator": r.predator,
            }
            for r in records
        ],
        "summary_snippet": {
            "means": summary.get("means"),
            "literal_threshold_met": summary.get("literal_threshold_met"),
            "fixture_valid": summary.get("fixture_valid"),
        },
        "assumptions": [
            "v2 exploratory; v1 holdout was invalid.",
            "Regime id hidden from gate; shown in replay for debug.",
            "Counterfactuals use one-hot blend_move execution.",
        ],
    }
    data = json.dumps(payload)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Regime Arena Replay v2</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 1rem; background: #111; color: #eee; }}
#arena {{ display: grid; gap: 2px; width: 360px; height: 360px; background: #222; }}
.cell {{ width: 100%; aspect-ratio: 1; background: #1a2a1a; border-radius: 2px; }}
.cell.wall {{ background: #444; }}
.cell.agent {{ background: #4af; }}
.cell.pred {{ background: #f44; }}
.cell.pellet {{ background: #fb4; }}
.panel {{ max-width: 720px; }}
.bar {{ height: 12px; background: #333; margin: 4px 0; }}
.fill {{ height: 100%; background: #6c6; }}
</style>
</head>
<body>
<h1>Regime Arena Replay v2</h1>
<p id="fixture-status"></p>
<p id="regime"></p>
<label for="slider">Replay step</label>
<input type="range" id="slider" min="0" max="{max(0, len(records) - 1)}" value="0" style="width: 100%"/>
<div id="arena"></div>
<div class="panel" id="weights"></div>
<div class="panel" id="cf"></div>
<div class="panel" id="assumptions"></div>
<script>
const data = {data};
document.getElementById('fixture-status').textContent = data.summary_snippet.fixture_valid === true ? 'Fixture diagnostic: valid for the stated toy gate.' : data.summary_snippet.fixture_valid === false ? 'FIXTURE INVALID: do not interpret gate performance as adaptation.' : 'Fixture status unavailable: do not interpret gate performance.';
const grid = data.grid;
const arena = document.getElementById('arena');
arena.style.gridTemplateColumns = 'repeat(' + grid + ', 1fr)';
function render(step) {{
  const rec = data.records[step];
  document.getElementById('regime').textContent =
    'Step ' + rec.t + ' · regime ' + rec.regime + ' (debug) · reward ' + rec.reward.toFixed(3);
  const pellets = new Set(rec.pellets.map(p => p[0] + ',' + p[1]));
  const pred = rec.predator ? (rec.predator[0] + ',' + rec.predator[1]) : null;
  arena.innerHTML = '';
  for (let y = 0; y < grid; y++) {{
    for (let x = 0; x < grid; x++) {{
      const d = document.createElement('div');
      d.className = 'cell';
      if (rec.walls[x][y]) d.classList.add('wall');
      const key = x + ',' + y;
      if (pellets.has(key)) d.classList.add('pellet');
      if (pred === key) d.classList.add('pred');
      if (rec.agent[0] === x && rec.agent[1] === y) d.classList.add('agent');
      arena.appendChild(d);
    }}
  }}
  const names = ['seek','wall','evade'];
  let whtml = '<h3>Gate weights</h3>';
  rec.weights.forEach((w, i) => {{
    whtml += names[i] + ' <div class="bar"><div class="fill" style="width:' + (w*100).toFixed(1) + '%"></div></div>';
  }});
  document.getElementById('weights').innerHTML = whtml;
  document.getElementById('cf').innerHTML =
    '<h3>Counterfactual expert rewards</h3>' + rec.expert_cf.map((v,i)=>names[i]+': '+v.toFixed(3)).join(' · ');
}}
document.getElementById('slider').oninput = (e) => render(+e.target.value);
document.getElementById('assumptions').innerHTML =
  '<h3>Assumptions</h3><ul>' + data.assumptions.map(a=>'<li>'+a+'</li>').join('') + '</ul>';
render(0);
</script>
</body>
</html>"""
    path.write_text(html, encoding="utf-8")


def fast_config() -> ArenaConfig:
    return ArenaConfig(
        train_episodes=4,
        eval_episodes=3,
        episode_steps=39,
        regime_period=13,
        gate_epochs=3,
        max_step_budget=80_000,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regime Arena train/eval and artifact export")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fast", action="store_true", help="Reduced episodes for CI/smoke runs")
    args = parser.parse_args(argv)
    cfg = fast_config() if args.fast else ArenaConfig()
    assert_disjoint_worlds(cfg)
    assert_all_regimes(cfg)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    budget = StepBudget(cfg.max_step_budget)
    clock = WallClock()

    fixture = expert_regime_fixture_diagnostic(cfg, budget, clock)
    static = fit_static_blend(cfg, budget, clock)
    x, y = collect_training_data(cfg, static, budget, clock)
    target_diag = training_target_diagnostics(y)
    gate = TinyArenaGate(_rng(1717))
    losses = []
    for _ in range(cfg.gate_epochs):
        clock.check()
        losses.append(gate.train_epoch(x, y, cfg.gate_lr))

    replay_wid = list_eval_world_ids(cfg)[0]
    _, replay_records, _ = run_episode(
        replay_wid,
        "learned",
        cfg,
        budget,
        clock,
        gate=gate,
        static=static,
        record=True,
    )

    summary = run_benchmark(cfg, gate, static, budget, clock)
    summary["schema_version"] = 2
    summary["experiment"] = "regime-arena-v2-exploratory"
    summary["train_world_ids"] = list_train_world_ids(cfg)
    summary["eval_world_ids_frozen"] = list_eval_world_ids(cfg)
    summary["regime_step_counts"] = regime_step_counts(cfg)
    summary["fixture_diagnostic"] = fixture
    summary["fixture_valid"] = fixture["fixture_valid"]
    summary["training_target_diagnostics"] = target_diag
    summary["training_loss_last"] = float(losses[-1]) if losses else None
    summary["interpret_gate_performance"] = bool(fixture["fixture_valid"])
    summary["limitations"] = [
        "Exploratory v2 after review; v1 primary_success is void.",
        "literal_threshold_met is descriptive only.",
        "oracle_regime_labeled is not an upper bound.",
    ]
    if not fixture["fixture_valid"]:
        summary["limitations"].append(
            "fixture_invalid: experts lack per-regime advantage; do not interpret gate metrics."
        )

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    replay = {
        "world_id": replay_wid,
        "steps": [
            {
                "t": r.t,
                "regime": r.regime,
                "regime_name": REGIME_NAMES[r.regime],
                "agent": r.agent,
                "weights": r.weights,
                "reward": r.reward,
                "expert_cf": r.expert_cf,
                "walls": r.walls,
                "pellets": r.pellets,
                "predator": r.predator,
            }
            for r in replay_records
        ],
    }
    (out / "replay.json").write_text(json.dumps(replay, indent=2), encoding="utf-8")
    write_replay_html(out / "replay.html", replay_records, cfg, summary)
    print(
        json.dumps(
            {
                "fixture_valid": summary["fixture_valid"],
                "literal_threshold_met": summary["literal_threshold_met"],
                "interpret_gate_performance": summary["interpret_gate_performance"],
                "means": summary["means"],
                "simulated_transitions_total": summary["simulated_transitions_total"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())