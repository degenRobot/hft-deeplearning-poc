"""Self-contained Regime Arena: tiny gate over three movement experts in a switching grid world."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

EXPERT_IDS = ("seek", "wall", "evade")
REGIME_NAMES = ("open", "maze", "predator")
HISTORY_LEN = 8
FEATURES_PER_FRAME = 6
INPUT_DIM = HISTORY_LEN * FEATURES_PER_FRAME
MAX_STEPS_BUDGET = 500_000

PolicyName = Literal[
    "uniform",
    "expert_seek",
    "expert_wall",
    "expert_evade",
    "static",
    "adaptive",
    "learned",
    "oracle",
]


def normalize_weights(values: np.ndarray) -> np.ndarray:
    v = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = float(v.sum())
    if total <= 0:
        return np.full(3, 1 / 3)
    return v / total


def uniform_weights() -> np.ndarray:
    return np.full(3, 1 / 3)


@dataclass
class ArenaConfig:
    grid: int = 24
    train_seeds: tuple[int, ...] = (7, 17, 27)
    eval_seeds: tuple[int, ...] = (42, 99)
    train_episodes: int = 24
    eval_episodes: int = 16
    episode_steps: int = 72
    regime_period: int = 36
    gate_epochs: int = 12
    gate_lr: float = 0.08
    adaptive_eta: float = 0.35
    max_step_budget: int = MAX_STEPS_BUDGET


@dataclass
class StepRecord:
    t: int
    regime: int  # debug / oracle only — not a gate feature
    agent: tuple[int, int]
    weights: list[float]
    policy: str
    reward: float
    expert_cf: list[float]
    features: list[float]


class TinyArenaGate:
    """48 -> 24 -> 12 -> 3 softmax gate (NumPy only)."""

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

    def forward(self, flat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = flat.reshape(-1).astype(np.float64)
        if x.size != INPUT_DIM:
            raise ValueError("gate expects flattened history")
        h1 = np.maximum(x @ self.w1 + self.b1, 0)
        h2 = np.maximum(h1 @ self.w2 + self.b2, 0)
        logits = h2 @ self.w3 + self.b3
        logits -= logits.max()
        probs = normalize_weights(np.exp(np.clip(logits, -40, 40)))
        return probs, h1, h2, logits

    def predict(self, history: np.ndarray) -> np.ndarray:
        probs, _, _, _ = self.forward(history.reshape(-1))
        return probs

    def train_epoch(self, x: np.ndarray, y: np.ndarray, lr: float) -> float:
        loss = 0.0
        for row, target in zip(x, y, strict=True):
            probs, h1, h2, _ = self.forward(row)
            loss += -float(np.log(max(probs[target], 1e-12)))
            grad = probs.copy()
            grad[target] -= 1.0
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
            self.w3 -= lr * gw3
            self.b3 -= lr * gb3
            self.w2 -= lr * gw2
            self.b2 -= lr * gb2
            self.w1 -= lr * gw1
            self.b1 -= lr * gb1
        return loss / max(len(x), 1)


@dataclass
class WorldState:
    walls: np.ndarray
    pellets: set[tuple[int, int]]
    agent: tuple[int, int]
    predator: tuple[int, int] | None
    regime: int
    step: int
    seed: int

    def copy(self) -> WorldState:
        return WorldState(
            self.walls.copy(),
            set(self.pellets),
            self.agent,
            self.predator,
            self.regime,
            self.step,
            self.seed,
        )


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def build_world(seed: int, cfg: ArenaConfig) -> WorldState:
    rng = _rng(seed)
    g = cfg.grid
    walls = np.zeros((g, g), dtype=np.int8)
    # border walls
    walls[0, :] = walls[-1, :] = walls[:, 0] = walls[:, -1] = 1
    for _ in range(18 + (seed % 7)):
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
    return WorldState(walls, pellets, (ax, ay), (px, py), 0, 0, seed)


def regime_at_step(step: int, cfg: ArenaConfig) -> int:
    return (step // cfg.regime_period) % 3


def apply_regime(world: WorldState, regime: int, cfg: ArenaConfig) -> None:
    world.regime = regime
    g = cfg.grid
    if regime == 2:
        world.predator = world.predator or (g // 2, g // 2)
    else:
        world.predator = None
    if regime == 1:
        rng = _rng(world.seed + regime * 997)
        for _ in range(12):
            x, y = int(rng.integers(2, g - 2)), int(rng.integers(2, g - 2))
            if (x, y) != world.agent:
                world.walls[x, y] = 1


def nearest_pellet(agent: tuple[int, int], pellets: set[tuple[int, int]]) -> tuple[int, int] | None:
    if not pellets:
        return None
    return min(pellets, key=lambda p: abs(p[0] - agent[0]) + abs(p[1] - agent[1]))


def min_wall_distance(walls: np.ndarray, agent: tuple[int, int]) -> float:
    x, y = agent
    dists = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if walls[nx, ny]:
            dists.append(0.0)
        else:
            dists.append(1.0)
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
    ax, ay = world.agent
    wall_neighbors = 0
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if world.walls[ax + dx, ay + dy]:
                wall_neighbors += 1
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
            # slide along wall clockwise
            side = (-dy, dx)
            if not world.walls[ax + side[0], ay + side[1]]:
                return side
    target = nearest_pellet(world.agent, world.pellets)
    if target is None:
        return 0, 0
    dx = int(math.copysign(1, target[0] - ax)) if target[0] != ax else 0
    dy = int(math.copysign(1, target[1] - ay)) if target[1] != ay else 0
    return _pick_move(world, dx, dy)


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


def step_world(
    world: WorldState,
    move: tuple[int, int],
    cfg: ArenaConfig,
) -> float:
    ax, ay = world.agent
    nx, ny = ax + move[0], ay + move[1]
    reward = -0.01
    if not world.walls[nx, ny]:
        world.agent = (nx, ny)
    if world.agent in world.pellets:
        world.pellets.remove(world.agent)
        reward += 1.0
        px, py = int(_rng(world.seed + world.step).integers(1, cfg.grid - 1)), int(
            _rng(world.seed + world.step + 3).integers(1, cfg.grid - 1)
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
            g = cfg.grid
            world.agent = (g // 2, g // 2)
    world.step += 1
    return reward


def counterfactual_rewards(world: WorldState, cfg: ArenaConfig) -> np.ndarray:
    base = world.copy()
    rewards = []
    for fn in EXPERT_MOVES:
        trial = base.copy()
        move = fn(trial, cfg)
        r = step_world(trial, move, cfg)
        rewards.append(r)
    return np.array(rewards, dtype=np.float64)


def policy_weights(
    name: PolicyName,
    history: np.ndarray,
    gate: TinyArenaGate | None,
    static: np.ndarray,
    adaptive: np.ndarray,
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
        return normalize_weights(adaptive)
    if name == "learned":
        if gate is None:
            raise ValueError("learned policy requires gate")
        return gate.predict(history)
    if name == "oracle":
        if regime is None:
            raise ValueError("oracle needs regime")
        # open->seek, maze->wall, predator->evade
        picks = (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
        return picks[regime]
    raise ValueError(f"unknown policy {name}")


def run_episode(
    seed: int,
    policy: PolicyName,
    cfg: ArenaConfig,
    *,
    gate: TinyArenaGate | None = None,
    static: np.ndarray | None = None,
    record: bool = False,
) -> tuple[float, list[StepRecord], int]:
    world = build_world(seed, cfg)
    static = static if static is not None else uniform_weights()
    adaptive_scores = np.zeros(3)
    history = np.zeros((HISTORY_LEN, FEATURES_PER_FRAME))
    recent: list[float] = []
    total_reward = 0.0
    records: list[StepRecord] = []
    steps = 0
    prev_regime = -1

    for t in range(cfg.episode_steps):
        if steps >= cfg.max_step_budget:
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
        cf = counterfactual_rewards(world, cfg)
        weights = policy_weights(
            policy,
            flat,
            gate,
            static,
            adaptive_scores,
            regime if policy == "oracle" else None,
        )
        move = blend_move(world, weights, cfg)
        reward = step_world(world, move, cfg)
        total_reward += reward
        recent.append(1.0 if reward > 0.5 else 0.0)
        if policy == "adaptive":
            adaptive_scores += cfg.adaptive_eta * cf
        steps += 1
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
                )
            )
    return total_reward / cfg.episode_steps, records, steps


def fit_static_blend(cfg: ArenaConfig) -> np.ndarray:
    totals = np.zeros(3)
    for seed in cfg.train_seeds:
        for ep in range(min(8, cfg.train_episodes)):
            world = build_world(seed + ep * 31, cfg)
            for t in range(cfg.episode_steps):
                regime = regime_at_step(t, cfg)
                apply_regime(world, regime, cfg)
                cf = counterfactual_rewards(world, cfg)
                totals += np.maximum(cf, 0)
                move = blend_move(world, uniform_weights(), cfg)
                step_world(world, move, cfg)
    return normalize_weights(totals)


def collect_training_data(cfg: ArenaConfig, static: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    steps = 0
    for seed in cfg.train_seeds:
        for ep in range(cfg.train_episodes):
            world = build_world(seed + ep * 31, cfg)
            history = np.zeros((HISTORY_LEN, FEATURES_PER_FRAME))
            recent: list[float] = []
            prev = -1
            for t in range(cfg.episode_steps):
                if steps >= cfg.max_step_budget // 2:
                    return np.asarray(xs), np.asarray(ys, dtype=np.int64)
                regime = regime_at_step(t, cfg)
                if regime != prev:
                    apply_regime(world, regime, cfg)
                    prev = regime
                recent_rate = float(np.mean(recent[-10:])) if recent else 0.0
                frame = observe(world, recent_rate, cfg)
                history = np.roll(history, -1, axis=0)
                history[-1] = frame
                cf = counterfactual_rewards(world, cfg)
                label = int(np.argmax(cf))
                xs.append(history.reshape(-1).copy())
                ys.append(label)
                weights = uniform_weights() * 0.5 + static * 0.5
                move = blend_move(world, weights, cfg)
                reward = step_world(world, move, cfg)
                recent.append(1.0 if reward > 0.5 else 0.0)
                steps += 1
    return np.asarray(xs), np.asarray(ys, dtype=np.int64)


def bootstrap_ci(values_by_seed: dict[int, list[float]], n: int = 400) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    seeds = list(values_by_seed)
    if not seeds:
        return 0.0, 0.0
    means = []
    for _ in range(n):
        pick = rng.choice(seeds, size=len(seeds), replace=True)
        sample = [np.mean(values_by_seed[s]) for s in pick]
        means.append(float(np.mean(sample)))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def post_switch_adaptation(records: list[StepRecord], window: int = 8) -> float | None:
    if len(records) < window + 2:
        return None
    switches = [i for i in range(1, len(records)) if records[i].regime != records[i - 1].regime]
    if not switches:
        return None
    deltas = []
    for idx in switches:
        before = np.array(records[max(0, idx - window) : idx][-1].weights)
        after = np.array(records[min(len(records) - 1, idx + window)].weights)
        deltas.append(float(np.linalg.norm(after - before)))
    return float(np.mean(deltas)) if deltas else None


def run_benchmark(cfg: ArenaConfig, gate: TinyArenaGate, static: np.ndarray) -> dict[str, object]:
    policies: list[PolicyName] = [
        "uniform",
        "expert_seek",
        "expert_wall",
        "expert_evade",
        "static",
        "adaptive",
        "learned",
        "oracle",
    ]
    step_budget = 0
    by_policy_seed: dict[str, dict[int, list[float]]] = {p: {} for p in policies}
    adaptation: dict[str, float | None] = {}
    started = time.perf_counter()

    for policy in policies:
        for seed in cfg.eval_seeds:
            scores = []
            rec: list[StepRecord] = []
            for ep in range(cfg.eval_episodes):
                if step_budget >= cfg.max_step_budget:
                    break
                score, records, used = run_episode(
                    seed + ep * 17,
                    policy,
                    cfg,
                    gate=gate,
                    static=static,
                    record=(policy == "learned" and ep == 0 and seed == cfg.eval_seeds[0]),
                )
                scores.append(score)
                step_budget += used
                if records:
                    rec = records
            by_policy_seed[policy][seed] = scores
            if policy == "learned" and rec:
                adaptation[policy] = post_switch_adaptation(rec)

    means = {
        policy: float(np.mean([v for scores in by_policy_seed[policy].values() for v in scores]))
        for policy in policies
    }
    seed_means = {
        policy: {str(seed): float(np.mean(vals)) for seed, vals in by_policy_seed[policy].items()}
        for policy in policies
    }
    ci = {
        policy: bootstrap_ci(by_policy_seed[policy]) for policy in policies if policy != "oracle"
    }
    learned = means["learned"]
    success = learned > means["uniform"] and learned > means["static"]
    return {
        "means": means,
        "seed_means": seed_means,
        "bootstrap_95_by_seed_cluster": {k: {"low": v[0], "high": v[1]} for k, v in ci.items()},
        "post_switch_weight_delta_mean": adaptation,
        "primary_success": success,
        "hypothesis": (
            "learned gate mean on held-out seeds exceeds uniform and static; "
            "oracle is eval-only upper bound"
        ),
        "step_budget_used": step_budget,
        "runtime_seconds": time.perf_counter() - started,
        "gate_parameters": gate.parameter_count,
        "static_weights": static.tolist(),
    }


def write_replay_html(path: Path, records: list[StepRecord], cfg: ArenaConfig, summary: dict) -> None:
    payload = {
        "records": [
            {
                "t": r.t,
                "regime": REGIME_NAMES[r.regime],
                "agent": list(r.agent),
                "weights": r.weights,
                "reward": r.reward,
                "expert_cf": r.expert_cf,
            }
            for r in records
        ],
        "summary_snippet": {
            "means": summary.get("means"),
            "primary_success": summary.get("primary_success"),
        },
        "assumptions": [
            "Regime id is hidden from the gate; shown in replay for debugging only.",
            "Counterfactual expert rewards computed before each blended move.",
            "Synthetic toy only; not trading or robotics evidence.",
        ],
    }
    data = json.dumps(payload)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Regime Arena Replay</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 1rem; background: #111; color: #eee; }}
#arena {{ display: grid; gap: 2px; width: 360px; height: 360px; background: #222; }}
.cell {{ width: 100%; aspect-ratio: 1; background: #1a2a1a; border-radius: 2px; }}
.cell.wall {{ background: #444; }}
.cell.agent {{ background: #4af; }}
.cell.pred {{ background: #f44; }}
.panel {{ max-width: 720px; }}
.bar {{ height: 12px; background: #333; margin: 4px 0; }}
.fill {{ height: 100%; background: #6c6; }}
</style>
</head>
<body>
<h1>Regime Arena Replay</h1>
<p id="regime"></p>
<input type="range" id="slider" min="0" max="{max(0, len(records) - 1)}" value="0" style="width: 100%"/>
<div id="arena"></div>
<div class="panel" id="weights"></div>
<div class="panel" id="cf"></div>
<div class="panel" id="assumptions"></div>
<script>
const data = {data};
const grid = {cfg.grid};
const arena = document.getElementById('arena');
arena.style.gridTemplateColumns = 'repeat(' + grid + ', 1fr)';
function render(step) {{
  const rec = data.records[step];
  document.getElementById('regime').textContent =
    'Step ' + rec.t + ' · regime ' + rec.regime + ' (debug) · reward ' + rec.reward.toFixed(3);
  arena.innerHTML = '';
  for (let i = 0; i < grid * grid; i++) {{
    const d = document.createElement('div');
    d.className = 'cell';
    const x = Math.floor(i / grid), y = i % grid;
    if (rec.agent[0] === x && rec.agent[1] === y) d.classList.add('agent');
    arena.appendChild(d);
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
        train_episodes=6,
        eval_episodes=4,
        episode_steps=24,
        gate_epochs=4,
        max_step_budget=80_000,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regime Arena train/eval and artifact export")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fast", action="store_true", help="Reduced episodes for CI/smoke runs")
    args = parser.parse_args(argv)
    cfg = fast_config() if args.fast else ArenaConfig()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    static = fit_static_blend(cfg)
    x, y = collect_training_data(cfg, static)
    gate = TinyArenaGate(_rng(17))
    losses = []
    for _ in range(cfg.gate_epochs):
        losses.append(gate.train_epoch(x, y, cfg.gate_lr))

    _, replay_records, _ = run_episode(
        cfg.eval_seeds[0],
        "learned",
        cfg,
        gate=gate,
        static=static,
        record=True,
    )

    summary = run_benchmark(cfg, gate, static)
    summary["schema_version"] = 1
    summary["train_seeds"] = list(cfg.train_seeds)
    summary["eval_seeds"] = list(cfg.eval_seeds)
    summary["training_loss_last"] = float(losses[-1]) if losses else None
    summary["limitations"] = [
        "Partial observability; regime may be inferable from predator feature.",
        "No statistical claim beyond bootstrap cluster CI by seed.",
        "Single short run; null gate result is valid.",
    ]

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    replay = {
        "steps": [
            {
                "t": r.t,
                "regime": r.regime,
                "regime_name": REGIME_NAMES[r.regime],
                "agent": r.agent,
                "weights": r.weights,
                "reward": r.reward,
                "expert_cf": r.expert_cf,
            }
            for r in replay_records
        ]
    }
    (out / "replay.json").write_text(json.dumps(replay, indent=2), encoding="utf-8")
    write_replay_html(out / "replay.html", replay_records, cfg, summary)
    print(json.dumps({"primary_success": summary["primary_success"], "means": summary["means"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())