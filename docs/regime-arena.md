# Regime Arena (synthetic toy)

CPU-only demonstration of the Market Gate **pattern**. **v2** is revised exploratory measurement after independent review; v1 artifacts at `artifacts/regime-arena` are superseded for interpretation but preserved on disk.

## v2 evaluation (frozen before run)

**Eval world IDs:** integers `2_000_000 + eval_seed * 100 + episode` for `eval_seeds=(2001, 2099)` and `episode in 0..eval_episodes-1`. **Train world IDs:** `1_000_000 + train_seed * 100 + episode` for `train_seeds=(1007, 1017, 1027)`. Sets are asserted disjoint by construction.

**Hypothesis (descriptive only):** on held-out worlds with all three regimes present each episode, a tiny gate trained on train IDs alone may exceed uniform and static on mean pellet reward per step.

**Literal threshold (not “robust success”):** `learned_mean > uniform_mean` and `learned_mean > static_mean` on the frozen eval IDs. This boolean does not imply meaningful effect size, valid fixture, or trading/robotics utility.

**Stop:** complete train+eval batch, or global simulated transition budget (including counterfactual rollouts), or 90s wall clock per CLI invocation, or non-finite weights.

**Fixture gate:** before interpreting the learned gate, a diagnostic must show each expert has positive mean counterfactual advantage in at least one regime on train worlds (one-hot execution path). If not, mark `fixture_invalid` and do not claim adaptation.

## Baselines

Uniform, each expert (one-hot via `blend_move`), static blend fit on **the same train episodes** as collection, softmax exponential adaptive baseline, learned gate. Oracle/regime-labeled policy is optional debug only — **not** an upper bound.

## Run

```sh
cd /Users/msmart/CodexWorktrees/idea-factory-20261114-hft
PYTHONPATH=src /Users/msmart/local-model-gym/.venv/bin/python -m market_gate.regime_arena \
  --output-dir /Users/msmart/.local/state/idea-factory/run-20261114/artifacts/regime-arena-v2
```

Artifacts stay outside Git. Use `--fast` for CI smoke (still must visit all three regimes).