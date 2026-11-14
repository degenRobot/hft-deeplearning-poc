# Regime Arena (synthetic toy)

CPU-only demonstration of the Market Gate **pattern**: three legible movement experts, a tiny softmax gate over a short feature history, and deterministic safety outside the network. This is a **mechanism toy**, not evidence of trading, robotics, or novel ML.

## Frozen hypothesis (before evaluation)

When the arena’s latent rule regime switches mid-episode (open field → maze → predator patrol), a tiny learned gate trained only on held-in world seeds will achieve higher mean **pellet score per step** on held-out seeds than:

- uniform blending,
- each fixed expert alone,
- a static blend fit only on training rollouts,
- a simple causal adaptive reweighting baseline,

and will remain below an **oracle** policy that sees the true regime id (evaluation comparator only).

**Primary success check (held-out seeds 42 and 99 only):** learned gate mean score strictly greater than uniform **and** static, reported with seed-level means and a bootstrap 95% interval clustered by seed. **Null result is valid:** if the gate does not win, record falsification without tuning on eval seeds.

**Stop conditions:** complete the preregistered train batch (seeds 7, 17, 27) and eval batch (42, 99); or 500k environment steps across all policies in one run; or non-finite weights; or 90s wall clock per CLI invocation.

## Observable features

The gate never receives the true regime id. Inputs are: pellet bearing, wall proximity, predator proximity, recent collection rate, and short history (8 frames). Regime id appears only in debug telemetry, HTML replay labels, and the oracle comparator.

## Baselines

| Policy | Description |
| --- | --- |
| `uniform` | Equal expert weights |
| `expert_seek` / `expert_wall` / `expert_evade` | Single expert only |
| `static` | Fixed blend from training-set linear score fit (no eval tuning) |
| `adaptive` | Causal exponential weights from per-step expert counterfactual rewards |
| `learned` | NumPy MLP gate |
| `oracle` | Regime-conditioned best expert (eval only) |

All non-oracle policies share the same observations and step feedback. Counterfactual expert rewards are computed each step before blending.

## Run

```sh
/Users/msmart/local-model-gym/.venv/bin/python -m market_gate.regime_arena \
  --output-dir /Users/msmart/.local/state/idea-factory/run-20261114/artifacts/regime-arena
```

Artifacts are written outside Git under the chosen output directory: `summary.json`, `replay.json`, `replay.html`.