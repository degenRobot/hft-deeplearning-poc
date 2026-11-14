# Regime Arena (synthetic toy)

CPU-only demonstration of the Market Gate **pattern**. v2 is exploratory measurement after independent review; the first version's holdout and regime coverage were invalid. This is not evidence about trading or robotics.

## v2 evaluation

Train and evaluation worlds use separate namespaces and are asserted disjoint. Every episode includes all three regimes. A descriptive threshold records whether learned mean reward exceeds uniform and static; a true boolean is not evidence of meaningful adaptation.

**Fixture gate:** on training worlds, each expert must show positive mean counterfactual advantage over the experts' mean in at least one regime, and all regimes must be present. If any expert lacks this advantage, `fixture_valid` is false and the gate results must not be interpreted as adaptation. This is a minimal diagnostic, not proof that a regime-switch task is informative or that an expert is optimal long term.

**Stops:** complete the batch, exceed the simulated transition cap (including counterfactual transitions), encounter non-finite weights or gradients, or reach 90 seconds. The wall-clock check is cooperative at rollout steps and between training epochs; it cannot preempt an individual training epoch or NumPy operation. Budget checks can also overshoot by a transition before raising.

## Baselines

Uniform, each one-hot expert, a static heuristic, causal softmax exponential adaptive baseline, and a learned gate. Static weights are softmaxed sums of immediate counterfactual returns along **uniform-policy** trajectories on the same training world IDs. The supervised collection uses a 50/50 uniform/static blend and may visit different states. The static procedure does not optimize full-episode return. The adaptive baseline observes each expert's one-step counterfactual reward, which gives it information beyond the gate's observation history at inference. The regime-labeled debug comparator is not an upper bound.

## Run

Install the repository's existing dependencies in your chosen environment, then from the repository root:

```sh
PYTHONPATH=src python -m unittest tests.test_regime_arena -v
PYTHONPATH=src python -m market_gate.regime_arena --output-dir /tmp/regime-arena-v2
PYTHONPATH=src python scripts/render_regime_arena.py /tmp/regime-arena-v2
```

Use `--fast` for a small smoke run. The renderer accepts only v2 replay artifacts, including their v2 summary and per-step world grids. Outputs are synthetic JSON and a local replay; they are not committed by the command above.
