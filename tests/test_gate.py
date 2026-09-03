from pathlib import Path

from market_gate.gate import blend_and_smooth, load_numpy_gate, static_weights, uniform_weights


def test_all_weight_policies_sum_to_one() -> None:
    previous = uniform_weights()
    neural = load_numpy_gate(Path("models/gate-demo.npz")).predict([(0.0,) * 10] * 30)
    for weights in (previous, static_weights(), neural, blend_and_smooth(neural, previous, 0.4)):
        assert sum(weights.values()) == 1.0
        assert all(value >= 0 for value in weights.values())
