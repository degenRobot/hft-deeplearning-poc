from market_gate.gate import TinyMLPGate, blend_and_smooth, static_weights, uniform_weights


def test_all_weight_policies_sum_to_one() -> None:
    previous = uniform_weights()
    neural = TinyMLPGate().predict([(0.0,) * 10] * 30)
    for weights in (previous, static_weights(), neural, blend_and_smooth(neural, previous, 0.4)):
        assert sum(weights.values()) == 1.0
        assert all(value >= 0 for value in weights.values())
