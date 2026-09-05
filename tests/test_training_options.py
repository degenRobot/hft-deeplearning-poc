import pytest

from market_gate.training_options import TrainingOptions


def test_options_defaults_and_boundaries():
    options = TrainingOptions()
    assert options.validate() is options
    assert options.parameter_count == 21443
    assert options.to_dict() == {
        "hidden_1": 64,
        "hidden_2": 32,
        "epochs": 12,
        "learning_rate": 0.001,
        "max_rl_steps": 120,
        "seed": 7,
    }
    TrainingOptions(8, 1024, 1, 1e-5, 15, 0).validate()
    TrainingOptions(1024, 8, 50, 0.01, 300, 2**31 - 1).validate()


@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden_1", 1025),
        ("hidden_2", 7),
        ("hidden_1", True),
        ("hidden_2", "64"),
        ("hidden_2", 32.0),
        ("epochs", 51),
        ("epochs", False),
        ("learning_rate", float("nan")),
        ("learning_rate", float("inf")),
        ("learning_rate", float("-inf")),
        ("learning_rate", True),
        ("learning_rate", "0.001"),
        ("learning_rate", 0),
        ("learning_rate", 0.010001),
        ("max_rl_steps", 14),
        ("max_rl_steps", 301),
        ("seed", -1),
        ("seed", 2**31),
    ],
)
def test_options_reject_invalid_types_and_values(field, value):
    with pytest.raises(ValueError, match=field):
        TrainingOptions(**{field: value}).validate()
