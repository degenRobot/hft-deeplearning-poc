import pytest

from market_gate.config import LabConfig


def test_invalid_patch_leaves_previous_configuration_intact() -> None:
    config = LabConfig()
    with pytest.raises(ValueError, match="higher_level_influence"):
        config.patch({"higher_level_influence": 1.1})
    assert config.higher_level_influence == 0.35
