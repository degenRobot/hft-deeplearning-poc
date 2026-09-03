import pytest

from market_gate.config import LabConfig


def test_invalid_patch_leaves_previous_configuration_intact() -> None:
    config = LabConfig()
    with pytest.raises(ValueError, match="higher_level_influence"):
        config.patch({"higher_level_influence": 1.1})
    assert config.higher_level_influence == 0.35


def test_symbol_must_be_a_simple_uppercase_market_pair() -> None:
    config = LabConfig()
    with pytest.raises(ValueError, match="symbol"):
        config.patch({"symbol": "btc/usdt"})
    assert config.symbol == "BTCUSDT"


def test_flow_window_has_server_side_bounds() -> None:
    config = LabConfig()
    with pytest.raises(ValueError, match="flow_window_trades"):
        config.patch({"flow_window_trades": 3})
    assert config.flow_window_trades == 64
