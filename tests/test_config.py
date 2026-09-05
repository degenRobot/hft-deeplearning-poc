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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gate_interval_ms", 500.5),
        ("stale_after_ms", True),
        ("flow_window_trades", 12.5),
    ],
)
def test_integer_fields_reject_fractional_and_boolean_values(field: str, value: object) -> None:
    config = LabConfig()
    with pytest.raises(ValueError, match=f"{field} must be an integer"):
        config.patch({field: value})


def test_symbol_type_error_is_reported_as_validation_error() -> None:
    config = LabConfig()
    with pytest.raises(ValueError, match="symbol"):
        config.patch({"symbol": 1234567})


def test_replay_symbol_constraint_is_atomic_and_live_symbols_remain_supported() -> None:
    config = LabConfig(source="binance", symbol="ETHUSDT")
    before = config.public()
    with pytest.raises(ValueError, match="replay fixture supports BTCUSDT"):
        config.patch({"source": "replay", "gate_mode": "uniform"})
    assert config.public() == before
    config.patch({"source": "replay", "symbol": "BTCUSDT"})
    assert config.source == "replay"
