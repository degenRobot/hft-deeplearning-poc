"""Human-editable TOML configuration with conservative bounds."""

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class LabConfig:
    source: str = "replay"
    symbol: str = "BTCUSDT"
    gate_mode: str = "neural"
    higher_level_influence: float = 0.35
    gate_interval_ms: int = 1000
    expert_strength: float = 1.0
    base_spread_bps: float = 2.0
    max_inventory: float = 0.5
    stale_after_ms: int = 2500

    def validate(self) -> None:
        if self.source not in {"replay", "binance"}:
            raise ValueError("source must be replay or binance")
        if self.gate_mode not in {"neural", "uniform", "static"}:
            raise ValueError("gate_mode must be neural, uniform, or static")
        if not 0.0 <= self.higher_level_influence <= 1.0:
            raise ValueError("higher_level_influence must be between 0 and 1")
        if self.gate_interval_ms < 100:
            raise ValueError("gate_interval_ms must be at least 100")
        if self.base_spread_bps <= 0 or self.max_inventory <= 0 or self.stale_after_ms <= 0:
            raise ValueError("risk limits must be positive")

    def patch(self, values: dict[str, object]) -> None:
        permitted = set(asdict(self))
        for key, value in values.items():
            if key not in permitted:
                raise ValueError(f"unknown config field: {key}")
        candidate = LabConfig(**(asdict(self) | values))
        candidate.validate()
        for key, value in asdict(candidate).items():
            setattr(self, key, value)

    def public(self) -> dict[str, object]:
        return asdict(self)


def load_config(path: str | Path) -> LabConfig:
    with Path(path).open("rb") as handle:
        config = LabConfig(**tomllib.load(handle))
    config.validate()
    return config
