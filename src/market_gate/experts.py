"""Small, bounded, legible market microstructure experts."""

from math import tanh

EXPERT_IDS = ("microprice", "flow", "reversion")


def clamp(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def microprice_pressure(mid: float, imbalance: float, microprice: float) -> float:
    displacement_bps = 10_000 * (microprice - mid) / mid if mid else 0.0
    return clamp(tanh(imbalance * 1.2 + displacement_bps * 0.25))


def trade_flow_impulse(signed_volume: float, total_volume: float, arrivals: int) -> float:
    ratio = signed_volume / total_volume if total_volume else 0.0
    return clamp(tanh(ratio * min(arrivals, 8) / 3))


def short_reversion(mid: float, fair_value: float) -> float:
    distance_bps = 10_000 * (mid - fair_value) / fair_value if fair_value else 0.0
    return clamp(-tanh(distance_bps * 0.15))
