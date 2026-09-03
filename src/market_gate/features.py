"""Causal one-second feature frames; no future timestamps enter a frame."""

from collections import deque
from math import sqrt

from .contracts import FeatureFrame


class FeatureBuilder:
    """Maintains only past/current observations and emits ten values per closed second."""

    def __init__(self) -> None:
        self.mids: deque[float] = deque(maxlen=6)
        self.frame_mids: deque[float] = deque(maxlen=30)
        self.current_second: int | None = None
        self.signed_volume = 0.0
        self.total_volume = 0.0
        self.arrivals = 0
        self.quote_updates = 0

    def observe_book(self, mid: float) -> None:
        self.mids.append(mid)
        self.quote_updates += 1

    def observe_trade(self, size: float, aggressor: str) -> None:
        self.total_volume += size
        self.signed_volume += size if aggressor == "buy" else -size
        self.arrivals += 1

    def advance(
        self, timestamp_ms: int, mid: float, spread_bps: float, imbalance: float, microprice: float
    ) -> FeatureFrame | None:
        second = timestamp_ms // 1000
        if self.current_second is None:
            self.current_second = second
            return None
        if second == self.current_second:
            return None
        prior_mid = self.frame_mids[-1] if self.frame_mids else mid
        ret_1 = (mid / prior_mid - 1.0) if prior_mid else 0.0
        ret_5 = (mid / self.frame_mids[0] - 1.0) if self.frame_mids else 0.0
        returns = [
            self.frame_mids[i] / self.frame_mids[i - 1] - 1 for i in range(1, len(self.frame_mids))
        ]
        volatility = sqrt(sum(item * item for item in returns) / len(returns)) if returns else 0.0
        fair = sum(self.frame_mids) / len(self.frame_mids) if self.frame_mids else mid
        micro_disp = (microprice - mid) / mid if mid else 0.0
        flow_ratio = self.signed_volume / self.total_volume if self.total_volume else 0.0
        fair_distance = (mid - fair) / fair if fair else 0.0
        values = (
            ret_1,
            ret_5,
            volatility,
            spread_bps,
            imbalance,
            micro_disp,
            flow_ratio,
            float(self.arrivals),
            float(self.quote_updates),
            fair_distance,
        )
        self.frame_mids.append(mid)
        self.signed_volume = self.total_volume = 0.0
        self.arrivals = self.quote_updates = 0
        self.current_second = second
        return FeatureFrame(timestamp_ms, values, spread_bps >= 0)
