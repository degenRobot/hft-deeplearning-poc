"""Deterministic quote calculation and hard data-quality suppression."""

from dataclasses import dataclass


@dataclass(frozen=True)
class QuoteResult:
    quote: dict[str, float] | None
    reason: str | None


def make_quote(
    *,
    mid: float,
    observed_spread_bps: float,
    signal: float,
    inventory: float,
    now_ms: int,
    message_ts_ms: int,
    base_spread_bps: float,
    max_inventory: float,
    stale_after_ms: int,
) -> QuoteResult:
    if mid <= 0 or observed_spread_bps < 0:
        return QuoteResult(None, "bad_book")
    if now_ms - message_ts_ms > stale_after_ms:
        return QuoteResult(None, "stale_data")
    if abs(inventory) >= max_inventory:
        return QuoteResult(None, "inventory_cap")
    half_bps = max(base_spread_bps / 2, observed_spread_bps / 2)
    fair = mid * (1 + signal * 0.00004 - inventory / max_inventory * 0.00003)
    half = fair * half_bps / 10_000
    return QuoteResult({"bid": round(fair - half, 2), "ask": round(fair + half, 2)}, None)
