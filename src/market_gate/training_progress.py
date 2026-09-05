"""Optimizer progress and a public-rate compute estimate, never a billing receipt."""

from datetime import UTC, datetime

MODAL_PRICING = {
    "checked_on": "2026-09-05",
    "source_url": "https://modal.com/pricing",
    "cpu_core_second_usd": 0.0000131,
    "memory_gib_second_usd": 0.00000222,
    "cpu_cores": 2,
    "memory_gib": 2,
    "timeout_seconds": 600,
}


def training_progress(state: dict, now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    dataset = state.get("dataset") or {}
    latest = state.get("latest") or {}
    total = dataset.get("epochs", 0) + dataset.get("rl_examples", 0)
    total = total if dataset.get("epochs") and total > 0 else None
    completed = latest.get("step", 0)
    status = state.get("status")
    stage = status if status != "running" else latest.get("phase", "preparing")
    if status == "running" and total and completed >= total:
        stage = "finalizing"

    def elapsed(key):
        value = state.get(key)
        return max(0.0, (now - datetime.fromisoformat(value)).total_seconds()) if value else None

    remote_elapsed = elapsed("execution_started_at")
    rate = (
        MODAL_PRICING["cpu_cores"] * MODAL_PRICING["cpu_core_second_usd"]
        + MODAL_PRICING["memory_gib"] * MODAL_PRICING["memory_gib_second_usd"]
    )
    return {
        "stage": stage,
        "completed_steps": completed,
        "total_steps": total,
        "percent": 100
        if status == "completed"
        else min(99, int(100 * completed / total))
        if total
        else None,
        "elapsed_seconds": elapsed("started_at"),
        "remote_elapsed_seconds": remote_elapsed,
        "compute_estimate_usd": remote_elapsed * rate if remote_elapsed is not None else None,
    }
