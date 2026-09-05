"""Bounded, in-memory contextual-bandit adaptation of the running gate's head."""

import hashlib
import time
from collections import deque
from datetime import UTC, datetime

import numpy as np

from .experts import EXPERT_IDS


class LiveLearner:
    """Called by the runtime owner; one pending action means no off-policy reuse.

    Frozen hidden layers retain the loaded demo representation. Only the 99
    output parameters learn. Reward is a delayed directional proxy, not a fill.
    """

    def __init__(self, engine):
        self.engine = engine
        self.enabled = False
        self.interval_seconds = 10
        self.learning_rate = 0.001
        self.base = (
            {name: getattr(engine.gate, name).copy(order="K") for name in ("w3", "b3")}
            if engine.gate
            else {}
        )
        self.base_version = engine.gate.model_version if engine.gate else "unavailable"
        self.rng = np.random.default_rng(17)
        self.history = deque(maxlen=40)
        self.pending = None
        self.next_sample_ms = 0
        self.valid_since_ms = 0
        self.feed_continuity_generation = engine.feed_continuity_generation
        self.updates = 0
        self.discarded = 0
        self.baseline = 0.0
        self.stage = "off"
        self.error = None
        self.last_update_ms = None
        self.started_ms = None
        self.last_duration_ms = None

    def configure(self, values):
        if set(values) - {"enabled", "interval_seconds", "reset"}:
            raise ValueError("Unknown live learning setting")
        if "enabled" in values and type(values["enabled"]) is not bool:
            raise ValueError("enabled must be boolean")
        if "reset" in values and type(values["reset"]) is not bool:
            raise ValueError("reset must be boolean")
        interval = values.get("interval_seconds", self.interval_seconds)
        if type(interval) is not int or not 5 <= interval <= 60:
            raise ValueError("Choose an update interval from 5 to 60 seconds")
        if values.get("enabled") and (
            self.engine.gate is None or self.engine.config.gate_mode != "neural"
        ):
            raise ValueError("Select the neural gate before enabling RL")
        if values.get("reset"):
            if self.engine.gate:
                for name, array in self.base.items():
                    setattr(self.engine.gate, name, array.copy(order="K"))
                self.engine.gate.model_version = self.base_version
                self.engine.last_gate_ts_ms = -self.engine.config.gate_interval_ms
            self.__init__(self.engine)
        was_enabled = self.enabled
        self.interval_seconds = interval
        if "enabled" in values:
            self.enabled = values["enabled"]
        self.pending = None
        self.next_sample_ms = 0
        self.error = None
        self.stage = "warming_up" if self.enabled else "paused" if self.updates else "off"
        if self.enabled and not was_enabled:
            self.valid_since_ms = int(time.time() * 1000)
        if self.enabled and self.started_ms is None:
            self.started_ms = int(time.time() * 1000)
        return self.status()

    def _fresh(self, now):
        e = self.engine
        return (
            e.feed_status == "running"
            and e._book_fresh(now)
            and e.decision is not None
            and 0 <= now - e.last_receive_ts_ms <= e.config.stale_after_ms
            and 0 <= now - e.last_book_receive_ts_ms <= e.config.stale_after_ms
        )

    def tick(self, now=None):
        now = int(time.time() * 1000) if now is None else now
        if not self.enabled:
            return
        e = self.engine
        if e.feed_continuity_generation != self.feed_continuity_generation:
            self.feed_continuity_generation = e.feed_continuity_generation
            self.discarded += int(self.pending is not None)
            self.pending = None
            self.valid_since_ms = now
        if not self._fresh(now):
            self.discarded += int(self.pending is not None)
            self.pending = None
            self.valid_since_ms = now
            self.stage = "waiting_for_feed"
            return
        if e.gate is None or e.config.gate_mode != "neural":
            self.enabled = False
            self.pending = None
            self.stage = "paused"
            return
        times = list(e.frame_times)
        if (
            len(times) != 30
            or times[0] < self.valid_since_ms
            or any(b - a != 1000 for a, b in zip(times, times[1:]))
            or not 0 <= now - times[-1] <= 2000
        ):
            self.discarded += int(self.pending is not None)
            self.pending = None
            self.stage = "warming_up"
            return
        if self.pending is not None:
            sample = self.pending
            if now < sample["selected_ms"] + 5000:
                self.stage = "observing_reward"
                return
            if now > sample["selected_ms"] + 6500:
                self.pending = None
                self.discarded += 1
                self.stage = "waiting_for_feed"
                return
            # Require a new book actually received after the outcome deadline.
            if e.last_book_receive_ts_ms < sample["selected_ms"] + 5000:
                return
            self.pending = None
            try:
                self._update(sample, now)
            except (ValueError, FloatingPointError, OverflowError):
                self.enabled = False
                self.stage = "failed"
                self.error = "Non-finite learning step rejected; last valid weights retained."
                return
            self.stage = "waiting_for_interval"
            return
        if now < self.next_sample_ms:
            self.stage = "waiting_for_interval"
            return
        features = np.asarray(e.frames, dtype=np.float32)
        if not np.isfinite(features).all():
            self.stage = "warming_up"
            return
        before, h1, h2 = self._forward(features)
        sampling = before.astype(np.float64)
        sampling /= sampling.sum()
        action = int(self.rng.choice(3, p=sampling))
        self.pending = {
            "features": features.copy(),
            "h1": h1,
            "h2": h2,
            "before": before,
            "action": action,
            "score": e.decision.scores[EXPERT_IDS[action]],
            "mid": (e.bid + e.ask) / 2,
            "selected_ms": now,
            "input_ms": times[-1],
        }
        self.next_sample_ms = now + self.interval_seconds * 1000
        self.stage = "observing_reward"

    def _forward(self, features, w3=None, b3=None):
        g = self.engine.gate
        with np.errstate(over="raise", invalid="raise"):
            h1 = np.maximum(features.reshape(-1) @ g.w1 + g.b1, 0)
            h2 = np.maximum(h1 @ g.w2 + g.b2, 0)
            logits = h2 @ (g.w3 if w3 is None else w3) + (g.b3 if b3 is None else b3)
            exponent = np.exp(np.clip(logits - logits.max(), -50, 50))
            probs = exponent / exponent.sum()
        if not np.isfinite(probs).all():
            raise ValueError("invalid probabilities")
        return probs, h1, h2

    def _update(self, sample, now):
        started = time.perf_counter()
        g = self.engine.gate
        prices = (self.engine.bid, self.engine.ask, sample["mid"])
        if any(not np.isfinite(x) or x <= 0 for x in prices) or not np.isfinite(sample["score"]):
            raise ValueError("invalid reward inputs")
        move_bps = 10000 * (((self.engine.bid + self.engine.ask) / 2) / sample["mid"] - 1)
        if not np.isfinite(move_bps):
            raise ValueError("invalid price move")
        raw_reward = sample["score"] * move_bps
        if not np.isfinite(raw_reward):
            raise ValueError("invalid reward")
        reward = float(np.clip(raw_reward, -5, 5))
        advantage = (reward - self.baseline) / 5
        before = sample["before"]
        grad_logits = before.copy()
        grad_logits[sample["action"]] -= 1
        grad_logits *= advantage
        grad_w = np.outer(sample["h2"], grad_logits)
        norm = float(
            np.sqrt(
                np.sum(grad_w.astype(np.float64) ** 2) + np.sum(grad_logits.astype(np.float64) ** 2)
            )
        )
        if not np.isfinite(norm):
            raise ValueError("invalid gradient")
        scale = min(1.0, 1.0 / max(norm, 1e-12))
        next_w = g.w3 - self.learning_rate * scale * grad_w
        next_b = g.b3 - self.learning_rate * scale * grad_logits
        after, _, _ = self._forward(sample["features"], next_w, next_b)
        loss = float(-advantage * np.log(max(float(before[sample["action"]]), 1e-20)))
        if not all(np.isfinite(x).all() for x in (next_w, next_b, loss, reward)):
            raise ValueError("invalid update")
        layers = []
        for name, weight, gradient, updated in (
            ("Hidden 1 · frozen", g.w1, np.zeros(1), g.w1),
            ("Hidden 2 · frozen", g.w2, np.zeros(1), g.w2),
            ("Output weights", g.w3, grad_w * scale, next_w),
            ("Output bias", g.b3, grad_logits * scale, next_b),
        ):
            layers.append(
                {
                    "name": name,
                    "gradient_norm": float(np.linalg.norm(gradient)),
                    "weight_delta_norm": float(np.linalg.norm(updated - weight)),
                    "weight_norm": float(np.linalg.norm(updated)),
                }
            )
        # Single owner, no await: validate all candidates before the atomic head swap.
        g.w3, g.b3 = next_w, next_b
        self.updates += 1
        g.model_version = f"{self.base_version}-live-rl-{self.updates}"
        self.engine.last_gate_ts_ms = -self.engine.config.gate_interval_ms
        self.baseline = 0.9 * self.baseline + 0.1 * reward
        self.last_update_ms = now
        self.last_duration_ms = (time.perf_counter() - started) * 1000
        self.history.append(
            {
                "step": self.updates,
                "phase": "rl",
                "loss": loss,
                "reward": reward,
                "action": sample["action"],
                "input_end_ts_ms": sample["input_ms"],
                "target_ts_ms": self.engine.last_book_receive_ts_ms,
                "selected_ts_ms": sample["selected_ms"],
                "features": sample["features"].tolist(),
                "activations": {
                    "hidden_1": sample["h1"].tolist(),
                    "hidden_2": sample["h2"].tolist(),
                },
                "outputs_before": before.tolist(),
                "outputs_after": after.tolist(),
                "layers": layers,
            }
        )

    def status(self):
        now = int(time.time() * 1000)
        head = self.engine.gate
        return {
            "enabled": self.enabled,
            "interval_seconds": self.interval_seconds,
            "stage": self.stage,
            "updates": self.updates,
            "discarded": self.discarded,
            "source": self.engine.config.source,
            "symbol": self.engine.config.symbol,
            "run_id": self.engine.run_id,
            "model_version": head.model_version if head else None,
            "base_model": self.base_version,
            "trainable_parameters": 99,
            "last_update_ms": self.last_update_ms,
            "step_duration_ms": self.last_duration_ms,
            "next_update_in_seconds": max(
                0,
                (
                    (
                        self.pending["selected_ms"] + 5000
                        if self.pending
                        else self.next_sample_ms + 5000
                    )
                    - now
                )
                / 1000,
            )
            if self.enabled and self.stage not in {"warming_up", "waiting_for_feed"}
            else None,
            "reward": self.history[-1]["reward"] if self.history else None,
            "weight_delta": sum(x["weight_delta_norm"] for x in self.history[-1]["layers"])
            if self.history
            else 0,
            "head_sha256": hashlib.sha256(head.w3.tobytes() + head.b3.tobytes()).hexdigest()
            if head
            else None,
            "error": self.error,
        }

    def training_snapshot(self):
        status = (
            "running"
            if self.enabled
            else "failed"
            if self.error
            else "stopped"
            if self.updates
            else "idle"
        )
        return {
            "schema_version": 1,
            "continuous": True,
            "run_id": self.engine.run_id,
            "status": status,
            "backend": "local",
            "can_stop": False,
            "updated_at": datetime.now(UTC).isoformat(),
            "dataset": None,
            "latest": self.history[-1] if self.history else None,
            "history": list(self.history),
            "evaluation": None,
            "error": self.error,
            "learning": self.status(),
        }
