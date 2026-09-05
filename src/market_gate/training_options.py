"""Shared, dependency-free limits for the teaching training run."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrainingOptions:
    hidden_1: int = 64
    hidden_2: int = 32
    epochs: int = 12
    learning_rate: float = 0.001
    max_rl_steps: int = 120
    seed: int = 7

    def validate(self) -> TrainingOptions:
        for name, lower, upper in (
            ("hidden_1", 8, 1024),
            ("hidden_2", 8, 1024),
            ("epochs", 1, 50),
            ("max_rl_steps", 15, 300),
            ("seed", 0, 2**31 - 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in {lower}..{upper}")
        if (
            type(self.learning_rate) not in (int, float)
            or not 1e-5 <= self.learning_rate <= 0.01
            or not math.isfinite(self.learning_rate)
        ):
            raise ValueError("learning_rate must be a finite number in 0.00001..0.01")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def parameter_count(self) -> int:
        return 301 * self.hidden_1 + (self.hidden_1 + 1) * self.hidden_2 + (self.hidden_2 + 1) * 3
