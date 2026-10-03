"""Exploration schedules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from lunarlander_rl.config import ConfigError


@dataclass(frozen=True)
class EpsilonConfig:
    """Epsilon-greedy exploration schedule.

    ``linear_steps`` anneals from ``start`` to ``end`` over ``decay_steps`` environment steps
    (the usual choice for DQN). ``exponential_episodes`` multiplies epsilon by ``decay_rate``
    after every episode, down to ``end`` (the v1.0 behaviour). Per-step schedules make runs
    with different episode lengths explore for the same number of steps.
    """

    start: float = 1.0
    end: float = 0.05
    schedule: Literal["linear_steps", "exponential_episodes"] = "linear_steps"
    decay_steps: int = 100_000
    decay_rate: float = 0.995

    def __post_init__(self) -> None:
        if not 0.0 <= self.end <= self.start <= 1.0:
            raise ConfigError("epsilon: need 0 <= end <= start <= 1")
        if self.decay_steps <= 0:
            raise ConfigError("epsilon.decay_steps must be positive")
        if not 0.0 < self.decay_rate <= 1.0:
            raise ConfigError("epsilon.decay_rate must be in (0, 1]")


class EpsilonSchedule:
    def __init__(self, config: EpsilonConfig) -> None:
        self.config = config
        self.value = config.start
        self._steps = 0

    def step(self) -> None:
        """Advance by one environment step."""
        self._steps += 1
        c = self.config
        if c.schedule == "linear_steps":
            fraction = min(1.0, self._steps / c.decay_steps)
            self.value = c.start + fraction * (c.end - c.start)

    def end_episode(self) -> None:
        c = self.config
        if c.schedule == "exponential_episodes":
            self.value = max(c.end, self.value * c.decay_rate)
