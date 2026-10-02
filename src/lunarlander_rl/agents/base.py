"""The interface every agent implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy.typing as npt

Observation = npt.NDArray[Any]


@dataclass(frozen=True)
class Transition:
    """One environment step.

    ``terminated`` and ``truncated`` are kept apart on purpose. Termination is a property
    of the MDP (the lander crashed or came to rest), so value targets must not bootstrap
    past it. Truncation is only the time limit cutting the episode short: the state still
    has a future, and treating it as terminal biases value estimates.
    """

    obs: Observation
    action: int
    reward: float
    next_obs: Observation
    terminated: bool
    truncated: bool

    @property
    def done(self) -> bool:
        """The episode ended, for either reason; the next step starts a new one."""
        return self.terminated or self.truncated


class Agent(ABC):
    """A learning agent for a discrete-action environment.

    The training loop calls :meth:`act` then :meth:`observe` once per environment step.
    All learning happens inside :meth:`observe`: an off-policy agent may update on every
    step, an on-policy agent when ``transition.done`` closes an episode.
    """

    # File extension of this agent's checkpoints, including the dot.
    checkpoint_suffix: ClassVar[str] = ".pt"
    # Names of the diagnostics that :meth:`observe` may return (loss, epsilon, ...).
    metric_names: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def act(self, obs: Observation) -> int:
        """Choose an action while training. May explore and may record internal state."""

    @abstractmethod
    def predict(self, obs: Observation, *, deterministic: bool = True) -> int:
        """Choose an action for evaluation.

        Must not change anything that training depends on: no learning state and no
        training random stream. Runs with and without evaluation then train identically.
        """

    def observe(self, transition: Transition) -> Mapping[str, float]:
        """Learn from one transition and return any diagnostics produced by the update."""
        return {}

    def seed_eval(self, seed: int) -> None:
        """Reseed the randomness used by non-deterministic :meth:`predict` calls.

        Called at the start of every evaluation episode, so that a stochastic evaluation
        depends only on the policy and the episode seed.
        """
        return None

    @abstractmethod
    def save(self, path: Path) -> None:
        """Write everything needed to restore the policy to ``path``."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Restore the policy written by :meth:`save`."""
