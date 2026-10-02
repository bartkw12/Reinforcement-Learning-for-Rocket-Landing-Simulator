"""Uniformly random policy: the floor every learning agent must beat."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np

from lunarlander_rl.agents.base import Agent, Observation
from lunarlander_rl.agents.registry import register
from lunarlander_rl.seeding import make_rng


@dataclass(frozen=True)
class RandomAgentConfig:
    """The random agent has no hyperparameters."""


@register("random", RandomAgentConfig)
class RandomAgent(Agent):
    checkpoint_suffix: ClassVar[str] = ".json"

    def __init__(
        self,
        config: RandomAgentConfig,
        observation_space: gym.Space[Any],
        action_space: gym.spaces.Discrete,
        *,
        seed: int,
        device: str = "cpu",
    ) -> None:
        self.num_actions = int(action_space.n)
        self._rng = make_rng(seed, "agent")
        self._eval_rng = np.random.default_rng(seed)

    def act(self, obs: Observation) -> int:
        return int(self._rng.integers(self.num_actions))

    def predict(self, obs: Observation, *, deterministic: bool = True) -> int:
        # A uniform policy has no greedy action, so ``deterministic`` has no effect.
        return int(self._eval_rng.integers(self.num_actions))

    def seed_eval(self, seed: int) -> None:
        self._eval_rng = np.random.default_rng(seed)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"agent": "random", "num_actions": self.num_actions}) + "\n")

    def load(self, path: Path) -> None:
        saved = json.loads(path.read_text())
        if saved.get("num_actions") != self.num_actions:
            raise ValueError(f"checkpoint {path} was saved for a different action space")
