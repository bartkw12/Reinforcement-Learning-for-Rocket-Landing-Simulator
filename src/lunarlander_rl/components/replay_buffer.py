"""Uniform experience replay backed by preallocated NumPy arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Batch:
    obs: npt.NDArray[np.float32]
    actions: npt.NDArray[np.int64]
    rewards: npt.NDArray[np.float32]
    next_obs: npt.NDArray[np.float32]
    # Only true termination stops bootstrapping. Time-limit truncation is not stored as an
    # episode end at all: from the value function's point of view the state goes on.
    terminated: npt.NDArray[np.float32]


class ReplayBuffer:
    """Fixed-capacity ring buffer; once full, the oldest transitions are overwritten."""

    def __init__(self, capacity: int, obs_shape: tuple[int, ...]) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self._next = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, obs: Any, action: int, reward: float, next_obs: Any, terminated: bool) -> None:
        i = self._next
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.terminated[i] = terminated
        self._next = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch:
        """Sample uniformly with replacement from the stored transitions."""
        if self._size == 0:
            raise ValueError("cannot sample from an empty buffer")
        idx = rng.integers(0, self._size, size=batch_size)
        return Batch(
            obs=self.obs[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_obs=self.next_obs[idx],
            terminated=self.terminated[idx],
        )
