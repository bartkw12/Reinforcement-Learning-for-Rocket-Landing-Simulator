"""Q-learning with a tile-coded linear action-value function.

Port of the v1.0 "tabular" Q-learning agent. Strictly this is linear function
approximation: Q(s, a) is the sum of one weight per active tile, and the update is
semi-gradient Q-learning (Sutton & Barto, 2018, sec. 10.1 and 9.5.4).

Fixes relative to v1.0:

* The TD target no longer bootstraps from terminal states.
* Training actually explores: v1.0 called the greedy policy during training, so its
  epsilon schedule never had any effect.
* Time-limit truncation ends the episode instead of being ignored, and is not treated
  as termination.
* Tile bounds are configurable (v1.0 rescaled velocities over +-10, so the whole range
  visited in practice fell inside a single tile), tilings use asymmetric offsets, and
  the hash-table size and its collisions are exposed.
* Evaluation never assigns new tiles, so it cannot change what the agent has learned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from lunarlander_rl.agents.base import Agent, Observation, Transition
from lunarlander_rl.agents.registry import register
from lunarlander_rl.components.schedules import EpsilonConfig, EpsilonSchedule
from lunarlander_rl.components.tile_coding import TileCoder, observation_space_bounds
from lunarlander_rl.config import ConfigError
from lunarlander_rl.seeding import make_rng


@dataclass(frozen=True)
class QLearningConfig:
    # Step size for a full update; each of the num_tilings active weights moves by
    # alpha / num_tilings of the TD error.
    alpha: float = 0.3
    gamma: float = 0.99
    epsilon: EpsilonConfig = field(default_factory=EpsilonConfig)
    num_tilings: int = 16
    tiles_per_dim: int = 4
    table_size: int = 2**18
    # (low, high) per continuous dimension. None uses the observation space's bounds,
    # as v1.0 did.
    bounds: tuple[tuple[float, float], ...] | None = None
    # Observation dimensions used as discrete inputs; LunarLander's leg-contact flags.
    discrete_dims: tuple[int, ...] = (6, 7)
    # Initial value of Q(s, a) for every state and action.
    initial_value: float = 0.0
    # Re-introduce v1.0's two learning bugs, to measure their effect in the replication
    # study: training acts greedily (argmax, ties to action 0, no exploration) and the TD
    # target bootstraps from terminal states. Never use otherwise.
    reproduce_v1_bugs: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ConfigError("q_learning: alpha must be in (0, 1]")
        if not 0.0 <= self.gamma <= 1.0:
            raise ConfigError("q_learning: gamma must be in [0, 1]")


@register("q_learning", QLearningConfig)
class QLearningAgent(Agent):
    checkpoint_suffix: ClassVar[str] = ".npz"
    metric_names: ClassVar[tuple[str, ...]] = ("td_error", "epsilon", "table_usage")

    def __init__(
        self,
        config: QLearningConfig,
        observation_space: gym.Space[Any],
        action_space: gym.spaces.Discrete,
        *,
        seed: int,
        device: str = "cpu",
    ) -> None:
        if not isinstance(observation_space, gym.spaces.Box) or len(observation_space.shape) != 1:
            raise ConfigError("q_learning needs a flat Box observation space")
        self.config = config
        self.num_actions = int(action_space.n)
        dims = range(observation_space.shape[0])
        continuous = [d for d in dims if d not in config.discrete_dims]
        bounds = (
            list(config.bounds)
            if config.bounds is not None
            else observation_space_bounds(observation_space.low, observation_space.high, continuous)
        )
        self.coder = TileCoder(
            bounds,
            num_tilings=config.num_tilings,
            tiles_per_dim=config.tiles_per_dim,
            table_size=config.table_size,
            continuous_dims=continuous,
            discrete_dims=config.discrete_dims,
        )
        # One extra column, never updated, stands in for tiles that evaluation meets but
        # training never visited (index -1).
        self.weights = np.full(
            (self.num_actions, config.table_size + 1),
            config.initial_value / config.num_tilings,
            dtype=np.float64,
        )
        self.step_size = config.alpha / config.num_tilings
        self.epsilon = EpsilonSchedule(config.epsilon)
        self._rng = make_rng(seed, "agent")
        self._memo: tuple[bytes, npt.NDArray[np.int64]] | None = None

    def _features(self, obs: Observation) -> npt.NDArray[np.int64]:
        # Each observation is coded up to three times per step (act, then as obs and
        # next_obs in observe); remembering the last one avoids repeating the work.
        key = np.asarray(obs).tobytes()
        if self._memo is None or self._memo[0] != key:
            self._memo = (key, self.coder(obs))
        return self._memo[1]

    def q_values(self, features: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        values: npt.NDArray[np.float64] = self.weights[:, features].sum(axis=1)
        return values

    def act(self, obs: Observation) -> int:
        if self.config.reproduce_v1_bugs:
            return int(np.argmax(self.q_values(self._features(obs))))
        if self._rng.random() < self.epsilon.value:
            return int(self._rng.integers(self.num_actions))
        q = self.q_values(self._features(obs))
        # Break ties at random: with zero-initialised weights every action ties at first,
        # and argmax would always pick action 0.
        return int(self._rng.choice(np.flatnonzero(q == q.max())))

    def predict(self, obs: Observation, *, deterministic: bool = True) -> int:
        # Greedy whatever ``deterministic`` says: the learned policy is the greedy one.
        features = self.coder(obs, readonly=True)
        return int(np.argmax(self.q_values(features)))

    def observe(self, transition: Transition) -> dict[str, float]:
        features = self._features(transition.obs)
        target = transition.reward
        if not transition.terminated or self.config.reproduce_v1_bugs:
            next_q = self.q_values(self._features(transition.next_obs))
            target += self.config.gamma * float(next_q.max())
        td_error = target - float(self.weights[transition.action, features].sum())
        # add.at accumulates correctly if hash collisions repeat an index.
        np.add.at(self.weights[transition.action], features, self.step_size * td_error)

        metrics = {
            "td_error": abs(td_error),
            "epsilon": self.epsilon.value,
            "table_usage": self.coder.table.usage,
        }
        self.epsilon.step()
        if transition.done:
            self.epsilon.end_episode()
        return metrics

    def save(self, path: Path) -> None:
        table = self.coder.table.state_dict()
        with path.open("wb") as handle:
            np.savez(
                handle,
                weights=self.weights,
                table_keys=table["keys"],
                table_values=table["values"],
                table_collisions=np.int64(table["collisions"]),
                epsilon=np.float64(self.epsilon.value),
            )

    def load(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            if data["weights"].shape != self.weights.shape:
                raise ValueError(f"checkpoint {path} does not match this agent's configuration")
            self.weights = data["weights"].copy()
            self.coder.table.load_state_dict(
                {
                    "keys": data["table_keys"],
                    "values": data["table_values"],
                    "collisions": int(data["table_collisions"]),
                }
            )
            self.epsilon.value = float(data["epsilon"])
        self._memo = None
