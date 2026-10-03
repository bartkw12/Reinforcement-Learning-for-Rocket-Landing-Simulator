"""Deep Q-Network (Mnih et al., 2015), with optional Double DQN targets (van Hasselt et al., 2016).

Fixes relative to v1.0:

* The replay buffer stores ``terminated`` rather than ``terminated or truncated``, so the
  1000-step time limit no longer cuts off bootstrapping.
* The debug ``print`` of every training batch is gone.
* The target network syncs every N environment steps by default (v1.0 synced every 10
  episodes, which is still available for the replication study).
* Learning starts after a configurable warm-up, gradients can be clipped, the loss can be
  Huber or MSE, and exploration can follow a per-step schedule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import gymnasium as gym
import torch
import torch.nn.functional as F

from lunarlander_rl.agents.base import Agent, Observation, Transition
from lunarlander_rl.agents.registry import register
from lunarlander_rl.components.networks import mlp
from lunarlander_rl.components.replay_buffer import ReplayBuffer
from lunarlander_rl.components.schedules import EpsilonConfig, EpsilonSchedule
from lunarlander_rl.config import ConfigError
from lunarlander_rl.seeding import make_rng


@dataclass(frozen=True)
class DQNConfig:
    hidden_sizes: tuple[int, ...] = (64, 64)
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 100_000
    # Environment steps collected before the first gradient update.
    learning_starts: int = 1_000
    # One round of ``gradient_steps`` updates every ``train_frequency`` environment steps.
    train_frequency: int = 1
    gradient_steps: int = 1
    target_update_interval: int = 1_000
    target_update_unit: Literal["steps", "episodes"] = "steps"
    double: bool = False
    loss: Literal["huber", "mse"] = "huber"
    max_grad_norm: float | None = None
    epsilon: EpsilonConfig = field(default_factory=EpsilonConfig)

    def __post_init__(self) -> None:
        for name in (
            "batch_size",
            "buffer_size",
            "train_frequency",
            "gradient_steps",
            "target_update_interval",
        ):
            if getattr(self, name) <= 0:
                raise ConfigError(f"dqn.{name} must be positive")
        if self.learning_starts < 0:
            raise ConfigError("dqn.learning_starts must be non-negative")


def td_targets(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    next_q_target: torch.Tensor,
    gamma: float,
    next_q_online: torch.Tensor | None = None,
) -> torch.Tensor:
    """One-step TD targets ``r + gamma * (1 - terminated) * Q'(s', a*)``.

    ``a*`` maximises the target network's values (DQN) or, when ``next_q_online`` is
    given, the online network's values (Double DQN), which reduces overestimation.
    """
    if next_q_online is None:
        next_values = next_q_target.max(dim=1).values
    else:
        best = next_q_online.argmax(dim=1, keepdim=True)
        next_values = next_q_target.gather(1, best).squeeze(1)
    return rewards + gamma * (1.0 - terminated) * next_values


@register("dqn", DQNConfig)
class DQNAgent(Agent):
    checkpoint_suffix: ClassVar[str] = ".pt"
    metric_names: ClassVar[tuple[str, ...]] = ("loss", "q_mean", "epsilon")

    def __init__(
        self,
        config: DQNConfig,
        observation_space: gym.Space[Any],
        action_space: gym.spaces.Discrete,
        *,
        seed: int,
        device: str = "cpu",
    ) -> None:
        if not isinstance(observation_space, gym.spaces.Box) or len(observation_space.shape) != 1:
            raise ConfigError("dqn needs a flat Box observation space")
        self.config = config
        self.device = torch.device(device)
        self.num_actions = int(action_space.n)
        obs_dim = observation_space.shape[0]
        self.q_network = mlp(obs_dim, config.hidden_sizes, self.num_actions).to(self.device)
        self.target_network = mlp(obs_dim, config.hidden_sizes, self.num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=config.lr)
        self.buffer = ReplayBuffer(config.buffer_size, (obs_dim,))
        self.epsilon = EpsilonSchedule(config.epsilon)
        self._rng = make_rng(seed, "agent")
        self._steps = 0
        self._episodes = 0

    def _greedy(self, obs: Observation) -> int:
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            return int(self.q_network(obs_tensor.unsqueeze(0)).argmax(dim=1).item())

    def act(self, obs: Observation) -> int:
        if self._rng.random() < self.epsilon.value:
            return int(self._rng.integers(self.num_actions))
        return self._greedy(obs)

    def predict(self, obs: Observation, *, deterministic: bool = True) -> int:
        # Greedy whatever ``deterministic`` says: the learned policy is the greedy one.
        return self._greedy(obs)

    def observe(self, transition: Transition) -> dict[str, float]:
        c = self.config
        self.buffer.add(
            transition.obs,
            transition.action,
            transition.reward,
            transition.next_obs,
            transition.terminated,
        )
        metrics = {"epsilon": self.epsilon.value}
        self._steps += 1
        self.epsilon.step()

        if (
            self._steps >= c.learning_starts
            and len(self.buffer) >= c.batch_size
            and self._steps % c.train_frequency == 0
        ):
            losses, q_means = zip(*(self._update() for _ in range(c.gradient_steps)), strict=True)
            metrics["loss"] = sum(losses) / len(losses)
            metrics["q_mean"] = sum(q_means) / len(q_means)

        if c.target_update_unit == "steps" and self._steps % c.target_update_interval == 0:
            self.sync_target()
        if transition.done:
            self._episodes += 1
            self.epsilon.end_episode()
            if (
                c.target_update_unit == "episodes"
                and self._episodes % c.target_update_interval == 0
            ):
                self.sync_target()
        return metrics

    def _update(self) -> tuple[float, float]:
        c = self.config
        batch = self.buffer.sample(c.batch_size, self._rng)

        def tensor(array: Any) -> torch.Tensor:
            return torch.as_tensor(array, device=self.device)

        obs, next_obs = tensor(batch.obs), tensor(batch.next_obs)
        q_values = self.q_network(obs).gather(1, tensor(batch.actions).unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            targets = td_targets(
                tensor(batch.rewards),
                tensor(batch.terminated),
                self.target_network(next_obs),
                c.gamma,
                self.q_network(next_obs) if c.double else None,
            )
        if c.loss == "huber":
            loss = F.smooth_l1_loss(q_values, targets)
        else:
            loss = F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        if c.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), c.max_grad_norm)
        self.optimizer.step()
        return float(loss.item()), float(q_values.mean().item())

    def sync_target(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, path: Path) -> None:
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
            },
            path,
        )

    def load(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.q_network.load_state_dict(state["q_network"])
        self.target_network.load_state_dict(state["target_network"])
