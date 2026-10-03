"""REINFORCE (Williams, 1992): Monte Carlo policy gradient, updated once per episode.

The policy gradient ``E[sum_t grad log pi(a_t | s_t) * A_t]`` uses one of three signals
``A_t``, selected by ``baseline``:

* ``none``: the discounted return ``G_t`` itself (vanilla REINFORCE, highest variance).
* ``normalize``: ``G_t`` standardised within the episode (v1.0's choice). It acts as a
  crude baseline and fixes the gradient scale, but is not an unbiased estimator.
* ``value``: ``G_t - V(s_t)`` with a learned state-value network ("REINFORCE with
  baseline", Sutton & Barto, 2018, sec. 13.4). Unbiased, and lower variance.

Fixes relative to v1.0: the constructor honours its arguments (v1.0 silently used the
config module's learning rate and hidden size), and with a value baseline an episode cut
off by the time limit bootstraps from ``V(s_T)`` instead of treating the cut as the end.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from lunarlander_rl.agents.base import Agent, Observation, Transition
from lunarlander_rl.agents.registry import register
from lunarlander_rl.components.networks import mlp
from lunarlander_rl.components.returns import discounted_returns
from lunarlander_rl.config import ConfigError
from lunarlander_rl.seeding import derive_seed

Baseline = Literal["none", "normalize", "value"]


@dataclass(frozen=True)
class ReinforceConfig:
    hidden_sizes: tuple[int, ...] = (128,)
    lr: float = 1e-3
    gamma: float = 0.99
    baseline: Baseline = "normalize"
    value_hidden_sizes: tuple[int, ...] = (128,)
    value_lr: float = 1e-3
    # Weight of the entropy bonus, which discourages premature collapse to one action.
    entropy_coef: float = 0.0
    max_grad_norm: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ConfigError("reinforce: gamma must be in [0, 1]")
        if self.entropy_coef < 0.0:
            raise ConfigError("reinforce: entropy_coef must be non-negative")


def advantages(
    returns: torch.Tensor, baseline: Baseline, values: torch.Tensor | None = None
) -> torch.Tensor:
    """The per-step learning signal for the chosen baseline (see the module docstring)."""
    if baseline == "none":
        return returns
    if baseline == "normalize":
        centred = returns - returns.mean()
        if len(returns) < 2:
            return centred
        return centred / (returns.std() + 1e-8)
    if values is None:
        raise ValueError("the value baseline needs value estimates")
    return returns - values.detach()


@register("reinforce", ReinforceConfig)
class ReinforceAgent(Agent):
    checkpoint_suffix: ClassVar[str] = ".pt"
    metric_names: ClassVar[tuple[str, ...]] = ("policy_loss", "value_loss", "entropy")

    def __init__(
        self,
        config: ReinforceConfig,
        observation_space: gym.Space[Any],
        action_space: gym.spaces.Discrete,
        *,
        seed: int,
        device: str = "cpu",
    ) -> None:
        if not isinstance(observation_space, gym.spaces.Box) or len(observation_space.shape) != 1:
            raise ConfigError("reinforce needs a flat Box observation space")
        self.config = config
        self.device = torch.device(device)
        obs_dim = observation_space.shape[0]
        self.policy = mlp(obs_dim, config.hidden_sizes, int(action_space.n)).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.lr)
        self.value: torch.nn.Sequential | None = None
        self.value_optimizer: torch.optim.Optimizer | None = None
        if config.baseline == "value":
            self.value = mlp(obs_dim, config.value_hidden_sizes, 1).to(self.device)
            self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=config.value_lr)
        # Separate generators: evaluation never consumes training randomness.
        self._generator = torch.Generator().manual_seed(derive_seed(seed, "agent"))
        self._eval_generator = torch.Generator().manual_seed(derive_seed(seed, "eval"))
        self._obs: list[Observation] = []
        self._actions: list[int] = []
        self._rewards: list[float] = []

    def _probabilities(self, obs: Observation) -> torch.Tensor:
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            return torch.softmax(self.policy(obs_tensor.unsqueeze(0)), dim=-1).squeeze(0).cpu()

    def act(self, obs: Observation) -> int:
        probabilities = self._probabilities(obs)
        return int(torch.multinomial(probabilities, 1, generator=self._generator).item())

    def predict(self, obs: Observation, *, deterministic: bool = True) -> int:
        probabilities = self._probabilities(obs)
        if deterministic:
            return int(probabilities.argmax().item())
        return int(torch.multinomial(probabilities, 1, generator=self._eval_generator).item())

    def seed_eval(self, seed: int) -> None:
        self._eval_generator.manual_seed(seed)

    def observe(self, transition: Transition) -> dict[str, float]:
        self._obs.append(transition.obs)
        self._actions.append(transition.action)
        self._rewards.append(transition.reward)
        if not transition.done:
            return {}
        try:
            return self._update(transition)
        finally:
            self._obs, self._actions, self._rewards = [], [], []

    def _update(self, last: Transition) -> dict[str, float]:
        c = self.config
        obs = torch.as_tensor(np.stack(self._obs), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(self._actions, dtype=torch.int64, device=self.device)

        bootstrap = 0.0
        if last.truncated and not last.terminated and self.value is not None:
            with torch.no_grad():
                final = torch.as_tensor(last.next_obs, dtype=torch.float32, device=self.device)
                bootstrap = float(self.value(final.unsqueeze(0)).item())
        returns = torch.as_tensor(
            discounted_returns(self._rewards, c.gamma, bootstrap),
            dtype=torch.float32,
            device=self.device,
        )

        metrics: dict[str, float] = {}
        values = None
        if self.value is not None and self.value_optimizer is not None:
            values = self.value(obs).squeeze(1)
            value_loss = F.mse_loss(values, returns)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            if c.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), c.max_grad_norm)
            self.value_optimizer.step()
            metrics["value_loss"] = float(value_loss.item())

        log_probs = torch.log_softmax(self.policy(obs), dim=-1)
        chosen = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        signal = advantages(returns, c.baseline, values)
        # Summed over the episode: the textbook estimator of the policy gradient.
        policy_loss = -(chosen * signal).sum() - c.entropy_coef * entropy.sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        if c.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), c.max_grad_norm)
        self.optimizer.step()
        metrics["policy_loss"] = float(policy_loss.item())
        metrics["entropy"] = float(entropy.mean().item())
        return metrics

    def save(self, path: Path) -> None:
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "value": None if self.value is None else self.value.state_dict(),
            },
            path,
        )

    def load(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.policy.load_state_dict(state["policy"])
        if self.value is not None:
            if state["value"] is None:
                raise ValueError(f"checkpoint {path} has no value network")
            self.value.load_state_dict(state["value"])
