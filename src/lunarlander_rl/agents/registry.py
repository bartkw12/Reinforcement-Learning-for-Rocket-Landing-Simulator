"""Registry mapping agent names in config files to implementations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import gymnasium as gym

from lunarlander_rl.agents.base import Policy
from lunarlander_rl.config import AgentConfig, ConfigError, from_dict

A = TypeVar("A", bound=Policy)


@dataclass(frozen=True)
class AgentSpec:
    config_cls: type[Any]
    agent_cls: Callable[..., Policy]


_REGISTRY: dict[str, AgentSpec] = {}


def register(name: str, config_cls: type[Any]) -> Callable[[type[A]], type[A]]:
    """Class decorator registering an agent under ``name``.

    ``config_cls`` is the dataclass of the agent's hyperparameters. The agent is built as
    ``agent_cls(config, observation_space, action_space, seed=..., device=...)``.
    """

    def decorator(agent_cls: type[A]) -> type[A]:
        if name in _REGISTRY:
            raise ValueError(f"agent {name!r} is already registered")
        _REGISTRY[name] = AgentSpec(config_cls=config_cls, agent_cls=agent_cls)
        return agent_cls

    return decorator


def available_agents() -> list[str]:
    return sorted(_REGISTRY)


def build_agent(
    config: AgentConfig,
    observation_space: gym.Space[Any],
    action_space: gym.Space[Any],
    *,
    seed: int,
    device: str = "cpu",
) -> Policy:
    if config.name not in _REGISTRY:
        raise ConfigError(f"unknown agent {config.name!r}; available: {available_agents()}")
    if not isinstance(action_space, gym.spaces.Discrete):
        raise ConfigError(f"agents need a discrete action space, got {action_space}")
    spec = _REGISTRY[config.name]
    params = from_dict(spec.config_cls, config.params, where="agent.params")
    return spec.agent_cls(params, observation_space, action_space, seed=seed, device=device)
