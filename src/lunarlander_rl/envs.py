"""Environment construction."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from lunarlander_rl.config import EnvConfig

Env = gym.Env[Any, Any]


def make_env(config: EnvConfig, *, render_mode: str | None = None) -> Env:
    """Build the environment described by ``config``.

    The environment is left unseeded: callers seed it through ``env.reset(seed=...)``,
    once for a training environment and once per episode for evaluation.
    """
    kwargs: dict[str, Any] = dict(config.kwargs)
    if config.max_episode_steps is not None:
        kwargs["max_episode_steps"] = config.max_episode_steps
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return gym.make(config.id, **kwargs)
