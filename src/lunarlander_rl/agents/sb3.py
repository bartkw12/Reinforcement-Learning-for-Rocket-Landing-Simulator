"""Stable-Baselines3 DQN and PPO as reference baselines.

SB3 runs its own training loop. These adapters connect it to the project's
:class:`~lunarlander_rl.training.recorder.TrainingRecorder` through an SB3 callback, so an SB3
run produces the same run directory as the project's own agents: the same training log, the
same periodic evaluation schedule and the same final evaluation on the held-out seeds.
Evaluation always goes through this project's code, never SB3's ``evaluate_policy``.

Stable-Baselines3 is an optional dependency (``pip install -e ".[sb3]"``) and is imported
only when an SB3 agent is trained or loaded.

Two details differ from the project's own loop:

* The step budget is enforced exactly by stopping SB3 from the callback. For PPO, the last
  rollout is then collected but not trained on.
* Training-episode returns are summed from the float32 rewards of SB3's vectorised
  environment, so they can differ from float64 sums in the last digits. Evaluations are
  unaffected: they run in the project's own environments.
"""

from __future__ import annotations

import sys
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import gymnasium as gym
import numpy as np
import torch

from lunarlander_rl.agents.base import Observation, SelfTrainingAgent
from lunarlander_rl.agents.registry import register
from lunarlander_rl.config import ConfigError
from lunarlander_rl.seeding import derive_seed

if TYPE_CHECKING:
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecEnv

    from lunarlander_rl.envs import Env
    from lunarlander_rl.training.recorder import TrainingRecorder


@dataclass(frozen=True)
class SB3DQNConfig:
    """Arguments of ``stable_baselines3.DQN``. Defaults are SB3's own."""

    learning_rate: float = 1e-4
    buffer_size: int = 1_000_000
    learning_starts: int = 100
    batch_size: int = 32
    gamma: float = 0.99
    train_freq: int = 4
    # -1 performs as many gradient steps as environment steps were collected.
    gradient_steps: int = 1
    target_update_interval: int = 10_000
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    # Environment steps over which epsilon is annealed. SB3's exploration_fraction is
    # relative to the training budget; an absolute count keeps exploration the same
    # whatever the budget (and identical to this project's DQN).
    exploration_decay_steps: int = 100_000
    max_grad_norm: float = 10.0
    net_arch: tuple[int, ...] = (64, 64)

    def __post_init__(self) -> None:
        if self.exploration_decay_steps <= 0:
            raise ConfigError("sb3_dqn.exploration_decay_steps must be positive")


@dataclass(frozen=True)
class SB3PPOConfig:
    """Arguments of ``stable_baselines3.PPO``. Defaults are SB3's own."""

    # Parallel environments, stepped in lockstep in one process.
    n_envs: int = 1
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # Hidden layers of both the policy and the value network.
    net_arch: tuple[int, ...] = (64, 64)
    activation: Literal["tanh", "relu"] = "tanh"

    def __post_init__(self) -> None:
        if self.n_envs <= 0:
            raise ConfigError("sb3_ppo.n_envs must be positive")


def _require_sb3() -> Any:
    try:
        import stable_baselines3
    except ImportError as error:
        raise ImportError(
            'Stable-Baselines3 agents need the optional extra: pip install -e ".[sb3]"'
        ) from error
    return stable_baselines3


def _recorder_callback(recorder: TrainingRecorder, agent: _SB3Agent) -> BaseCallback:
    """An SB3 callback that reports every vectorised step to ``recorder``."""
    from stable_baselines3.common.callbacks import BaseCallback

    class RecorderCallback(BaseCallback):
        def _on_training_start(self) -> None:
            n_envs = self.training_env.num_envs
            self._returns = np.zeros(n_envs, dtype=np.float64)
            self._lengths = np.zeros(n_envs, dtype=np.int64)

        def _on_step(self) -> bool:
            rewards, dones, infos = (
                self.locals["rewards"],
                self.locals["dones"],
                self.locals["infos"],
            )
            self._returns += rewards
            self._lengths += 1
            recorder.add_steps(len(dones))
            for i in np.flatnonzero(dones):
                truncated = bool(infos[i].get("TimeLimit.truncated", False))
                recorder.end_episode(
                    episode_return=float(self._returns[i]),
                    length=int(self._lengths[i]),
                    terminated=not truncated,
                    truncated=truncated,
                    metrics=agent.latest_metrics(),
                )
                self._returns[i], self._lengths[i] = 0.0, 0
            recorder.evaluate_if_due()
            # Returning False stops SB3 at exactly the step budget.
            return not recorder.budget_reached

    return RecorderCallback()


class _SB3Agent(SelfTrainingAgent):
    checkpoint_suffix: ClassVar[str] = ".zip"
    algorithm: ClassVar[str]
    # Our metric name -> key in SB3's logger.
    logger_keys: ClassVar[dict[str, str]]

    def __init__(
        self,
        config: Any,
        observation_space: gym.Space[Any],
        action_space: gym.spaces.Discrete,
        *,
        seed: int,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.device = device
        self.seed = derive_seed(seed, "agent")
        self.model: BaseAlgorithm | None = None

    @property
    def n_envs(self) -> int:
        return 1

    @abstractmethod
    def _make_model(self, env: VecEnv, total_steps: int) -> BaseAlgorithm:
        """Build the SB3 algorithm for a run of ``total_steps`` environment steps."""

    def _require_model(self) -> BaseAlgorithm:
        if self.model is None:
            raise RuntimeError(f"{self.algorithm} agent has neither been trained nor loaded")
        return self.model

    def learn(self, env_factory: Callable[[], Env], recorder: TrainingRecorder) -> None:
        total_steps = recorder.max_steps
        if total_steps is None:
            raise ConfigError("Stable-Baselines3 agents need a step budget (train.total_steps)")
        _require_sb3()
        from stable_baselines3.common.logger import Logger
        from stable_baselines3.common.vec_env import DummyVecEnv

        env = DummyVecEnv([env_factory] * self.n_envs)
        try:
            self.model = self._make_model(env, total_steps)
            # A logger with no outputs: values stay readable for latest_metrics().
            self.model.set_logger(Logger(folder=None, output_formats=[]))
            recorder.start()
            self.model.learn(
                total_timesteps=total_steps,
                callback=_recorder_callback(recorder, self),
                # Never dump the logger: dumping clears the values latest_metrics() reads.
                log_interval=sys.maxsize,
                progress_bar=False,
            )
        finally:
            env.close()

    def latest_metrics(self) -> dict[str, float | str]:
        """The most recent training statistics SB3 has logged; empty before the first
        update."""
        values = self._require_model().logger.name_to_value
        return {name: float(values[key]) for name, key in self.logger_keys.items() if key in values}

    def predict(self, obs: Observation, *, deterministic: bool = True) -> int:
        action, _ = self._require_model().predict(obs, deterministic=deterministic)
        return int(np.asarray(action).item())

    def save(self, path: Path) -> None:
        self._require_model().save(path)

    def load(self, path: Path) -> None:
        algorithm_class = getattr(_require_sb3(), self.algorithm)
        self.model = algorithm_class.load(path, device=self.device)


@register("sb3_dqn", SB3DQNConfig)
class SB3DQNAgent(_SB3Agent):
    algorithm: ClassVar[str] = "DQN"
    metric_names: ClassVar[tuple[str, ...]] = ("loss", "epsilon")
    logger_keys: ClassVar[dict[str, str]] = {
        "loss": "train/loss",
        "epsilon": "rollout/exploration_rate",
    }

    def _make_model(self, env: VecEnv, total_steps: int) -> BaseAlgorithm:
        from stable_baselines3 import DQN

        c: SB3DQNConfig = self.config
        return DQN(
            "MlpPolicy",
            env,
            learning_rate=c.learning_rate,
            buffer_size=c.buffer_size,
            learning_starts=c.learning_starts,
            batch_size=c.batch_size,
            gamma=c.gamma,
            train_freq=c.train_freq,
            gradient_steps=c.gradient_steps,
            target_update_interval=c.target_update_interval,
            exploration_initial_eps=c.exploration_initial_eps,
            exploration_final_eps=c.exploration_final_eps,
            exploration_fraction=min(1.0, c.exploration_decay_steps / total_steps),
            max_grad_norm=c.max_grad_norm,
            policy_kwargs={"net_arch": list(c.net_arch)},
            seed=self.seed,
            device=self.device,
        )

    def predict(self, obs: Observation, *, deterministic: bool = True) -> int:
        # Always greedy, like this project's DQN. SB3's non-deterministic DQN prediction is
        # epsilon-greedy on NumPy's global generator, which training also draws from.
        return super().predict(obs, deterministic=True)


@register("sb3_ppo", SB3PPOConfig)
class SB3PPOAgent(_SB3Agent):
    algorithm: ClassVar[str] = "PPO"
    metric_names: ClassVar[tuple[str, ...]] = (
        "policy_loss",
        "value_loss",
        "entropy_loss",
        "approx_kl",
        "clip_fraction",
    )
    logger_keys: ClassVar[dict[str, str]] = {
        "policy_loss": "train/policy_gradient_loss",
        "value_loss": "train/value_loss",
        "entropy_loss": "train/entropy_loss",
        "approx_kl": "train/approx_kl",
        "clip_fraction": "train/clip_fraction",
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._eval_rng_state = torch.Generator().manual_seed(0).get_state()

    @property
    def n_envs(self) -> int:
        n: int = self.config.n_envs
        return n

    def _make_model(self, env: VecEnv, total_steps: int) -> BaseAlgorithm:
        from stable_baselines3 import PPO

        c: SB3PPOConfig = self.config
        layers = list(c.net_arch)
        return PPO(
            "MlpPolicy",
            env,
            learning_rate=c.learning_rate,
            n_steps=c.n_steps,
            batch_size=c.batch_size,
            n_epochs=c.n_epochs,
            gamma=c.gamma,
            gae_lambda=c.gae_lambda,
            clip_range=c.clip_range,
            ent_coef=c.ent_coef,
            vf_coef=c.vf_coef,
            max_grad_norm=c.max_grad_norm,
            policy_kwargs={
                "net_arch": {"pi": layers, "vf": layers},
                "activation_fn": torch.nn.Tanh if c.activation == "tanh" else torch.nn.ReLU,
            },
            seed=self.seed,
            device=self.device,
        )

    def seed_eval(self, seed: int) -> None:
        self._eval_rng_state = torch.Generator().manual_seed(seed).get_state()

    def predict(self, obs: Observation, *, deterministic: bool = True) -> int:
        if deterministic:
            return super().predict(obs, deterministic=True)
        # SB3 samples actions from PyTorch's global generator, which PPO's training also
        # uses. Sample from a separate, per-episode-seeded state so that stochastic
        # evaluation never changes training and depends only on the episode seed.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self._eval_rng_state)
            action = super().predict(obs, deterministic=False)
            self._eval_rng_state = torch.get_rng_state()
        return action
