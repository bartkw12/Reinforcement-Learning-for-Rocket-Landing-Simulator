"""Book-keeping shared by every training loop.

Whether an agent is trained by this project's loop or runs its own (Stable-Baselines3), it
reports environment steps and finished episodes to a :class:`TrainingRecorder`. The
recorder writes the training log, runs the periodic evaluations on schedule and keeps the
best checkpoint, so every run directory follows the same protocol.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from lunarlander_rl.agents.base import Policy
from lunarlander_rl.config import ExperimentConfig
from lunarlander_rl.envs import Env
from lunarlander_rl.evaluation import evaluate, summarize
from lunarlander_rl.tracking import CsvLogger

TRAIN_FIELDS = ("env_step", "episode", "return", "length", "terminated", "truncated")
EVAL_FIELDS = (
    "env_step",
    "episode",
    "mean_return",
    "std_return",
    "iqm_return",
    "median_return",
    "min_return",
    "max_return",
    "success_rate",
    "mean_length",
)


class TrainingRecorder:
    def __init__(
        self,
        config: ExperimentConfig,
        policy: Policy,
        eval_env: Env,
        train_log: CsvLogger,
        eval_log: CsvLogger,
        best_path: Path,
    ) -> None:
        self.config = config
        self.policy = policy
        self.eval_env = eval_env
        self.train_log = train_log
        self.eval_log = eval_log
        self.best_path = best_path
        self.env_step = 0
        self.episode = 0
        self.best_mean = -math.inf
        self.best_step: int | None = None
        self._next_evaluation = 0

    @property
    def max_steps(self) -> int | None:
        return self.config.train.total_steps

    @property
    def budget_reached(self) -> bool:
        train = self.config.train
        return (train.total_steps is not None and self.env_step >= train.total_steps) or (
            train.total_episodes is not None and self.episode >= train.total_episodes
        )

    def start(self) -> None:
        """Evaluate the untrained policy, which anchors the learning curve at step 0."""
        self._evaluate()
        self._next_evaluation = self.config.eval.interval_steps

    def add_steps(self, count: int = 1) -> None:
        """Count environment steps (one per environment of a vectorised step)."""
        self.env_step += count

    def end_episode(
        self,
        *,
        episode_return: float,
        length: int,
        terminated: bool,
        truncated: bool,
        metrics: Mapping[str, float | str] | None = None,
    ) -> None:
        self.episode += 1
        self.train_log.log(
            {
                "env_step": self.env_step,
                "episode": self.episode,
                "return": episode_return,
                "length": length,
                "terminated": int(terminated),
                "truncated": int(truncated),
                **(metrics or {}),
            }
        )

    def evaluate_if_due(self) -> None:
        """Run the periodic evaluation once ``env_step`` reaches the next multiple of the
        evaluation interval. A vectorised loop can step past that multiple; the evaluation
        then happens at the first step after it, and is logged with the actual step."""
        if self.env_step >= self._next_evaluation:
            self._evaluate()
            interval = self.config.eval.interval_steps
            self._next_evaluation = (self.env_step // interval + 1) * interval

    def _evaluate(self) -> None:
        eval_config = self.config.eval
        summary = summarize(
            evaluate(
                self.policy,
                self.eval_env,
                eval_config.seeds,
                deterministic=eval_config.deterministic,
                success_return=eval_config.success_return,
            )
        )
        self.eval_log.log(
            {
                "env_step": self.env_step,
                "episode": self.episode,
                **{k: summary[k] for k in EVAL_FIELDS[2:]},
            }
        )
        if self.config.train.save_best and summary["mean_return"] > self.best_mean:
            self.best_mean, self.best_step = summary["mean_return"], self.env_step
            self.policy.save(self.best_path)
