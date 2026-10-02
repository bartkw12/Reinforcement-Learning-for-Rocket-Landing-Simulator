"""Train one agent for one seed and record everything about the run."""

from __future__ import annotations

import math
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lunarlander_rl.agents import Agent, Transition, build_agent
from lunarlander_rl.config import ExperimentConfig, save_config
from lunarlander_rl.envs import Env, make_env
from lunarlander_rl.evaluation import evaluate, report, summarize
from lunarlander_rl.provenance import collect_provenance, utc_now
from lunarlander_rl.seeding import derive_seed, seed_everything
from lunarlander_rl.tracking import CsvLogger, RunPaths, write_json

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


class RunExistsError(FileExistsError):
    """The run directory already holds a completed run."""


def _prepare_run_dir(paths: RunPaths, overwrite: bool) -> None:
    """Create an empty run directory.

    A completed run is only replaced when ``overwrite`` is set. An incomplete one (from a
    crash or an interruption) is always discarded: runs restart from scratch rather than
    resume, because a partial run cannot be continued bit-for-bit.
    """
    if paths.root.exists() and any(paths.root.iterdir()):
        if not paths.config.is_file():
            raise FileExistsError(f"{paths.root} is not empty and is not a run directory")
        if paths.is_complete() and not overwrite:
            raise RunExistsError(f"{paths.root} already holds a completed run")
        shutil.rmtree(paths.root)
    paths.checkpoints.mkdir(parents=True)


def _build_agent(config: ExperimentConfig, env: Env) -> Agent:
    return build_agent(
        config.agent,
        env.observation_space,
        env.action_space,
        seed=config.seed,
        device=config.device,
    )


class _EpisodeMetrics:
    """Averages the diagnostics an agent reports over the course of an episode."""

    def __init__(self, names: tuple[str, ...]) -> None:
        self._names = names
        self.reset()

    def reset(self) -> None:
        self._sums = dict.fromkeys(self._names, 0.0)
        self._counts = dict.fromkeys(self._names, 0)

    def add(self, metrics: Mapping[str, float]) -> None:
        for name, value in metrics.items():
            if name not in self._sums:
                raise KeyError(f"agent reported {name!r}, which is not in its metric_names")
            self._sums[name] += float(value)
            self._counts[name] += 1

    def means(self) -> dict[str, float | str]:
        # An empty cell means the agent made no update during the episode.
        return {
            name: self._sums[name] / self._counts[name] if self._counts[name] else ""
            for name in self._names
        }


def run_experiment(
    config: ExperimentConfig, run_dir: str | Path, *, overwrite: bool = False
) -> dict[str, Any]:
    """Train ``config.agent`` on ``config.env`` and write the run directory.

    Returns the summary of the final evaluation. Given the same configuration, platform
    and package versions, the CSV files and ``final_eval.json`` are reproduced exactly.
    """
    paths = RunPaths(Path(run_dir))
    _prepare_run_dir(paths, overwrite)
    save_config(config, paths.config)
    meta = {**collect_provenance(), "status": "running"}
    write_json(paths.meta, meta)
    started = time.perf_counter()

    seed_everything(config.seed, torch_threads=config.torch_threads)
    train_env = make_env(config.env)
    eval_env = make_env(config.env)
    try:
        agent = _build_agent(config, train_env)
        max_steps = config.train.total_steps or math.inf
        max_episodes = config.train.total_episodes or math.inf
        env_step, episode = 0, 0
        best_mean, best_step = -math.inf, None
        best_path = paths.checkpoint("best", agent.checkpoint_suffix)

        with (
            CsvLogger(paths.train_csv, (*TRAIN_FIELDS, *agent.metric_names)) as train_log,
            CsvLogger(paths.eval_csv, EVAL_FIELDS) as eval_log,
        ):

            def periodic_evaluation() -> None:
                nonlocal best_mean, best_step
                results = evaluate(
                    agent,
                    eval_env,
                    config.eval.seeds,
                    deterministic=config.eval.deterministic,
                    success_return=config.eval.success_return,
                )
                summary = summarize(results)
                eval_log.log(
                    {
                        "env_step": env_step,
                        "episode": episode,
                        **{k: summary[k] for k in EVAL_FIELDS[2:]},
                    }
                )
                if config.train.save_best and summary["mean_return"] > best_mean:
                    best_mean, best_step = summary["mean_return"], env_step
                    agent.save(best_path)

            # Evaluating before any training anchors the learning curve at step 0.
            periodic_evaluation()

            episode_metrics = _EpisodeMetrics(agent.metric_names)
            episode_return, episode_length = 0.0, 0
            train_env.action_space.seed(derive_seed(config.seed, "action_space"))
            obs, _ = train_env.reset(seed=derive_seed(config.seed, "train_env"))

            while env_step < max_steps and episode < max_episodes:
                action = agent.act(obs)
                next_obs, reward, terminated, truncated, _ = train_env.step(action)
                transition = Transition(
                    obs=obs,
                    action=action,
                    reward=float(reward),
                    next_obs=next_obs,
                    terminated=terminated,
                    truncated=truncated,
                )
                episode_metrics.add(agent.observe(transition))
                env_step += 1
                episode_return += transition.reward
                episode_length += 1
                obs = next_obs

                if transition.done:
                    episode += 1
                    train_log.log(
                        {
                            "env_step": env_step,
                            "episode": episode,
                            "return": episode_return,
                            "length": episode_length,
                            "terminated": int(terminated),
                            "truncated": int(truncated),
                            **episode_metrics.means(),
                        }
                    )
                    episode_metrics.reset()
                    episode_return, episode_length = 0.0, 0
                    obs, _ = train_env.reset()

                if env_step % config.eval.interval_steps == 0:
                    periodic_evaluation()

        training_seconds = time.perf_counter() - started
        agent.save(paths.checkpoint("final", agent.checkpoint_suffix))

        def final_evaluation(policy: Agent) -> dict[str, Any]:
            return report(
                evaluate(
                    policy,
                    eval_env,
                    config.eval.final_seeds,
                    deterministic=config.eval.deterministic,
                    success_return=config.eval.success_return,
                )
            )

        final = final_evaluation(agent)
        best: dict[str, Any] | None = None
        if best_step is not None:
            best_agent = _build_agent(config, train_env)
            best_agent.load(best_path)
            best = {"env_step": best_step, **final_evaluation(best_agent)}
    finally:
        train_env.close()
        eval_env.close()

    # The primary result is the final policy. The best checkpoint was selected on the
    # periodic evaluation seeds, so its score on the held-out seeds is also unbiased, but
    # it is reported separately as a secondary result.
    write_json(
        paths.final_eval,
        {
            "env_step": env_step,
            "episode": episode,
            "deterministic": config.eval.deterministic,
            "final": final,
            "best": best,
        },
    )
    meta.update(
        status="completed",
        finished_at=utc_now(),
        env_steps=env_step,
        episodes=episode,
        training_seconds=round(training_seconds, 3),
        total_seconds=round(time.perf_counter() - started, 3),
    )
    write_json(paths.meta, meta)
    summary: dict[str, Any] = final["summary"]
    return summary
