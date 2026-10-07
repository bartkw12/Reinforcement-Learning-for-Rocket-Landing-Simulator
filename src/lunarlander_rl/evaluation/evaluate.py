"""Evaluate a policy on a fixed set of seeded episodes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from lunarlander_rl.agents import Policy, build_agent
from lunarlander_rl.config import ExperimentConfig, build_experiment_config, load_config, to_dict
from lunarlander_rl.envs import Env, make_env
from lunarlander_rl.evaluation.metrics import EpisodeResult, classify_outcome, summarize
from lunarlander_rl.seeding import seed_everything
from lunarlander_rl.tracking import RunPaths


def run_episode(
    env: Env,
    agent: Policy,
    seed: int,
    *,
    deterministic: bool = True,
    success_return: float = 200.0,
) -> EpisodeResult:
    """Run one evaluation episode whose start state is fixed by ``seed``."""
    env_id = env.spec.id if env.spec is not None else None
    agent.seed_eval(seed)
    obs, _ = env.reset(seed=seed)
    episode_return, length = 0.0, 0
    while True:
        action = agent.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_return += float(reward)
        length += 1
        if terminated or truncated:
            break
    return EpisodeResult(
        seed=seed,
        episode_return=episode_return,
        length=length,
        terminated=terminated,
        truncated=truncated,
        outcome=classify_outcome(env_id, obs, float(reward), terminated, truncated),
        success=terminated and episode_return >= success_return,
    )


def evaluate(
    agent: Policy,
    env: Env,
    seeds: Iterable[int],
    *,
    deterministic: bool = True,
    success_return: float = 200.0,
) -> list[EpisodeResult]:
    """Run one episode per seed. Every agent evaluated on the same seeds faces the same
    start states, which removes start-state luck from comparisons."""
    return [
        run_episode(env, agent, seed, deterministic=deterministic, success_return=success_return)
        for seed in seeds
    ]


def report(results: Sequence[EpisodeResult]) -> dict[str, Any]:
    """Summary statistics plus the per-episode results they were computed from."""
    return {"summary": summarize(results), "episodes": [r.to_dict() for r in results]}


def evaluate_checkpoint(
    run_dir: str | Path,
    *,
    checkpoint: str = "final",
    seeds: Iterable[int] | None = None,
    deterministic: bool | None = None,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """Re-evaluate a saved policy from a finished run.

    ``overrides`` change the stored configuration before the environment is built, for
    example ``["env.kwargs.enable_wind=true"]`` to test a policy under conditions it was
    not trained on. Defaults reproduce the run's own final evaluation.
    """
    paths = RunPaths(Path(run_dir))
    config: ExperimentConfig = load_config(paths.config)
    if overrides:
        config = build_experiment_config(to_dict(config), overrides)

    seed_everything(config.seed, torch_threads=config.torch_threads)
    env = make_env(config.env)
    try:
        agent = build_agent(
            config.agent,
            env.observation_space,
            env.action_space,
            seed=config.seed,
            device=config.device,
        )
        agent.load(paths.find_checkpoint(checkpoint))
        results = evaluate(
            agent,
            env,
            config.eval.final_seeds if seeds is None else seeds,
            deterministic=config.eval.deterministic if deterministic is None else deterministic,
            success_return=config.eval.success_return,
        )
    finally:
        env.close()
    return report(results)
