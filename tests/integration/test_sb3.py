"""Stable-Baselines3 agents run through the same pipeline as the project's own agents."""

import csv
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch

pytest.importorskip("stable_baselines3")

from lunarlander_rl.agents import build_agent
from lunarlander_rl.agents.sb3 import SB3DQNAgent, SB3PPOAgent
from lunarlander_rl.config import AgentConfig, ConfigError, build_experiment_config
from lunarlander_rl.evaluation import evaluate_checkpoint
from lunarlander_rl.tracking import read_json
from lunarlander_rl.training import run_experiment

SMALL_AGENTS: dict[str, dict[str, Any]] = {
    "sb3_dqn": {
        "name": "sb3_dqn",
        "params": {
            "net_arch": [32],
            "buffer_size": 1000,
            "learning_starts": 200,
            "batch_size": 32,
            "target_update_interval": 100,
            "exploration_decay_steps": 500,
        },
    },
    "sb3_ppo": {
        "name": "sb3_ppo",
        "params": {"n_envs": 2, "n_steps": 128, "batch_size": 64, "n_epochs": 2, "net_arch": [32]},
    },
}


def make_config(agent: str, **overrides: Any) -> Any:
    raw: dict[str, Any] = {
        "agent": SMALL_AGENTS[agent],
        "seed": 0,
        "train": {"total_steps": 1500},
        "eval": {"interval_steps": 500, "episodes": 2, "final_episodes": 3},
    }
    raw.update(overrides)
    return build_experiment_config(raw)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module", params=sorted(SMALL_AGENTS))
def finished_run(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Any:
    agent = request.param
    run_dir = tmp_path_factory.mktemp(agent) / "run"
    run_experiment(make_config(agent), run_dir)
    return agent, run_dir


def test_run_directory_matches_the_project_protocol(finished_run: Any) -> None:
    agent, run_dir = finished_run
    final_eval = read_json(run_dir / "final_eval.json")
    # The step budget is exact.
    assert final_eval["env_step"] == 1500
    assert [e["seed"] for e in final_eval["final"]["episodes"]] == [10_000, 10_001, 10_002]
    assert (run_dir / "checkpoints" / "final.zip").is_file()

    evaluations = read_rows(run_dir / "eval.csv")
    assert [int(r["env_step"]) for r in evaluations] == [0, 500, 1000, 1500]

    episodes = read_rows(run_dir / "train_episodes.csv")
    assert episodes, "no training episode finished"
    assert all(int(r["terminated"]) + int(r["truncated"]) == 1 for r in episodes)
    metric_columns = SB3DQNAgent.metric_names if agent == "sb3_dqn" else SB3PPOAgent.metric_names
    assert list(episodes[0])[6:] == list(metric_columns)
    # Training diagnostics appear once SB3 has made its first update.
    assert any(episodes[-1][name] for name in metric_columns)


def test_same_seed_reproduces_the_run(finished_run: Any, tmp_path: Path) -> None:
    agent, run_dir = finished_run
    run_experiment(make_config(agent), tmp_path / "again")
    for name in ("train_episodes.csv", "eval.csv", "final_eval.json"):
        assert (tmp_path / "again" / name).read_bytes() == (run_dir / name).read_bytes(), name


def test_evaluation_schedule_does_not_change_training(finished_run: Any, tmp_path: Path) -> None:
    agent, run_dir = finished_run
    config = make_config(agent, eval={"interval_steps": 250, "episodes": 1, "final_episodes": 3})
    run_experiment(config, tmp_path / "more_evals")
    assert (tmp_path / "more_evals" / "train_episodes.csv").read_bytes() == (
        run_dir / "train_episodes.csv"
    ).read_bytes()


@pytest.mark.parametrize("checkpoint", ["final", "best"])
def test_checkpoint_reproduces_the_reported_evaluation(finished_run: Any, checkpoint: str) -> None:
    _, run_dir = finished_run
    reported = read_json(run_dir / "final_eval.json")[checkpoint]
    assert evaluate_checkpoint(run_dir, checkpoint=checkpoint)["episodes"] == reported["episodes"]


def test_vectorised_budget_overshoots_by_less_than_one_step_of_all_envs(tmp_path: Path) -> None:
    config = make_config(
        "sb3_ppo",
        train={"total_steps": 1001},
        eval={"interval_steps": 300, "episodes": 1, "final_episodes": 1},
    )
    run_experiment(config, tmp_path / "run")
    assert read_json(tmp_path / "run" / "final_eval.json")["env_step"] == 1002
    # Evaluations happen at the first step past each multiple of the interval.
    steps = [int(r["env_step"]) for r in read_rows(tmp_path / "run" / "eval.csv")]
    assert steps == [0, 300, 600, 900]


def test_episode_budgets_are_rejected(tmp_path: Path) -> None:
    config = make_config("sb3_dqn", train={"total_steps": None, "total_episodes": 3})
    with pytest.raises(ConfigError, match="step budget"):
        run_experiment(config, tmp_path / "run")


def make_agent(name: str) -> Any:
    env = gym.make("LunarLander-v3")
    agent = build_agent(
        AgentConfig(name, SMALL_AGENTS[name]["params"]),
        env.observation_space,
        env.action_space,
        seed=0,
    )
    return agent, env


def test_exploration_is_annealed_over_absolute_steps() -> None:
    from stable_baselines3.common.vec_env import DummyVecEnv

    agent, env = make_agent("sb3_dqn")
    vec = DummyVecEnv([lambda: env])
    try:
        assert agent._make_model(vec, total_steps=1_000_000).exploration_fraction == 500 / 1e6
        assert agent._make_model(vec, total_steps=100).exploration_fraction == 1.0
    finally:
        vec.close()


def test_untrained_agent_cannot_act() -> None:
    agent, env = make_agent("sb3_ppo")
    env.close()
    with pytest.raises(RuntimeError, match="neither been trained nor loaded"):
        agent.predict(np.zeros(8, dtype=np.float32))


def test_stochastic_ppo_evaluation_is_isolated_and_reproducible() -> None:
    from stable_baselines3.common.vec_env import DummyVecEnv

    agent, env = make_agent("sb3_ppo")
    vec = DummyVecEnv([lambda: env])
    # An untrained policy is spread over all actions, which makes sampling visible.
    agent.model = agent._make_model(vec, total_steps=1000)
    vec.close()
    obs = np.zeros(8, dtype=np.float32)

    def sample(seed: int) -> list[int]:
        agent.seed_eval(seed)
        return [agent.predict(obs, deterministic=False) for _ in range(40)]

    torch.manual_seed(123)
    before = torch.get_rng_state()
    first = sample(7)
    # Sampling did not advance the global generator that training uses.
    assert torch.equal(torch.get_rng_state(), before)
    assert sample(7) == first
    assert sample(8) != first
