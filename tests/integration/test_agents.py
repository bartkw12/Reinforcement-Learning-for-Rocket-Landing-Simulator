"""Every learning agent, end to end through the training loop on LunarLander."""

from pathlib import Path
from typing import Any

import gymnasium as gym
import pytest

from lunarlander_rl.agents import build_agent
from lunarlander_rl.config import build_experiment_config, load_yaml
from lunarlander_rl.evaluation import evaluate_checkpoint
from lunarlander_rl.tracking import read_json
from lunarlander_rl.training import run_experiment

REPO_ROOT = Path(__file__).resolve().parents[2]

# Small settings so that a run takes about a second.
SMALL_AGENTS: dict[str, dict[str, Any]] = {
    "q_learning": {
        "name": "q_learning",
        "params": {"num_tilings": 4, "table_size": 4096, "epsilon": {"decay_steps": 500}},
    },
    "dqn": {
        "name": "dqn",
        "params": {
            "hidden_sizes": [32],
            "buffer_size": 1000,
            "learning_starts": 200,
            "batch_size": 32,
            "target_update_interval": 100,
            "epsilon": {"decay_steps": 500},
        },
    },
    "reinforce": {"name": "reinforce", "params": {"hidden_sizes": [32], "baseline": "value"}},
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


@pytest.fixture(scope="module", params=sorted(SMALL_AGENTS))
def finished_run(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Any:
    agent = request.param
    run_dir = tmp_path_factory.mktemp(agent) / "run"
    run_experiment(make_config(agent), run_dir)
    return agent, run_dir


def test_training_logs_finite_diagnostics(finished_run: Any) -> None:
    agent, run_dir = finished_run
    _header, *rows = (run_dir / "train_episodes.csv").read_text().splitlines()
    assert rows, "no episode finished"
    metric_cells = [
        float(cell)
        for row in rows
        for cell in row.split(",")[6:]  # agent-specific columns follow the 6 standard ones
        if cell
    ]
    assert metric_cells, f"{agent} logged no diagnostics"
    assert all(abs(value) < 1e12 for value in metric_cells)


def test_same_seed_reproduces_the_run(finished_run: Any, tmp_path: Path) -> None:
    agent, run_dir = finished_run
    run_experiment(make_config(agent), tmp_path / "again")
    for name in ("train_episodes.csv", "eval.csv", "final_eval.json"):
        assert (tmp_path / "again" / name).read_bytes() == (run_dir / name).read_bytes(), name


def test_evaluation_schedule_does_not_change_training(finished_run: Any, tmp_path: Path) -> None:
    agent, run_dir = finished_run
    config = make_config(agent, eval={"interval_steps": 250, "episodes": 1, "final_episodes": 2})
    run_experiment(config, tmp_path / "more_evals")
    assert (tmp_path / "more_evals" / "train_episodes.csv").read_bytes() == (
        run_dir / "train_episodes.csv"
    ).read_bytes()


@pytest.mark.parametrize("checkpoint", ["final", "best"])
def test_checkpoint_reproduces_the_reported_evaluation(finished_run: Any, checkpoint: str) -> None:
    _, run_dir = finished_run
    reported = read_json(run_dir / "final_eval.json")[checkpoint]
    reloaded = evaluate_checkpoint(run_dir, checkpoint=checkpoint)
    assert reloaded["episodes"] == reported["episodes"]


@pytest.mark.parametrize("name", ["q_learning", "dqn", "reinforce", "random"])
def test_shipped_agent_configs_are_valid(name: str) -> None:
    config = build_experiment_config(
        {"agent": load_yaml(REPO_ROOT / "configs" / "agent" / f"{name}.yaml")}
    )
    assert config.agent.name == name
    env = gym.make(config.env.id)
    try:
        # Building the agent validates its parameters against the agent's own schema.
        agent = build_agent(config.agent, env.observation_space, env.action_space, seed=0)
        obs, _ = env.reset(seed=0)
        assert agent.act(obs) in range(4)
    finally:
        env.close()
