"""Does each agent learn? Checked on CartPole, which is solvable in minutes.

These are behavioural tests of the whole learning pipeline: a bug that leaves every unit
test green (a wrong sign, a missing optimiser step) still shows up here as no learning.
Slow, so excluded from CI; run with ``pytest -m slow``.
"""

import csv
import importlib.util
from pathlib import Path
from typing import Any

import pytest

from lunarlander_rl.config import build_experiment_config
from lunarlander_rl.training import run_experiment

# CartPole-v1 gives +1 per step for up to 500 steps; a random policy scores about 20.
CARTPOLE_BOUNDS = [[-2.4, 2.4], [-3.0, 3.0], [-0.21, 0.21], [-3.5, 3.5]]

CASES: dict[str, tuple[dict[str, Any], int, float]] = {
    # agent config, training steps, best periodic evaluation mean that must be reached
    "q_learning": (
        {
            "name": "q_learning",
            "params": {
                "alpha": 0.1,
                "num_tilings": 8,
                "tiles_per_dim": 8,
                "table_size": 65536,
                "bounds": CARTPOLE_BOUNDS,
                "discrete_dims": [],
                "epsilon": {"start": 0.5, "end": 0.0, "decay_steps": 20_000},
            },
        },
        50_000,
        150.0,
    ),
    "dqn": (
        {
            "name": "dqn",
            "params": {
                "hidden_sizes": [64, 64],
                "buffer_size": 20_000,
                "learning_starts": 1_000,
                "target_update_interval": 500,
                "epsilon": {"end": 0.05, "decay_steps": 10_000},
            },
        },
        40_000,
        195.0,
    ),
    "reinforce": (
        {"name": "reinforce", "params": {"hidden_sizes": [64], "lr": 3e-3, "baseline": "value"}},
        60_000,
        195.0,
    ),
    # SB3 settings from RL Baselines3 Zoo's CartPole-v1 entries.
    "sb3_dqn": (
        {
            "name": "sb3_dqn",
            "params": {
                "learning_rate": 2.3e-3,
                "batch_size": 64,
                "buffer_size": 100_000,
                "learning_starts": 1_000,
                "target_update_interval": 10,
                "train_freq": 256,
                "gradient_steps": 128,
                "exploration_decay_steps": 8_000,
                "exploration_final_eps": 0.04,
                "net_arch": [256, 256],
            },
        },
        50_000,
        195.0,
    ),
    "sb3_ppo": (
        {
            "name": "sb3_ppo",
            "params": {
                "n_envs": 8,
                "n_steps": 32,
                "batch_size": 256,
                "gae_lambda": 0.8,
                "gamma": 0.98,
                "n_epochs": 20,
                "learning_rate": 1e-3,
            },
        },
        60_000,
        195.0,
    ),
}

HAS_SB3 = importlib.util.find_spec("stable_baselines3") is not None


@pytest.mark.slow
@pytest.mark.parametrize("agent", sorted(CASES))
def test_agent_learns_cartpole(agent: str, tmp_path: Path) -> None:
    if agent.startswith("sb3_") and not HAS_SB3:
        pytest.skip("Stable-Baselines3 is not installed")
    agent_config, steps, threshold = CASES[agent]
    config = build_experiment_config(
        {
            "agent": agent_config,
            "env": {"id": "CartPole-v1"},
            "seed": 0,
            "train": {"total_steps": steps},
            "eval": {"interval_steps": 5_000, "episodes": 5, "final_episodes": 5},
        }
    )
    run_experiment(config, tmp_path / "run")
    with (tmp_path / "run" / "eval.csv").open(newline="") as handle:
        curve = [float(row["mean_return"]) for row in csv.DictReader(handle)]
    assert curve[0] < 100, "an untrained policy should not already balance the pole"
    assert max(curve) >= threshold, f"{agent} learning curve: {curve}"
