"""The contract between the training loop and an agent, checked with a scripted agent."""

import csv
import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import pytest

from lunarlander_rl.agents import Agent, Observation, Transition, available_agents, build_agent
from lunarlander_rl.agents.registry import register
from lunarlander_rl.config import AgentConfig, ConfigError, build_experiment_config
from lunarlander_rl.training import run_experiment


@dataclass(frozen=True)
class ScriptedConfig:
    action: int = 0
    report_undeclared_metric: bool = False


@register("scripted", ScriptedConfig)
class ScriptedAgent(Agent):
    """Always takes the same action and records what the training loop shows it."""

    checkpoint_suffix: ClassVar[str] = ".txt"
    metric_names: ClassVar[tuple[str, ...]] = ("steps_seen", "episode_end")
    instances: ClassVar[list["ScriptedAgent"]] = []

    def __init__(
        self,
        config: ScriptedConfig,
        observation_space: gym.Space[Any],
        action_space: gym.spaces.Discrete,
        *,
        seed: int,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.transitions: list[Transition] = []
        ScriptedAgent.instances.append(self)

    def act(self, obs: Observation) -> int:
        return self.config.action

    def predict(self, obs: Observation, *, deterministic: bool = True) -> int:
        return self.config.action

    def observe(self, transition: Transition) -> Mapping[str, float]:
        self.transitions.append(transition)
        if self.config.report_undeclared_metric:
            return {"surprise": 1.0}
        metrics = {"steps_seen": float(len(self.transitions))}
        if transition.done:
            metrics["episode_end"] = 1.0
        return metrics

    def save(self, path: Path) -> None:
        path.write_text(str(self.config.action))

    def load(self, path: Path) -> None:
        assert int(path.read_text()) == self.config.action


def run(tmp_path: Path, **params: Any) -> ScriptedAgent:
    ScriptedAgent.instances.clear()
    config = build_experiment_config(
        {
            "agent": {"name": "scripted", "params": params},
            # A 20-step time limit forces truncations within a short run.
            "env": {"id": "LunarLander-v3", "max_episode_steps": 20},
            "train": {"total_steps": 100},
            "eval": {"interval_steps": 100, "episodes": 1, "final_episodes": 1},
        }
    )
    run_experiment(config, tmp_path / "run")
    return ScriptedAgent.instances[0]


def test_agent_sees_every_step_as_a_consistent_transition(tmp_path: Path) -> None:
    agent = run(tmp_path, action=2)
    transitions = agent.transitions
    assert len(transitions) == 100
    assert all(t.action == 2 for t in transitions)
    for previous, current in itertools.pairwise(transitions):
        if not previous.done:
            # Within an episode each transition starts where the last one ended.
            assert (current.obs == previous.next_obs).all()


def test_time_limit_is_reported_as_truncation_not_termination(tmp_path: Path) -> None:
    agent = run(tmp_path, action=0)
    endings = [t for t in agent.transitions if t.done]
    assert len(endings) == 5
    assert all(t.truncated and not t.terminated for t in endings)


def test_agent_metrics_are_averaged_per_episode(tmp_path: Path) -> None:
    run(tmp_path)
    with (tmp_path / "run" / "train_episodes.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    # Steps 1..20 were seen during the first episode; their mean is 10.5.
    assert float(rows[0]["steps_seen"]) == pytest.approx(10.5)
    assert float(rows[1]["steps_seen"]) == pytest.approx(30.5)
    # Reported once per episode, so the mean over its reports is 1.
    assert all(float(row["episode_end"]) == 1.0 for row in rows)


def test_undeclared_metric_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="surprise"):
        run(tmp_path, report_undeclared_metric=True)


def test_registry_validates_agent_name_parameters_and_action_space() -> None:
    env = gym.make("LunarLander-v3")
    continuous = gym.make("LunarLander-v3", continuous=True)
    try:
        assert {"random", "scripted"} <= set(available_agents())
        with pytest.raises(ConfigError, match="unknown agent"):
            build_agent(AgentConfig("dqn_typo"), env.observation_space, env.action_space, seed=0)
        with pytest.raises(ConfigError, match=r"agent\.params.*unknown key"):
            build_agent(
                AgentConfig("scripted", {"acton": 1}),
                env.observation_space,
                env.action_space,
                seed=0,
            )
        with pytest.raises(ConfigError, match="discrete action space"):
            build_agent(
                AgentConfig("random"),
                continuous.observation_space,
                continuous.action_space,
                seed=0,
            )
    finally:
        env.close()
        continuous.close()


def test_registering_a_name_twice_is_an_error() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register("random", ScriptedConfig)(ScriptedAgent)
