"""Regression tests for the update rules, including every learning bug fixed from v1.0."""

from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch

from lunarlander_rl.agents import Transition, build_agent
from lunarlander_rl.agents.dqn import DQNAgent, td_targets
from lunarlander_rl.agents.q_learning import QLearningAgent
from lunarlander_rl.agents.reinforce import ReinforceAgent, advantages
from lunarlander_rl.config import AgentConfig

ENV = gym.make("LunarLander-v3")
LEFT = np.array([-0.9, 1.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
RIGHT = np.array([0.9, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
GAMMA = 0.99


def make(name: str, **params: Any) -> Any:
    return build_agent(AgentConfig(name, params), ENV.observation_space, ENV.action_space, seed=0)


def transition(*, terminated: bool = False, truncated: bool = False, **kw: Any) -> Transition:
    values: dict[str, Any] = {
        "obs": LEFT,
        "action": 0,
        "reward": 1.0,
        "next_obs": RIGHT,
        "terminated": terminated,
        "truncated": truncated,
    }
    values.update(kw)
    return Transition(**values)


# --------------------------------------------------------------------------- Q-learning
def q_learning_with_known_next_value() -> QLearningAgent:
    agent: QLearningAgent = make("q_learning", alpha=0.5, gamma=GAMMA, num_tilings=8)
    # Make Q(RIGHT, 1) = 10 so that bootstrapping is visible; LEFT and RIGHT share no tiles.
    agent.weights[1, agent.coder(RIGHT)] = 10.0 / 8
    return agent


def test_q_learning_does_not_bootstrap_from_terminal_states() -> None:
    agent = q_learning_with_known_next_value()
    agent.observe(transition(terminated=True))
    # Q(LEFT, 0) moves alpha of the way from 0 to the target r = 1.
    assert agent.q_values(agent.coder(LEFT))[0] == pytest.approx(0.5 * 1.0)


def test_q_learning_bootstraps_through_truncation() -> None:
    agent = q_learning_with_known_next_value()
    agent.observe(transition(truncated=True))
    assert agent.q_values(agent.coder(LEFT))[0] == pytest.approx(0.5 * (1.0 + GAMMA * 10.0))


def test_q_learning_explores_during_training() -> None:
    # v1.0 always acted greedily in training. With epsilon = 1 every action must occur.
    agent: QLearningAgent = make("q_learning", epsilon={"start": 1.0, "end": 1.0})
    assert {agent.act(LEFT) for _ in range(200)} == {0, 1, 2, 3}


def test_q_learning_greedy_ties_are_broken_at_random() -> None:
    agent: QLearningAgent = make("q_learning", epsilon={"start": 0.0, "end": 0.0})
    assert len({agent.act(LEFT) for _ in range(200)}) == 4
    assert agent.predict(LEFT) == 0  # evaluation stays deterministic


def test_q_learning_epsilon_follows_its_schedule() -> None:
    agent: QLearningAgent = make("q_learning", epsilon={"start": 1.0, "end": 0.0, "decay_steps": 4})
    seen = [agent.observe(transition())["epsilon"] for _ in range(5)]
    assert seen == pytest.approx([1.0, 0.75, 0.5, 0.25, 0.0])


def test_q_learning_can_reproduce_v1_bugs() -> None:
    agent: QLearningAgent = make(
        "q_learning", alpha=0.5, gamma=GAMMA, num_tilings=8, reproduce_v1_bugs=True
    )
    agent.weights[1, agent.coder(RIGHT)] = 10.0 / 8
    # Bootstraps from the terminal state, as v1.0 did...
    agent.observe(transition(terminated=True))
    assert agent.q_values(agent.coder(LEFT))[0] == pytest.approx(0.5 * (1.0 + GAMMA * 10.0))
    # ...and trains greedily, ties going to action 0, even though epsilon is 1.
    assert agent.epsilon.value > 0.99
    assert {agent.act(RIGHT) for _ in range(50)} == {1}
    unvisited = np.array([2.4, -2.4, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    assert {agent.act(unvisited) for _ in range(50)} == {0}


def test_q_learning_evaluation_does_not_touch_the_tile_table() -> None:
    agent: QLearningAgent = make("q_learning")
    agent.act(LEFT)
    before = dict(agent.coder.table.dictionary)
    agent.predict(RIGHT)
    assert agent.coder.table.dictionary == before


# --------------------------------------------------------------------------------- DQN
def test_td_targets_stop_at_termination() -> None:
    targets = td_targets(
        rewards=torch.tensor([1.0, 1.0]),
        terminated=torch.tensor([1.0, 0.0]),
        next_q_target=torch.tensor([[5.0, 10.0], [5.0, 10.0]]),
        gamma=0.5,
    )
    torch.testing.assert_close(targets, torch.tensor([1.0, 6.0]))


def test_double_dqn_evaluates_the_online_networks_choice() -> None:
    next_q_target = torch.tensor([[5.0, 10.0]])
    next_q_online = torch.tensor([[3.0, 1.0]])  # online network prefers action 0
    rewards, terminated = torch.tensor([0.0]), torch.tensor([0.0])
    torch.testing.assert_close(
        td_targets(rewards, terminated, next_q_target, gamma=1.0), torch.tensor([10.0])
    )
    torch.testing.assert_close(
        td_targets(rewards, terminated, next_q_target, gamma=1.0, next_q_online=next_q_online),
        torch.tensor([5.0]),
    )


def test_dqn_replay_stores_termination_not_truncation() -> None:
    agent: DQNAgent = make("dqn", learning_starts=10_000)
    agent.observe(transition(truncated=True))
    agent.observe(transition(terminated=True))
    np.testing.assert_array_equal(agent.buffer.terminated[:2], [0.0, 1.0])


def test_dqn_can_reproduce_v1_bugs() -> None:
    agent: DQNAgent = make("dqn", learning_starts=10_000, reproduce_v1_bugs=True)
    agent.observe(transition(truncated=True))
    agent.observe(transition())
    np.testing.assert_array_equal(agent.buffer.terminated[:2], [1.0, 0.0])


def test_dqn_waits_for_learning_starts_then_updates() -> None:
    agent: DQNAgent = make("dqn", learning_starts=5, batch_size=2, hidden_sizes=[8])
    metrics = [agent.observe(transition()) for _ in range(6)]
    assert all("loss" not in m for m in metrics[:4])
    assert {"loss", "q_mean", "epsilon"} <= set(metrics[5])
    assert np.isfinite(metrics[5]["loss"])


@pytest.mark.parametrize(("unit", "interval", "syncs_after"), [("steps", 3, 3), ("episodes", 2, 2)])
def test_dqn_target_network_sync(unit: str, interval: int, syncs_after: int) -> None:
    agent: DQNAgent = make(
        "dqn",
        learning_starts=1,
        batch_size=1,
        hidden_sizes=[8],
        target_update_unit=unit,
        target_update_interval=interval,
    )

    def in_sync() -> bool:
        online, target = agent.q_network.state_dict(), agent.target_network.state_dict()
        return all(torch.equal(online[k], target[k]) for k in online)

    history = []
    for _ in range(syncs_after):
        # Every transition ends an episode, so steps and episodes advance together.
        agent.observe(transition(terminated=True))
        history.append(in_sync())
    assert history == [False] * (syncs_after - 1) + [True]


# --------------------------------------------------------------------------- REINFORCE
def test_advantages_for_each_baseline() -> None:
    returns = torch.tensor([3.0, 1.0, 2.0])
    torch.testing.assert_close(advantages(returns, "none"), returns)
    normalised = advantages(returns, "normalize")
    assert normalised.mean().item() == pytest.approx(0.0, abs=1e-6)
    assert normalised.std().item() == pytest.approx(1.0, abs=1e-6)
    torch.testing.assert_close(
        advantages(returns, "value", torch.tensor([1.0, 1.0, 1.0])), torch.tensor([2.0, 0.0, 1.0])
    )
    torch.testing.assert_close(advantages(torch.tensor([4.0]), "normalize"), torch.tensor([0.0]))
    with pytest.raises(ValueError):
        advantages(returns, "value")


def test_reinforce_respects_constructor_arguments() -> None:
    # v1.0 ignored its lr and hidden_dim arguments.
    agent: ReinforceAgent = make("reinforce", lr=0.123, hidden_sizes=[7])
    assert agent.optimizer.param_groups[0]["lr"] == 0.123
    assert agent.policy[0].out_features == 7


def constant_value_agent(value: float) -> ReinforceAgent:
    agent: ReinforceAgent = make("reinforce", baseline="value", gamma=GAMMA)
    assert agent.value is not None
    last = agent.value[-1]
    assert isinstance(last, torch.nn.Linear)
    with torch.no_grad():
        last.weight.zero_()
        last.bias.fill_(value)
    # Freeze it so the value loss below is computed against V = 5 exactly.
    agent.value_optimizer = torch.optim.SGD(agent.value.parameters(), lr=0.0)
    return agent


def test_reinforce_bootstraps_from_value_when_truncated() -> None:
    agent = constant_value_agent(5.0)
    agent.observe(transition(reward=0.0))
    metrics = agent.observe(transition(reward=0.0, truncated=True))
    returns = np.array([GAMMA**2 * 5.0, GAMMA * 5.0])
    assert metrics["value_loss"] == pytest.approx(np.mean((5.0 - returns) ** 2), rel=1e-5)


def test_reinforce_does_not_bootstrap_when_terminated() -> None:
    agent = constant_value_agent(5.0)
    agent.observe(transition(reward=0.0))
    metrics = agent.observe(transition(reward=0.0, terminated=True))
    assert metrics["value_loss"] == pytest.approx(25.0, rel=1e-5)


def test_reinforce_updates_once_per_episode() -> None:
    agent: ReinforceAgent = make("reinforce")
    before = [p.clone() for p in agent.policy.parameters()]
    assert agent.observe(transition(reward=1.0)) == {}
    assert all(torch.equal(a, b) for a, b in zip(before, agent.policy.parameters(), strict=True))
    metrics = agent.observe(transition(reward=-1.0, terminated=True))
    assert set(metrics) == {"policy_loss", "entropy"}
    assert not all(
        torch.equal(a, b) for a, b in zip(before, agent.policy.parameters(), strict=True)
    )


def test_reinforce_stochastic_evaluation_is_reproducible_per_seed() -> None:
    agent: ReinforceAgent = make("reinforce")

    def sample(seed: int) -> list[int]:
        agent.seed_eval(seed)
        return [agent.predict(LEFT, deterministic=False) for _ in range(30)]

    assert sample(3) == sample(3)
    assert sample(3) != sample(4)
