import numpy as np
import pytest

from lunarlander_rl.components.networks import mlp
from lunarlander_rl.components.replay_buffer import ReplayBuffer
from lunarlander_rl.components.returns import discounted_returns
from lunarlander_rl.components.schedules import EpsilonConfig, EpsilonSchedule
from lunarlander_rl.config import ConfigError


# ----------------------------------------------------------------------------- returns
def test_discounted_returns_by_hand() -> None:
    returns = discounted_returns([1.0, 0.0, 2.0], gamma=0.5)
    # G2 = 2, G1 = 0 + 0.5 * 2 = 1, G0 = 1 + 0.5 * 1 = 1.5
    np.testing.assert_allclose(returns, [1.5, 1.0, 2.0])


def test_discounted_returns_with_bootstrap() -> None:
    returns = discounted_returns([0.0, 0.0], gamma=0.9, bootstrap=10.0)
    np.testing.assert_allclose(returns, [8.1, 9.0])


def test_discounted_returns_edge_cases() -> None:
    assert discounted_returns([], gamma=0.9).shape == (0,)
    np.testing.assert_allclose(discounted_returns([1.0, 1.0, 1.0], gamma=1.0), [3.0, 2.0, 1.0])
    np.testing.assert_allclose(discounted_returns([1.0, 1.0], gamma=0.0), [1.0, 1.0])


# ----------------------------------------------------------------------- replay buffer
def test_replay_buffer_stores_transitions() -> None:
    buffer = ReplayBuffer(capacity=4, obs_shape=(2,))
    buffer.add([1.0, 2.0], 3, 0.5, [3.0, 4.0], True)
    assert len(buffer) == 1
    batch = buffer.sample(5, np.random.default_rng(0))
    assert batch.obs.shape == (5, 2) and batch.obs.dtype == np.float32
    np.testing.assert_array_equal(batch.obs, [[1.0, 2.0]] * 5)
    np.testing.assert_array_equal(batch.next_obs, [[3.0, 4.0]] * 5)
    np.testing.assert_array_equal(batch.actions, [3] * 5)
    np.testing.assert_array_equal(batch.rewards, [0.5] * 5)
    np.testing.assert_array_equal(batch.terminated, [1.0] * 5)


def test_replay_buffer_overwrites_oldest_when_full() -> None:
    buffer = ReplayBuffer(capacity=3, obs_shape=(1,))
    for i in range(5):
        buffer.add([i], i, float(i), [i + 1], False)
    assert len(buffer) == 3
    sampled = set(buffer.sample(200, np.random.default_rng(0)).actions.tolist())
    assert sampled == {2, 3, 4}


def test_replay_buffer_samples_only_filled_slots() -> None:
    buffer = ReplayBuffer(capacity=100, obs_shape=(1,))
    buffer.add([7.0], 1, 0.0, [8.0], False)
    buffer.add([9.0], 2, 0.0, [10.0], False)
    assert set(buffer.sample(100, np.random.default_rng(1)).actions.tolist()) == {1, 2}


def test_replay_buffer_validation() -> None:
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=0, obs_shape=(1,))
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=2, obs_shape=(1,)).sample(1, np.random.default_rng(0))


# --------------------------------------------------------------------------- schedules
def test_linear_epsilon_schedule() -> None:
    schedule = EpsilonSchedule(EpsilonConfig(start=1.0, end=0.1, decay_steps=10))
    values = []
    for _ in range(12):
        values.append(schedule.value)
        schedule.step()
    assert values[0] == 1.0
    assert values[5] == pytest.approx(0.55)
    assert values[10:] == pytest.approx([0.1, 0.1])
    schedule.end_episode()  # episode boundaries do not affect a per-step schedule
    assert schedule.value == pytest.approx(0.1)


def test_exponential_episode_schedule() -> None:
    config = EpsilonConfig(start=1.0, end=0.5, schedule="exponential_episodes", decay_rate=0.8)
    schedule = EpsilonSchedule(config)
    for _ in range(100):
        schedule.step()  # steps do not affect a per-episode schedule
    assert schedule.value == 1.0
    schedule.end_episode()
    assert schedule.value == pytest.approx(0.8)
    for _ in range(10):
        schedule.end_episode()
    assert schedule.value == 0.5


@pytest.mark.parametrize(
    "kwargs",
    [{"start": 0.1, "end": 0.5}, {"end": -0.1}, {"decay_steps": 0}, {"decay_rate": 1.5}],
)
def test_epsilon_config_validation(kwargs: dict[str, float]) -> None:
    with pytest.raises(ConfigError):
        EpsilonConfig(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------- networks
def test_mlp_shapes() -> None:
    import torch

    network = mlp(8, (32, 16), 4)
    assert network(torch.zeros(5, 8)).shape == (5, 4)
    linear = mlp(3, (), 2)
    assert len(linear) == 1
