"""Sanity checks on the installed environment that every experiment depends on."""

import gymnasium as gym
import numpy as np

ENV_ID = "LunarLander-v3"


def test_lunar_lander_spaces() -> None:
    env = gym.make(ENV_ID)
    try:
        assert env.observation_space.shape == (8,)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 4
    finally:
        env.close()


def test_lunar_lander_is_deterministic_given_seed() -> None:
    def rollout(seed: int, num_steps: int = 50) -> tuple[np.ndarray, list[float]]:
        env = gym.make(ENV_ID)
        try:
            obs, _ = env.reset(seed=seed)
            env.action_space.seed(seed)
            observations, rewards = [obs], []
            for _ in range(num_steps):
                obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
                observations.append(obs)
                rewards.append(float(reward))
                if terminated or truncated:
                    break
            return np.stack(observations), rewards
        finally:
            env.close()

    obs_a, rewards_a = rollout(seed=0)
    obs_b, rewards_b = rollout(seed=0)
    obs_c, _ = rollout(seed=1)

    np.testing.assert_array_equal(obs_a, obs_b)
    assert rewards_a == rewards_b
    assert not np.array_equal(obs_a[0], obs_c[0])
