import numpy as np
import pytest

from lunarlander_rl.evaluation.metrics import (
    CRASHED,
    LANDED,
    OUT_OF_BOUNDS,
    TERMINATED,
    TIMEOUT,
    EpisodeResult,
    classify_outcome,
    interquartile_mean,
    summarize,
    wilson_interval,
)

ENV_ID = "LunarLander-v3"
CENTRED = np.zeros(8)
OFF_SCREEN = np.array([1.02, 0.5, 0, 0, 0, 0, 0, 0])


def episode(episode_return: float, *, outcome: str = LANDED, length: int = 100) -> EpisodeResult:
    return EpisodeResult(
        seed=0,
        episode_return=episode_return,
        length=length,
        terminated=outcome != TIMEOUT,
        truncated=outcome == TIMEOUT,
        outcome=outcome,
        success=outcome != TIMEOUT and episode_return >= 200,
    )


def test_classify_outcome_for_lunar_lander() -> None:
    assert classify_outcome(ENV_ID, CENTRED, 100.0, True, False) == LANDED
    assert classify_outcome(ENV_ID, CENTRED, -100.0, True, False) == CRASHED
    assert classify_outcome(ENV_ID, OFF_SCREEN, -100.0, True, False) == OUT_OF_BOUNDS
    assert classify_outcome(ENV_ID, CENTRED, 0.3, False, True) == TIMEOUT


def test_classify_outcome_for_other_environments() -> None:
    assert classify_outcome("CartPole-v1", np.zeros(4), 1.0, True, False) == TERMINATED
    assert classify_outcome(None, np.zeros(4), 1.0, False, True) == TIMEOUT


def test_classify_outcome_requires_a_finished_episode() -> None:
    with pytest.raises(ValueError):
        classify_outcome(ENV_ID, CENTRED, 0.0, False, False)


def test_interquartile_mean_ignores_the_tails() -> None:
    # Eight values: the lowest two and highest two are dropped.
    assert interquartile_mean([-1000, 0, 1, 2, 3, 4, 5, 1000]) == pytest.approx(2.5)
    assert interquartile_mean([7.0]) == 7.0
    assert interquartile_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        interquartile_mean([])


def test_wilson_interval_matches_reference_values() -> None:
    low, high = wilson_interval(67, 100)
    assert (low, high) == pytest.approx((0.5730, 0.7545), abs=5e-4)
    # At the extremes the interval stays inside [0, 1] and is not degenerate.
    low, high = wilson_interval(0, 100)
    assert low == 0.0 and 0.03 < high < 0.04
    low, high = wilson_interval(100, 100)
    assert high == 1.0 and 0.96 < low < 0.97


def test_wilson_interval_validates_input() -> None:
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
    with pytest.raises(ValueError):
        wilson_interval(5, 4)


def test_summarize() -> None:
    results = [
        episode(250.0),
        episode(210.0),
        episode(-120.0, outcome=CRASHED),
        episode(60.0, outcome=TIMEOUT, length=1000),
    ]
    summary = summarize(results)
    assert summary["episodes"] == 4
    assert summary["mean_return"] == pytest.approx(100.0)
    assert summary["std_return"] == pytest.approx(np.std([250, 210, -120, 60], ddof=1))
    assert summary["iqm_return"] == pytest.approx(135.0)
    assert summary["median_return"] == pytest.approx(135.0)
    assert (summary["min_return"], summary["max_return"]) == (-120.0, 250.0)
    assert summary["success_rate"] == 0.5
    assert summary["success_rate_ci_low"] < 0.5 < summary["success_rate_ci_high"]
    assert summary["mean_length"] == pytest.approx(325.0)
    assert summary["outcomes"] == {CRASHED: 1, LANDED: 2, TIMEOUT: 1}


def test_summarize_single_episode_has_zero_spread() -> None:
    assert summarize([episode(10.0)])["std_return"] == 0.0
    with pytest.raises(ValueError):
        summarize([])
