import numpy as np
import pytest

from lunarlander_rl.analysis import stats
from lunarlander_rl.evaluation.metrics import interquartile_mean

RNG = np.random.default_rng(0)
SCORES = RNG.normal(150.0, 60.0, size=10)


def test_row_statistics_match_their_definitions() -> None:
    rows = RNG.normal(size=(4, 9))
    np.testing.assert_allclose(stats.row_mean(rows), rows.mean(axis=1))
    np.testing.assert_allclose(stats.row_median(rows), np.median(rows, axis=1))
    for row, value in zip(rows, stats.row_iqm(rows), strict=True):
        assert value == pytest.approx(interquartile_mean(row.tolist()))


def test_iqm_matches_scipy_trim_mean() -> None:
    scipy_stats = pytest.importorskip("scipy.stats")
    for n in (4, 5, 10, 13):
        values = RNG.normal(size=n)
        assert stats.row_iqm(values[np.newaxis])[0] == pytest.approx(
            scipy_stats.trim_mean(values, 0.25)
        )


def test_optimality_gap() -> None:
    gap = stats.row_optimality_gap(np.array([[250.0, 150.0, 100.0, 200.0]]))
    # Shortfalls below 200 are 0, 50, 100, 0; scores above 200 earn nothing extra.
    assert gap[0] == pytest.approx(37.5)


def test_interval_estimates() -> None:
    frame = stats.interval_estimates({"a": SCORES, "b": SCORES + 50.0}, resamples=2_000)
    assert sorted(frame["metric"].unique()) == ["IQM", "Mean", "Median", "Optimality gap"]
    assert len(frame) == 8
    assert (frame["ci_low"] <= frame["value"]).all() and (frame["value"] <= frame["ci_high"]).all()
    mean_a = frame[(frame["method"] == "a") & (frame["metric"] == "Mean")].iloc[0]
    assert mean_a["value"] == pytest.approx(SCORES.mean())
    # Same resamples as bootstrap_ci, so the two never report different intervals.
    low, high = stats.bootstrap_ci(SCORES, resamples=2_000)
    assert (mean_a["ci_low"], mean_a["ci_high"]) == pytest.approx((low, high))
    # Shifting every score up shifts the location metrics and lowers the gap.
    by = frame.set_index(["method", "metric"])["value"]
    assert by[("b", "IQM")] == pytest.approx(by[("a", "IQM")] + 50.0)
    assert by[("b", "Optimality gap")] < by[("a", "Optimality gap")]
    # Reproducible.
    again = stats.interval_estimates({"a": SCORES, "b": SCORES + 50.0}, resamples=2_000)
    assert frame.equals(again)


def test_probability_of_improvement() -> None:
    assert stats.probability_of_improvement([3.0, 4.0], [1.0, 2.0]) == 1.0
    assert stats.probability_of_improvement([1.0, 2.0], [3.0, 4.0]) == 0.0
    assert stats.probability_of_improvement([1.0, 2.0], [1.0, 2.0]) == 0.5
    assert stats.probability_of_improvement([2.0], [1.0, 2.0, 3.0]) == pytest.approx(0.5)


def test_probability_of_improvement_matches_mann_whitney() -> None:
    scipy_stats = pytest.importorskip("scipy.stats")
    x, y = RNG.normal(1.0, size=10), RNG.normal(size=8)
    u = scipy_stats.mannwhitneyu(x, y).statistic
    assert stats.probability_of_improvement(x, y) == pytest.approx(u / (len(x) * len(y)))


def test_probability_of_improvement_interval() -> None:
    estimate, low, high = stats.probability_of_improvement_ci(SCORES + 30.0, SCORES)
    assert low <= estimate <= high
    assert 0.5 < estimate < 1.0


def test_performance_profile() -> None:
    runs = [[0.0, 100.0, 200.0, 300.0], [150.0, 250.0, 250.0, 250.0]]
    profile, low, high = stats.performance_profile(runs, [-1.0, 125.0, 400.0], resamples=500)
    # Run fractions above 125 are 0.5 and 1.0, so the profile there is their mean.
    np.testing.assert_allclose(profile, [1.0, 0.75, 0.0])
    assert (low <= profile).all() and (profile <= high).all()


def test_curve_estimates() -> None:
    curves = np.vstack([np.linspace(0, 100, 5) + offset for offset in range(10)])
    estimate, low, high = stats.curve_estimates(curves, resamples=500)
    np.testing.assert_allclose(estimate, stats.row_iqm(curves.T))
    assert estimate.shape == low.shape == high.shape == (5,)
    assert (low <= estimate).all() and (estimate <= high).all()
    mean_estimate, _, _ = stats.curve_estimates(curves, stats.row_mean, resamples=100)
    np.testing.assert_allclose(mean_estimate, curves.mean(axis=0))
