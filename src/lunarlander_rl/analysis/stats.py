"""Statistics across independent training runs (seeds).

The aggregate metrics, probability of improvement and performance profiles follow
Agarwal et al., "Deep Reinforcement Learning at the Edge of the Statistical Precipice"
(NeurIPS 2021), specialised to a single environment: the unit of resampling is the training
run, and each method's runs are resampled separately (a stratified bootstrap).

Every generator is seeded, so reports are reproducible. With few runs (fewer than about 10)
percentile intervals are too narrow; they are reported as a guide to seed variation, not as
exact coverage.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

Array = npt.NDArray[np.float64]
Statistic = Callable[[Array], float]
# A statistic over the last axis of a 2-D array (one row per bootstrap sample).
RowStatistic = Callable[[Array], Array]

# LunarLander's conventional "solved" return, the target of the optimality gap.
SOLVED_RETURN = 200.0


def mean(values: Array) -> float:
    return float(np.mean(values))


def bootstrap_ci(
    values: npt.ArrayLike,
    statistic: Statistic = mean,
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of ``statistic`` over ``values``."""
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        raise ValueError("bootstrap_ci of no values")
    if data.size == 1:
        value = statistic(data)
        return value, value
    rng = np.random.default_rng(seed)
    samples = data[rng.integers(0, data.size, size=(resamples, data.size))]
    estimates = np.array([statistic(sample) for sample in samples])
    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.percentile(estimates, [tail, 100.0 - tail])
    return float(low), float(high)


# ------------------------------------------------------------------ aggregate metrics
def row_mean(samples: Array) -> Array:
    result: Array = samples.mean(axis=-1)
    return result


def row_median(samples: Array) -> Array:
    result: Array = np.median(samples, axis=-1)
    return result


def row_iqm(samples: Array) -> Array:
    """Interquartile mean along the last axis: the mean of the middle 50% of values, as
    ``scipy.stats.trim_mean(x, 0.25)`` computes it."""
    n = samples.shape[-1]
    cut = int(0.25 * n)
    ordered = np.sort(samples, axis=-1)
    result: Array = ordered[..., cut : n - cut].mean(axis=-1)
    return result


def row_optimality_gap(samples: Array, target: float = SOLVED_RETURN) -> Array:
    """How far runs fall short of ``target`` on average; exceeding it earns no credit, so
    the metric is not swayed by a few very high scores. Lower is better."""
    result: Array = np.maximum(target - samples, 0.0).mean(axis=-1)
    return result


AGGREGATES: dict[str, RowStatistic] = {
    "Median": row_median,
    "IQM": row_iqm,
    "Mean": row_mean,
    "Optimality gap": row_optimality_gap,
}


def interval_estimates(
    scores: Mapping[str, npt.ArrayLike],
    statistics: Mapping[str, RowStatistic] = AGGREGATES,
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Point estimate and percentile-bootstrap interval of each statistic for each method.

    ``scores`` maps a method to one score per run. Each method's runs are resampled
    independently; all statistics of a method share the same bootstrap samples.
    """
    rows: list[dict[str, Any]] = []
    tail = (1.0 - confidence) / 2.0 * 100.0
    for method, values in scores.items():
        data = np.asarray(values, dtype=np.float64)
        if data.size == 0:
            raise ValueError(f"no scores for {method!r}")
        # The same resamples as bootstrap_ci, so a metric reported in two tables has one
        # interval.
        rng = np.random.default_rng(seed)
        samples = data[rng.integers(0, data.size, size=(resamples, data.size))]
        for name, statistic in statistics.items():
            estimates = statistic(samples)
            low, high = np.percentile(estimates, [tail, 100.0 - tail])
            rows.append(
                {
                    "method": method,
                    "metric": name,
                    "value": float(statistic(data[np.newaxis, :])[0]),
                    "ci_low": float(low),
                    "ci_high": float(high),
                }
            )
    frame: pd.DataFrame = pd.DataFrame(rows)
    return frame


# ------------------------------------------------------- probability of improvement
def probability_of_improvement(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """P(X > Y): the chance that a random run of method X scores higher than a random run
    of method Y, counting ties as one half. This is the Mann-Whitney U statistic divided by
    the number of pairs."""
    a = np.asarray(x, dtype=np.float64)[:, np.newaxis]
    b = np.asarray(y, dtype=np.float64)[np.newaxis, :]
    return float(np.mean((a > b) + 0.5 * (a == b)))


def probability_of_improvement_ci(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    confidence: float = 0.95,
    resamples: int = 2_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """``(estimate, low, high)``, resampling the runs of X and of Y independently."""
    a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    a_samples = a[rng.integers(0, a.size, size=(resamples, a.size))]
    b_samples = b[rng.integers(0, b.size, size=(resamples, b.size))]
    greater = a_samples[:, :, np.newaxis] > b_samples[:, np.newaxis, :]
    equal = a_samples[:, :, np.newaxis] == b_samples[:, np.newaxis, :]
    estimates = (greater + 0.5 * equal).mean(axis=(1, 2))
    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.percentile(estimates, [tail, 100.0 - tail])
    return probability_of_improvement(a, b), float(low), float(high)


# ------------------------------------------------------------- performance profiles
def performance_profile(
    episode_returns: Sequence[npt.ArrayLike],
    thresholds: npt.ArrayLike,
    *,
    confidence: float = 0.95,
    resamples: int = 2_000,
    seed: int = 0,
) -> tuple[Array, Array, Array]:
    """For each threshold tau, the fraction of test episodes scoring above tau.

    ``episode_returns`` holds one sequence of test-episode returns per run. Each run's
    fraction is computed first and the fractions are averaged over runs, so every run
    counts equally; the interval resamples runs. Returns ``(profile, low, high)``.
    """
    taus = np.asarray(thresholds, dtype=np.float64)
    fractions = np.stack(
        [
            (np.asarray(r, dtype=np.float64)[:, np.newaxis] > taus).mean(axis=0)
            for r in episode_returns
        ]
    )
    rng = np.random.default_rng(seed)
    runs = fractions.shape[0]
    samples = fractions[rng.integers(0, runs, size=(resamples, runs))].mean(axis=1)
    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.percentile(samples, [tail, 100.0 - tail], axis=0)
    profile: Array = fractions.mean(axis=0)
    return profile, low, high


# ------------------------------------------------------------------- learning curves
def curve_estimates(
    curves: Array,
    statistic: RowStatistic = row_iqm,
    *,
    confidence: float = 0.95,
    resamples: int = 2_000,
    seed: int = 0,
) -> tuple[Array, Array, Array]:
    """``statistic`` over runs at every point of a learning curve, with bootstrap intervals.

    ``curves`` has one row per run and one column per evaluation point; runs are resampled
    as whole curves, which keeps each run's points together. Returns ``(estimate, low,
    high)``, each with one value per column.
    """
    runs = curves.shape[0]
    rng = np.random.default_rng(seed)
    # (resamples, runs, points) -> statistic over the runs axis.
    samples = curves[rng.integers(0, runs, size=(resamples, runs))]
    estimates = statistic(np.swapaxes(samples, 1, 2))
    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.percentile(estimates, [tail, 100.0 - tail], axis=0)
    estimate: Array = statistic(curves.T)
    return estimate, low, high
