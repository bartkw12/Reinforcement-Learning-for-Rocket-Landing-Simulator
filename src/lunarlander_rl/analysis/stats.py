"""Statistics across independent training runs (seeds)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

Statistic = Callable[[npt.NDArray[np.float64]], float]


def mean(values: npt.NDArray[np.float64]) -> float:
    return float(np.mean(values))


def bootstrap_ci(
    values: Sequence[float],
    statistic: Statistic = mean,
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of ``statistic`` over ``values``.

    The unit of resampling is the training run, so the interval reflects how much the
    result would change with different seeds. The generator is seeded, so reports are
    reproducible. With few runs (fewer than about 10) percentile intervals are too narrow;
    they are reported as a guide to seed variation, not as exact coverage.
    """
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
