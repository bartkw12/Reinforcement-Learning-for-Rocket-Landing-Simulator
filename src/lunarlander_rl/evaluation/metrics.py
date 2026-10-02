"""Per-episode results and the statistics computed from them."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

# How an evaluation episode ended.
LANDED = "landed"
CRASHED = "crashed"
OUT_OF_BOUNDS = "out_of_bounds"
TIMEOUT = "timeout"
TERMINATED = "terminated"


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    episode_return: float
    length: int
    terminated: bool
    truncated: bool
    outcome: str
    success: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_outcome(
    env_id: str | None,
    final_obs: Any,
    last_reward: float,
    terminated: bool,
    truncated: bool,
) -> str:
    """Name how an episode ended.

    LunarLander terminates in three ways, which its last step distinguishes: the lander
    comes to rest (final reward of exactly +100), leaves the viewport (``|x| >= 1``), or
    its body touches the ground (both of the latter give exactly -100).
    """
    if terminated:
        if env_id is not None and env_id.startswith("LunarLander"):
            if last_reward == 100.0:
                return LANDED
            if abs(float(final_obs[0])) >= 1.0:
                return OUT_OF_BOUNDS
            return CRASHED
        return TERMINATED
    if truncated:
        return TIMEOUT
    raise ValueError("episode has not ended")


def interquartile_mean(values: Sequence[float]) -> float:
    """Mean of the middle 50% of ``values``.

    More robust to outlier episodes than the mean and more statistically efficient than
    the median (Agarwal et al., 2021). Matches ``scipy.stats.trim_mean(values, 0.25)``.
    """
    if len(values) == 0:
        raise ValueError("interquartile_mean of an empty sequence")
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    cut = int(0.25 * len(ordered))
    return float(ordered[cut : len(ordered) - cut].mean())


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default).

    Unlike the normal approximation it stays inside [0, 1] and behaves sensibly when the
    observed rate is 0% or 100%, which is common for success rates.
    """
    if total <= 0:
        raise ValueError("total must be positive")
    if not 0 <= successes <= total:
        raise ValueError("successes must be between 0 and total")
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    # The bounds at 0 and 100% are exactly 0 and 1; avoid reporting rounding residue.
    low = 0.0 if successes == 0 else max(0.0, centre - half_width)
    high = 1.0 if successes == total else min(1.0, centre + half_width)
    return low, high


def summarize(results: Sequence[EpisodeResult]) -> dict[str, Any]:
    """Aggregate statistics over a set of evaluation episodes."""
    if len(results) == 0:
        raise ValueError("cannot summarise zero episodes")
    returns = np.asarray([r.episode_return for r in results], dtype=np.float64)
    successes = sum(r.success for r in results)
    low, high = wilson_interval(successes, len(results))
    return {
        "episodes": len(results),
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std(ddof=1)) if len(results) > 1 else 0.0,
        "iqm_return": interquartile_mean(returns.tolist()),
        "median_return": float(np.median(returns)),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
        "success_rate": successes / len(results),
        "success_rate_ci_low": low,
        "success_rate_ci_high": high,
        "mean_length": float(np.mean([r.length for r in results])),
        "outcomes": dict(sorted(Counter(r.outcome for r in results).items())),
    }
