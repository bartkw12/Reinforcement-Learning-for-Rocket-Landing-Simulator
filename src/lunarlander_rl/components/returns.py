"""Monte Carlo returns."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


def discounted_returns(
    rewards: Sequence[float], gamma: float, bootstrap: float = 0.0
) -> npt.NDArray[np.float64]:
    """Return ``G_t = r_t + gamma * G_{t+1}`` for every step of an episode.

    ``bootstrap`` is the value estimate of the state after the last reward. It is 0 when
    the episode terminated, and may be ``V(s_T)`` when a time limit truncated it.
    """
    returns = np.empty(len(rewards), dtype=np.float64)
    running = float(bootstrap)
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns
