"""Tile coding for linear value-function approximation.

Adapted from Richard S. Sutton's ``tiles3`` (http://incompleteideas.net/tiles/tiles3.html),
which is free for any use. Tile coding overlays ``num_tilings`` grids over the state space,
each shifted by a different offset; a state activates exactly one tile per grid, and its
value is the sum of the weights of the active tiles. Nearby states share tiles, so what is
learned about one state generalises to its neighbours.

Two choices from ``tiles3`` are kept deliberately:

* Asymmetric offsets: in tiling ``t`` dimension ``i`` is shifted by ``t * (2i + 1)``
  fractions of a tile, rather than by ``t`` in every dimension. Uniform offsets shift all
  tilings along the diagonal, which generalises poorly (Sutton & Barto, 2018, sec. 9.5.4).
* Hashing into a fixed-size table: the number of tile coordinates that can occur is
  astronomically large, but only those actually visited get an index. Once the table is
  full, new coordinates share indices (collisions), which this module counts.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

Coordinates = tuple[int, ...]


class IndexHashTable:
    """Assigns consecutive indices to tile coordinates until full, then hashes."""

    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self.size = size
        self.dictionary: dict[Coordinates, int] = {}
        self.collisions = 0

    def index(self, coordinates: Coordinates, *, readonly: bool = False) -> int:
        """Index of ``coordinates``. In ``readonly`` mode the table is never modified and
        coordinates without an index give -1 while the table still has room."""
        found = self.dictionary.get(coordinates)
        if found is not None:
            return found
        if len(self.dictionary) >= self.size:
            if readonly:
                return hash(coordinates) % self.size
            self.collisions += 1
            # Hashing a tuple of ints is deterministic in CPython (no hash randomisation).
            return hash(coordinates) % self.size
        if readonly:
            return -1
        new = len(self.dictionary)
        self.dictionary[coordinates] = new
        return new

    @property
    def usage(self) -> float:
        """Fraction of the table assigned to distinct tiles."""
        return len(self.dictionary) / self.size

    def state_dict(self) -> dict[str, Any]:
        keys = list(self.dictionary)
        return {
            "keys": np.asarray(keys, dtype=np.int64),
            "values": np.asarray([self.dictionary[k] for k in keys], dtype=np.int64),
            "collisions": self.collisions,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        keys, values = np.asarray(state["keys"]), np.asarray(state["values"])
        self.dictionary = {
            tuple(int(c) for c in key): int(value) for key, value in zip(keys, values, strict=True)
        }
        self.collisions = int(state["collisions"])


class TileCoder:
    """Maps an observation to the indices of its active tiles, one per tiling.

    Continuous dimensions are rescaled from ``bounds`` to ``tiles_per_dim`` tile widths
    (values outside the bounds fall in the edge tiles). Dimensions listed in
    ``discrete_dims`` (e.g. LunarLander's leg-contact flags) are not tiled: they select a
    separate set of tiles, so each discrete combination learns its own values.
    """

    def __init__(
        self,
        bounds: Sequence[tuple[float, float]],
        *,
        num_tilings: int,
        tiles_per_dim: int,
        table_size: int,
        continuous_dims: Sequence[int],
        discrete_dims: Sequence[int] = (),
    ) -> None:
        if len(bounds) != len(continuous_dims):
            raise ValueError("need one (low, high) pair per continuous dimension")
        low, high = np.asarray(bounds, dtype=np.float64).T
        if np.any(high <= low):
            raise ValueError("every bound needs low < high")
        if num_tilings <= 0 or tiles_per_dim <= 0:
            raise ValueError("num_tilings and tiles_per_dim must be positive")
        self.num_tilings = num_tilings
        self.tiles_per_dim = tiles_per_dim
        self.continuous_dims = np.asarray(continuous_dims, dtype=np.int64)
        self.discrete_dims = np.asarray(discrete_dims, dtype=np.int64)
        self._low = low
        self._scale = tiles_per_dim / (high - low)
        self.table = IndexHashTable(table_size)
        tiling = np.arange(num_tilings)[:, None]
        self._offsets = tiling * (2 * np.arange(len(continuous_dims)) + 1)[None, :]

    @property
    def table_size(self) -> int:
        return self.table.size

    def __call__(self, obs: Any, *, readonly: bool = False) -> npt.NDArray[np.int64]:
        """Active tile indices of ``obs``. With ``readonly``, unseen tiles give -1 instead
        of being added to the table, so evaluation leaves the coder unchanged."""
        obs = np.asarray(obs, dtype=np.float64)
        scaled = (obs[self.continuous_dims] - self._low) * self._scale
        # Values just outside the bounds land in the edge tiles rather than new ones.
        scaled = np.clip(scaled, 0.0, self.tiles_per_dim - 1e-9)
        quantised = np.floor(scaled * self.num_tilings).astype(np.int64)
        coords = (quantised[None, :] + self._offsets) // self.num_tilings
        discrete = tuple(int(v) for v in np.rint(obs[self.discrete_dims]))
        return np.fromiter(
            (
                self.table.index((t, *map(int, coords[t]), *discrete), readonly=readonly)
                for t in range(self.num_tilings)
            ),
            dtype=np.int64,
            count=self.num_tilings,
        )


def observation_space_bounds(low: Any, high: Any, dims: Sequence[int]) -> list[tuple[float, float]]:
    """Bounds taken from an observation space, for the dimensions that have finite ones."""
    bounds = []
    for d in dims:
        lo, hi = float(low[d]), float(high[d])
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError(f"observation dimension {d} is unbounded; give explicit bounds")
        bounds.append((lo, hi))
    return bounds
