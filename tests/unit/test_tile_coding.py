import numpy as np
import pytest

from lunarlander_rl.components.tile_coding import (
    IndexHashTable,
    TileCoder,
    observation_space_bounds,
)


def make_coder(**kwargs: object) -> TileCoder:
    settings: dict[str, object] = {
        "bounds": [(-1.0, 1.0), (0.0, 2.0)],
        "num_tilings": 8,
        "tiles_per_dim": 4,
        "table_size": 4096,
        "continuous_dims": [0, 1],
        "discrete_dims": [2],
    }
    settings.update(kwargs)
    return TileCoder(**settings)  # type: ignore[arg-type]


def test_one_active_tile_per_tiling() -> None:
    coder = make_coder()
    tiles = coder(np.array([0.1, 0.5, 0.0]))
    assert tiles.shape == (8,)
    assert len(set(tiles.tolist())) == 8


def test_coding_is_deterministic_and_order_independent() -> None:
    a, b = make_coder(), make_coder()
    states = [np.array([x, y, 0.0]) for x, y in [(0.1, 0.5), (-0.7, 1.9), (0.9, 0.0)]]
    first = [a(s) for s in states]
    # A fresh coder fed the same states in the same order assigns the same indices.
    np.testing.assert_array_equal(np.stack(first), np.stack([b(s) for s in states]))
    # Re-coding a known state never changes its indices.
    np.testing.assert_array_equal(a(states[0]), first[0])


def test_nearby_states_share_more_tiles_than_distant_ones() -> None:
    coder = make_coder()
    base = set(coder(np.array([0.0, 1.0, 0.0])).tolist())
    near = set(coder(np.array([0.02, 1.0, 0.0])).tolist())
    far = set(coder(np.array([0.9, 0.1, 0.0])).tolist())
    assert len(base & near) > len(base & far)
    assert len(base & far) == 0


def test_discrete_dimensions_select_separate_tiles() -> None:
    coder = make_coder()
    ground = set(coder(np.array([0.0, 1.0, 1.0])).tolist())
    air = set(coder(np.array([0.0, 1.0, 0.0])).tolist())
    assert ground.isdisjoint(air)


def test_states_outside_bounds_use_the_edge_tiles() -> None:
    coder = make_coder()
    np.testing.assert_array_equal(
        coder(np.array([5.0, 1.0, 0.0])), coder(np.array([1.0, 1.0, 0.0]))
    )


def test_tilings_use_asymmetric_offsets() -> None:
    # Move along dimension 0 only: with offsets t * (2i + 1), dimension 0 moves one
    # fraction of a tile per tiling and dimension 1 three, so tilings change at different
    # points. With uniform offsets the two dimensions would always change together.
    coder = make_coder(num_tilings=4, tiles_per_dim=1, bounds=[(0.0, 1.0), (0.0, 1.0)])
    offsets = coder._offsets
    np.testing.assert_array_equal(offsets[:, 0], [0, 1, 2, 3])
    np.testing.assert_array_equal(offsets[:, 1], [0, 3, 6, 9])


def test_table_counts_collisions_once_full() -> None:
    coder = make_coder(table_size=10)
    rng = np.random.default_rng(0)
    for _ in range(50):
        tiles = coder(np.array([rng.uniform(-1, 1), rng.uniform(0, 2), 0.0]))
        assert tiles.min() >= 0 and tiles.max() < 10
    assert coder.table.usage == 1.0
    assert coder.table.collisions > 0


def test_readonly_coding_never_changes_the_table() -> None:
    coder = make_coder()
    known = coder(np.array([0.0, 1.0, 0.0]))
    size = len(coder.table.dictionary)
    np.testing.assert_array_equal(coder(np.array([0.0, 1.0, 0.0]), readonly=True), known)
    unseen = coder(np.array([0.9, 1.9, 1.0]), readonly=True)
    assert (unseen == -1).all()
    assert len(coder.table.dictionary) == size


def test_table_state_round_trip() -> None:
    coder = make_coder()
    for x in np.linspace(-1, 1, 7):
        coder(np.array([x, 1.0, 0.0]))
    restored = IndexHashTable(coder.table.size)
    restored.load_state_dict(coder.table.state_dict())
    assert restored.dictionary == coder.table.dictionary


def test_validation() -> None:
    with pytest.raises(ValueError):
        make_coder(bounds=[(-1.0, 1.0)])
    with pytest.raises(ValueError):
        make_coder(bounds=[(1.0, -1.0), (0.0, 1.0)])
    with pytest.raises(ValueError):
        IndexHashTable(0)


def test_observation_space_bounds() -> None:
    low, high = np.array([-1.0, -np.inf]), np.array([1.0, np.inf])
    assert observation_space_bounds(low, high, [0]) == [(-1.0, 1.0)]
    with pytest.raises(ValueError, match="unbounded"):
        observation_space_bounds(low, high, [1])
