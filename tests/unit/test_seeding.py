import random

import numpy as np
import torch

from lunarlander_rl.seeding import derive_seed, make_rng, seed_everything


def test_derive_seed_is_stable_and_separates_streams() -> None:
    assert derive_seed(0, "train_env") == derive_seed(0, "train_env")
    assert derive_seed(0, "train_env") != derive_seed(0, "agent")
    assert derive_seed(0, "train_env") != derive_seed(1, "train_env")


def test_derive_seed_does_not_depend_on_hash_randomisation() -> None:
    # Pinned value: a change here would silently change every experiment.
    assert derive_seed(0, "train_env") == 234875633


def test_make_rng_reproduces_its_stream() -> None:
    first = make_rng(5, "agent").integers(0, 1000, size=20)
    second = make_rng(5, "agent").integers(0, 1000, size=20)
    np.testing.assert_array_equal(first, second)


def test_seed_everything_makes_all_generators_reproducible() -> None:
    def draw() -> tuple[float, float, float]:
        seed_everything(123)
        return (
            random.random(),
            float(np.random.rand()),  # noqa: NPY002 - the legacy global generator is the subject
            float(torch.rand(1).item()),
        )

    assert draw() == draw()


def test_seed_everything_sets_thread_count() -> None:
    seed_everything(0, torch_threads=1)
    assert torch.get_num_threads() == 1
