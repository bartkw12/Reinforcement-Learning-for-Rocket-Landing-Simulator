"""Deterministic seeding of every source of randomness in a run."""

from __future__ import annotations

import random
import zlib

import numpy as np
import torch


def derive_seed(base_seed: int, stream: str) -> int:
    """Derive an independent seed for a named random stream of a run.

    Giving each consumer (training environment, agent exploration, ...) its own stream
    means that adding or reordering random draws in one of them cannot change the others.
    """
    stream_id = zlib.crc32(stream.encode("utf-8"))
    return int(np.random.SeedSequence([base_seed, stream_id]).generate_state(1)[0])


def make_rng(base_seed: int, stream: str) -> np.random.Generator:
    return np.random.default_rng(derive_seed(base_seed, stream))


def seed_everything(seed: int, *, torch_threads: int = 1, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch, and request deterministic PyTorch algorithms.

    Environments are seeded separately, through ``env.reset(seed=...)``.
    """
    random.seed(seed)
    # Legacy global generator: this project draws from explicit Generators, but
    # third-party code may still use the global one.
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(deterministic)
