"""Neural network building blocks."""

from __future__ import annotations

from collections.abc import Sequence

from torch import nn


def mlp(input_dim: int, hidden_sizes: Sequence[int], output_dim: int) -> nn.Sequential:
    """Fully connected network with ReLU activations between layers and a linear output."""
    layers: list[nn.Module] = []
    previous = input_dim
    for size in hidden_sizes:
        layers += [nn.Linear(previous, size), nn.ReLU()]
        previous = size
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)
