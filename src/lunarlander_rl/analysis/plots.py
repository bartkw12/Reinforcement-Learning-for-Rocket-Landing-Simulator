"""Figure helpers shared by every report.

Figures are built with Matplotlib's object-oriented API (no pyplot), so they render
without a display and never leak global state between figures. Every plotting function is
wrapped in :func:`styled`, which applies the house style in ``style.mplstyle``.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path
from typing import ParamSpec, TypeVar

import matplotlib.style
from matplotlib.figure import Figure

STYLE_PATH = Path(__file__).with_name("style.mplstyle")

# Okabe-Ito colours. Each method keeps its colour in every figure of the project.
ALGORITHM_COLORS = {
    "q_learning": "#0072B2",
    "dqn": "#D55E00",
    "reinforce": "#009E73",
    "sb3_dqn": "#CC79A7",
    "sb3_ppo": "#E69F00",
    "random": "#999999",
}
ALGORITHM_NAMES = {
    "q_learning": "Q-learning",
    "dqn": "DQN",
    "reinforce": "REINFORCE",
    "sb3_dqn": "SB3 DQN",
    "sb3_ppo": "SB3 PPO",
    "random": "Random",
}
REFERENCE_COLOR = "#000000"
SOLVED_RETURN = 200.0

P = ParamSpec("P")
R = TypeVar("R")


def styled(function: Callable[P, R]) -> Callable[P, R]:
    """Run a plotting function inside the house style.

    Matplotlib reads style settings when an artist is created, so the whole function,
    not just the figure's creation, has to run inside the style context.
    """

    @functools.wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with matplotlib.style.context(STYLE_PATH):
            return function(*args, **kwargs)

    return wrapper


def new_figure(width: float, height: float) -> Figure:
    """A figure sized in inches, laid out automatically."""
    return Figure(figsize=(width, height), layout="constrained")


def save_figure(fig: Figure, path_without_suffix: str | Path) -> list[Path]:
    """Save as PNG (300 dpi, for the README) and PDF (vector, for the report), with no
    timestamps embedded so that regenerating unchanged figures gives identical files."""
    stem = Path(path_without_suffix)
    stem.parent.mkdir(parents=True, exist_ok=True)
    png, pdf = stem.with_suffix(".png"), stem.with_suffix(".pdf")
    fig.savefig(png, dpi=300, metadata={"Software": None})
    fig.savefig(pdf, metadata={"CreationDate": None, "Producer": None, "Creator": None})
    return [png, pdf]


def color_of(variant: str, agent: str | None = None) -> str:
    """The house colour of a method, looked up by variant label and then by agent name."""
    return ALGORITHM_COLORS.get(variant) or ALGORITHM_COLORS.get(agent or "", "#444444")
