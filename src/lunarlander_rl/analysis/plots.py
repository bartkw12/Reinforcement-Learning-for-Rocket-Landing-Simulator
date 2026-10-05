"""Figure helpers shared by every report.

Figures are built with Matplotlib's object-oriented API (no pyplot), so they render
without a display and never leak global state between figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from matplotlib.figure import Figure

# Okabe-Ito palette: distinguishable with the common forms of colour blindness. Each
# algorithm keeps its colour in every figure of the project.
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

STYLE: dict[str, Any] = {
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
}


def new_figure(width: float, height: float) -> Figure:
    """A figure sized in inches; :func:`save_figure` applies the house style."""
    return Figure(figsize=(width, height), layout="constrained")


def apply_style(fig: Figure) -> None:
    """Style every axis of ``fig`` (rcParams only apply to axes created inside them)."""
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=STYLE["xtick.labelsize"])
        ax.title.set_fontsize(STYLE["axes.titlesize"])
        ax.xaxis.label.set_fontsize(STYLE["axes.labelsize"])
        ax.yaxis.label.set_fontsize(STYLE["axes.labelsize"])


def save_figure(fig: Figure, path_without_suffix: str | Path) -> list[Path]:
    """Save as PNG (for the README) and PDF (for the report), with no timestamps
    embedded so that regenerating unchanged figures gives identical files."""
    stem = Path(path_without_suffix)
    stem.parent.mkdir(parents=True, exist_ok=True)
    apply_style(fig)
    png, pdf = stem.with_suffix(".png"), stem.with_suffix(".pdf")
    fig.savefig(png, dpi=200, metadata={"Software": None})
    fig.savefig(pdf, metadata={"CreationDate": None, "Producer": None, "Creator": None})
    return [png, pdf]
