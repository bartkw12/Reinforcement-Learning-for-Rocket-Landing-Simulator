"""Figures for ablations: variants of one method that differ in a few settings."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from lunarlander_rl.analysis.aggregate import completed_runs
from lunarlander_rl.analysis.benchmark import ordered
from lunarlander_rl.analysis.plots import REFERENCE_COLOR, SOLVED_RETURN, new_figure, save_figure
from lunarlander_rl.analysis.tuning import settings
from lunarlander_rl.config import load_config

# Okabe-Ito palette (colour-blind safe) for telling variants of one method apart.
VARIANT_PALETTE = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442",
)  # fmt: skip


def variant_colors(variants: Sequence[str]) -> dict[str, str]:
    return {v: VARIANT_PALETTE[i % len(VARIANT_PALETTE)] for i, v in enumerate(variants)}


def plot_final_returns(
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    out: str | Path,
    order: Sequence[str] | None = None,
) -> list[Path]:
    """Each variant's final test return: seeds as dots, the mean with its 95% interval, and
    the mean best-checkpoint return as an open diamond."""
    rows = ordered(summary, order)
    colors = variant_colors(rows["variant"].tolist())
    fig = new_figure(7.5, 0.42 * len(rows) + 1.3)
    ax = fig.subplots()
    positions = np.arange(len(rows))[::-1]
    for y, (_, row) in zip(positions, rows.iterrows(), strict=True):
        color = colors[row["variant"]]
        seeds = runs.loc[runs["variant"] == row["variant"], "mean_return"]
        ax.scatter(seeds, np.full(len(seeds), y), s=12, color=color, alpha=0.45)
        ax.errorbar(
            row["mean_return"],
            y,
            xerr=[[row["mean_return"] - row["mean_return_ci_low"]],
                  [row["mean_return_ci_high"] - row["mean_return"]]],
            fmt="o", color=color, markersize=5, capsize=2, linewidth=1.2,
        )  # fmt: skip
        ax.scatter(row["best_mean_return"], y, marker="D", s=26, facecolor="none",
                   edgecolor=color, linewidth=1.0, zorder=3)  # fmt: skip
    ax.set_yticks(positions, [settings(v) for v in rows["variant"]])
    ax.axvline(SOLVED_RETURN, color=REFERENCE_COLOR, linewidth=0.6, linestyle=":")
    ax.set_xlabel("Test return (100 episodes)")
    fig.legend(
        handles=[
            Line2D([], [], marker="o", color="#444444", label="Final policy, mean [95% CI]"),
            Line2D([], [], marker="o", linestyle="", markersize=3, color="#444444", alpha=0.45,
                   label="Final policy, one seed"),
            Line2D([], [], marker="D", linestyle="", markerfacecolor="none",
                   markeredgecolor="#444444", label="Best checkpoint, mean"),
        ],
        loc="outside lower center",
        ncols=3,
        frameon=False,
    )  # fmt: skip
    return save_figure(fig, out)


def training_metric(experiment_dir: str | Path, metric: str, bin_steps: int) -> pd.DataFrame:
    """A per-episode training diagnostic (``q_mean``, say) averaged into bins of
    ``bin_steps`` environment steps, per run, so that runs can be averaged together."""
    frames = []
    for paths in completed_runs(experiment_dir):
        episodes = pd.read_csv(paths.train_csv, usecols=["env_step", metric]).dropna()
        episodes["bin"] = (episodes["env_step"] // bin_steps + 1) * bin_steps
        binned = episodes.groupby("bin", as_index=False)[metric].mean()
        config = load_config(paths.config)
        binned["variant"], binned["seed"] = config.variant, config.seed
        frames.append(binned)
    frame: pd.DataFrame = pd.concat(frames, ignore_index=True)
    return frame


def plot_training_metric(
    curves: pd.DataFrame,
    metric: str,
    y_label: str,
    out: str | Path,
    order: Sequence[str] | None = None,
) -> list[Path]:
    """Mean over seeds of a binned training diagnostic, with the seed range as a band."""
    fig = new_figure(6.5, 3.4)
    ax = fig.subplots()
    variants = ordered(curves[["variant"]].drop_duplicates(), order)["variant"].tolist()
    colors = variant_colors(variants)
    for variant in variants:
        wide = curves.loc[curves["variant"] == variant].pivot(
            index="bin", columns="seed", values=metric
        )
        ax.fill_between(wide.index, wide.min(axis=1), wide.max(axis=1),
                        color=colors[variant], alpha=0.15)  # fmt: skip
        ax.plot(wide.index, wide.mean(axis=1), color=colors[variant], linewidth=1.6,
                label=settings(variant))  # fmt: skip
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(y_label)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1e3:.0f}k")
    fig.legend(loc="outside right center", frameon=False)
    return save_figure(fig, out)
