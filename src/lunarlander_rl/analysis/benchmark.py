"""Tables and figures for experiments that compare agents at an equal step budget."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from lunarlander_rl.analysis.plots import (
    ALGORITHM_COLORS,
    ALGORITHM_NAMES,
    REFERENCE_COLOR,
    SOLVED_RETURN,
    new_figure,
    save_figure,
)
from lunarlander_rl.analysis.tables import markdown_table, number, percent, with_interval


def display_name(variant: str) -> str:
    return ALGORITHM_NAMES.get(variant, variant)


def ordered(frame: pd.DataFrame, order: Sequence[str] | None) -> pd.DataFrame:
    """Rows in ``order`` of their variant (any others last, alphabetically)."""
    rank = {variant: i for i, variant in enumerate(order or ())}
    keys = [(rank.get(v, len(rank)), v) for v in frame["variant"]]
    positions = sorted(range(len(keys)), key=keys.__getitem__)
    result: pd.DataFrame = frame.iloc[positions].reset_index(drop=True)
    return result


def table(summary: pd.DataFrame, order: Sequence[str] | None = None) -> str:
    return markdown_table(
        ordered(summary, order),
        [
            ("Agent", lambda r: display_name(r["variant"])),
            ("Seeds", lambda r: f"{r['seeds']:.0f}"),
            ("Env steps", lambda r: f"{r['env_steps'] / 1e3:,.0f}k"),
            (
                "Final return [95% CI]",
                lambda r: with_interval(
                    r["mean_return"], r["mean_return_ci_low"], r["mean_return_ci_high"]
                ),
            ),
            ("IQM over seeds", lambda r: number(r["iqm_over_seeds"])),
            (
                "Seed range",
                lambda r: f"{number(r['seed_min_return'])} to {number(r['seed_max_return'])}",
            ),
            (
                "Success [95% CI]",
                lambda r: with_interval(
                    r["success_rate"], r["success_rate_ci_low"], r["success_rate_ci_high"], percent
                ),
            ),
            ("Reached 200", lambda r: _reached(r)),
            ("Best checkpoint return", lambda r: number(r["best_mean_return"])),
            ("Training time", lambda r: f"{r['training_minutes']:.0f} min"),
        ],
    )


def _reached(row: pd.Series) -> str:
    """``"8/10 runs, median 180k"``: runs whose evaluation reached a return of 200, and the
    median steps it took (shown only if at least half of the runs got there)."""
    text = f"{row['runs_reaching_200']:.0f}/{row['seeds']:.0f} runs"
    if not pd.isna(row["median_steps_to_200"]):
        text += f", median {row['median_steps_to_200'] / 1e3:,.0f}k"
    return text


def outcome_table(summary: pd.DataFrame, order: Sequence[str] | None = None) -> str:
    """How the final policies' test episodes ended, averaged over seeds."""
    return markdown_table(
        ordered(summary, order),
        [
            ("Agent", lambda r: display_name(r["variant"])),
            ("Landed", lambda r: percent(r["frac_landed"])),
            ("Crashed", lambda r: percent(r["frac_crashed"])),
            ("Out of bounds", lambda r: percent(r["frac_out_of_bounds"])),
            ("Timed out", lambda r: percent(r["frac_timeout"])),
        ],
    )


def plot_learning_curves(
    evaluations: pd.DataFrame, out: str | Path, order: Sequence[str] | None = None
) -> list[Path]:
    """Periodic evaluation return against environment steps.

    The line is the mean over seeds of each evaluation's mean return (10 episodes on fixed
    seeds); the band spans the lowest to the highest seed.
    """
    fig = new_figure(6.5, 3.8)
    ax = fig.subplots()
    variants = ordered(evaluations[["variant", "agent"]].drop_duplicates(), order)
    handles: list[Artist] = []
    for _, row in variants.iterrows():
        group = evaluations.loc[evaluations["variant"] == row["variant"]]
        wide = group.pivot(index="env_step", columns="seed", values="mean_return")
        color = ALGORITHM_COLORS.get(row["variant"], ALGORITHM_COLORS.get(row["agent"], "#444444"))
        ax.fill_between(wide.index, wide.min(axis=1), wide.max(axis=1), color=color, alpha=0.15)
        ax.plot(wide.index, wide.mean(axis=1), color=color, linewidth=1.6)
        handles.append(
            Line2D([], [], color=color, linewidth=1.6, label=display_name(row["variant"]))
        )
    ax.axhline(SOLVED_RETURN, color=REFERENCE_COLOR, linewidth=0.6, linestyle=":")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Evaluation return")
    ax.set_ylim(bottom=max(ax.get_ylim()[0], -600))
    ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1e3:.0f}k")
    handles += [
        Patch(color="#444444", alpha=0.15, label="Seed range"),
        Line2D([], [], color=REFERENCE_COLOR, linewidth=0.6, linestyle=":", label="Solved (200)"),
    ]
    fig.legend(handles=handles, loc="outside right center", frameon=False)
    return save_figure(fig, out)
