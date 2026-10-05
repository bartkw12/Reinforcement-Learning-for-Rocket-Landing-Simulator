"""Tables and figures comparing the v1.0 results with their v2 replication."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from lunarlander_rl.analysis.plots import (
    ALGORITHM_COLORS,
    ALGORITHM_NAMES,
    REFERENCE_COLOR,
    SOLVED_RETURN,
    new_figure,
    save_figure,
)
from lunarlander_rl.analysis.tables import markdown_table, number, percent, with_interval

AGENT_ORDER = ("q_learning", "dqn", "reinforce")
_VARIANT = re.compile(r"^(?P<agent>[a-z_]+?)_cfg(?P<config>\d+)(?P<bugs>_v1bugs)?$")
BUGGY_COLOR = "#7F7F7F"


def parse_variant(variant: str) -> tuple[str, int, bool]:
    """``"dqn_cfg1_v1bugs"`` -> ``("dqn", 1, True)``."""
    match = _VARIANT.match(variant)
    if match is None:
        raise ValueError(f"not a replication variant: {variant!r}")
    return match["agent"], int(match["config"]), match["bugs"] is not None


def describe(variant: str) -> str:
    agent, config, bugs = parse_variant(variant)
    text = f"{ALGORITHM_NAMES[agent]}, config {config}"
    return text + " (v1 bugs)" if bugs else text


def comparison(summary: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """v2 summaries side by side with the v1.0 numbers, in report order."""
    merged = summary.merge(
        reference.rename(
            columns={
                "mean_return": "v1_mean_return",
                "std_return": "v1_std_return",
                "success_rate": "v1_success_rate",
                "episodes": "v1_eval_episodes",
            }
        ),
        on="variant",
        how="left",
    )
    parsed = merged["variant"].map(parse_variant)
    merged["config"] = parsed.map(lambda p: p[1])
    merged["v1_bugs"] = parsed.map(lambda p: p[2])
    merged["agent_rank"] = parsed.map(lambda p: AGENT_ORDER.index(p[0]))
    merged = merged.sort_values(["agent_rank", "config", "v1_bugs"], ignore_index=True)
    # Bug-reproducing variants are compared with the v1 number of the config they copy.
    for i, row in merged.iterrows():
        if row["v1_bugs"]:
            source_variant = str(row["variant"]).removesuffix("_v1bugs")
            source = reference.loc[reference["variant"] == source_variant]
            if not source.empty:
                merged.loc[i, "v1_mean_return"] = source["mean_return"].iloc[0]
                merged.loc[i, "v1_success_rate"] = source["success_rate"].iloc[0]
    result: pd.DataFrame = merged.drop(columns="agent_rank")
    return result


def table(rows: pd.DataFrame) -> str:
    return markdown_table(
        rows,
        [
            ("Configuration", lambda r: describe(r["variant"])),
            ("Budget", lambda r: f"{r['episodes']:,.0f} ep / {r['env_steps'] / 1e3:,.0f}k steps"),
            ("v1 return", lambda r: number(r["v1_mean_return"])),
            (
                "v2 return [95% CI]",
                lambda r: with_interval(
                    r["mean_return"], r["mean_return_ci_low"], r["mean_return_ci_high"]
                ),
            ),
            (
                "v2 seed range",
                lambda r: f"{number(r['seed_min_return'])} to {number(r['seed_max_return'])}",
            ),
            ("v1 success", lambda r: percent(r["v1_success_rate"])),
            (
                "v2 success [95% CI]",
                lambda r: with_interval(
                    r["success_rate"],
                    r["success_rate_ci_low"],
                    r["success_rate_ci_high"],
                    percent,
                ),
            ),
        ],
    )


def plot_learning_curves(curves: pd.DataFrame, out: str | Path) -> list[Path]:
    """v1.0-style training curves (100-episode moving average of training returns).

    One panel per configuration. Thin lines are seeds; the thick line is their mean.
    Bug-reproducing variants are drawn in grey in the panel of the config they copy.
    """
    fig = new_figure(9.0, 7.0)
    axes = fig.subplots(len(AGENT_ORDER), 3, squeeze=False)
    for variant, group in curves.groupby("variant"):
        agent, config, bugs = parse_variant(str(variant))
        ax = axes[AGENT_ORDER.index(agent), config - 1]
        color = BUGGY_COLOR if bugs else ALGORITHM_COLORS[agent]
        wide = group.pivot(index="episode", columns="seed", values="moving_average")
        for seed in wide.columns:
            ax.plot(wide.index, wide[seed], color=color, alpha=0.25, linewidth=0.6)
        ax.plot(
            wide.index,
            wide.mean(axis=1),
            color=color,
            linewidth=1.6,
            linestyle="--" if bugs else "-",
        )
        ax.set_title(f"{ALGORITHM_NAMES[agent]}, config {config}")
    for ax in axes.flat:
        ax.axhline(SOLVED_RETURN, color=REFERENCE_COLOR, linewidth=0.6, linestyle=":")
        ax.set_ylim(-500, 320)
    for ax in axes[-1]:
        ax.set_xlabel("Training episode")
    for ax in axes[:, 0]:
        ax.set_ylabel("Return (100-episode avg.)")
    fig.legend(
        handles=[
            Line2D([], [], color="#444444", linewidth=1.6, label="v2, mean over seeds"),
            Line2D([], [], color="#444444", linewidth=0.6, alpha=0.4, label="v2, single seed"),
            Line2D(
                [],
                [],
                color=BUGGY_COLOR,
                linewidth=1.6,
                linestyle="--",
                label="v2 with v1 bugs re-introduced",
            ),
            Line2D(
                [], [], color=REFERENCE_COLOR, linewidth=0.6, linestyle=":", label="Solved (200)"
            ),
        ],
        loc="outside lower center",
        ncols=4,
        frameon=False,
    )
    return save_figure(fig, out)


def plot_final_comparison(rows: pd.DataFrame, runs: pd.DataFrame, out: str | Path) -> list[Path]:
    """Final test performance: v1.0's single run against v2's seeds."""
    fig = new_figure(9.0, 0.45 * len(rows) + 1.2)
    ax_return, ax_success = fig.subplots(1, 2, sharey=True)
    positions = np.arange(len(rows))[::-1]
    for y, (_, row) in zip(positions, rows.iterrows(), strict=True):
        agent, _, bugs = parse_variant(row["variant"])
        color = BUGGY_COLOR if bugs else ALGORITHM_COLORS[agent]
        seeds = runs.loc[runs["variant"] == row["variant"]]
        for ax, column, scale in (
            (ax_return, "mean_return", 1.0),
            (ax_success, "success_rate", 100.0),
        ):
            ax.scatter(seeds[column] * scale, np.full(len(seeds), y), s=10, color=color, alpha=0.4)
            low = (row[column] - row[f"{column}_ci_low"]) * scale
            high = (row[f"{column}_ci_high"] - row[column]) * scale
            ax.errorbar(
                row[column] * scale,
                y,
                xerr=[[low], [high]],
                fmt="o",
                color=color,
                markersize=5,
                capsize=2,
                linewidth=1.2,
            )
            if not pd.isna(row[f"v1_{column}"]):
                ax.scatter(
                    row[f"v1_{column}"] * scale,
                    y,
                    marker="D",
                    s=28,
                    facecolor="none",
                    edgecolor=REFERENCE_COLOR,
                    linewidth=1.0,
                    zorder=3,
                )
    ax_return.set_yticks(positions, [describe(v) for v in rows["variant"]])
    ax_return.axvline(SOLVED_RETURN, color=REFERENCE_COLOR, linewidth=0.6, linestyle=":")
    ax_return.set_xlabel("Mean test return (100 episodes)")
    ax_success.set_xlabel("Success rate (%)")
    ax_success.set_xlim(-3, 103)
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="D",
                linestyle="",
                markerfacecolor="none",
                markeredgecolor=REFERENCE_COLOR,
                label="v1 (one run)",
            ),
            Line2D([], [], marker="o", color="#444444", label="v2 mean [95% CI]"),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=3,
                color="#444444",
                alpha=0.4,
                label="v2 seeds",
            ),
        ],
        loc="outside lower center",
        ncols=3,
        frameon=False,
    )
    return save_figure(fig, out)
