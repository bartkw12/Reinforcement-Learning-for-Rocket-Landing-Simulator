"""Tables and figures for robustness re-evaluations (see ``lunarlander_rl.robustness``)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from lunarlander_rl.analysis.aggregate import OUTCOMES
from lunarlander_rl.analysis.benchmark import display_name, ordered
from lunarlander_rl.analysis.plots import (
    ALGORITHM_COLORS,
    REFERENCE_COLOR,
    SOLVED_RETURN,
    new_figure,
    save_figure,
    styled,
)
from lunarlander_rl.analysis.stats import bootstrap_ci
from lunarlander_rl.analysis.tables import markdown_table, percent, with_interval
from lunarlander_rl.config import load_config
from lunarlander_rl.robustness import RobustnessSpec, result_paths
from lunarlander_rl.tracking import read_json


def collect(spec: RobustnessSpec) -> pd.DataFrame:
    """One row per (run, condition)."""
    rows: list[dict[str, Any]] = []
    for paths, condition, path in result_paths(spec):
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}; run scripts/robustness.py first")
        summary = read_json(path)["summary"]
        config = load_config(paths.config)
        row: dict[str, Any] = {
            "variant": config.variant,
            "agent": config.agent.name,
            "seed": config.seed,
            "condition": condition.label,
            "level": condition.level,
            "mean_return": summary["mean_return"],
            "success_rate": summary["success_rate"],
        }
        for outcome in OUTCOMES:
            row[f"frac_{outcome}"] = summary["outcomes"].get(outcome, 0) / summary["episodes"]
        rows.append(row)
    frame: pd.DataFrame = pd.DataFrame(rows)
    return frame


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Per (variant, condition): means over seeds with bootstrap intervals over seeds."""
    rows: list[dict[str, Any]] = []
    for (variant, condition), group in results.groupby(["variant", "condition"], sort=False):
        row: dict[str, Any] = {
            "variant": variant,
            "agent": group["agent"].iloc[0],
            "condition": condition,
            "level": float(group["level"].iloc[0]),
            "seeds": len(group),
        }
        for column in ("mean_return", "success_rate"):
            values = group[column].astype(float).tolist()
            row[column] = float(np.mean(values))
            row[f"{column}_ci_low"], row[f"{column}_ci_high"] = bootstrap_ci(values)
        for outcome in OUTCOMES:
            row[f"frac_{outcome}"] = float(group[f"frac_{outcome}"].mean())
        rows.append(row)
    frame: pd.DataFrame = pd.DataFrame(rows).sort_values(["variant", "level"], ignore_index=True)
    return frame


def _wide(summary: pd.DataFrame, order: Sequence[str] | None) -> tuple[pd.DataFrame, list[str]]:
    conditions = (
        summary[["condition", "level"]].drop_duplicates().sort_values("level")["condition"]
    ).tolist()
    variants = ordered(summary[["variant"]].drop_duplicates(), order)
    return variants, conditions


def table(summary: pd.DataFrame, order: Sequence[str] | None = None) -> str:
    """Mean test return [95% CI over seeds] for each method and condition."""
    variants, conditions = _wide(summary, order)
    lookup = summary.set_index(["variant", "condition"])

    def cell(variant: str, condition: str) -> str:
        r = lookup.loc[(variant, condition)]
        return with_interval(r["mean_return"], r["mean_return_ci_low"], r["mean_return_ci_high"])

    def column(condition: str) -> tuple[str, Any]:
        return condition, lambda row: cell(row["variant"], condition)

    return markdown_table(
        variants,
        [("Agent", lambda row: display_name(row["variant"])), *(column(c) for c in conditions)],
    )


def success_table(summary: pd.DataFrame, order: Sequence[str] | None = None) -> str:
    """Success rate and the most common way of failing, per method and condition."""
    variants, conditions = _wide(summary, order)
    lookup = summary.set_index(["variant", "condition"])

    def cell(variant: str, condition: str) -> str:
        r = lookup.loc[(variant, condition)]
        failures = {o: r[f"frac_{o}"] for o in OUTCOMES if o != "landed"}
        worst = max(failures, key=failures.__getitem__)
        text = percent(r["success_rate"])
        if failures[worst] >= 0.05:
            text += f" ({percent(failures[worst])} {worst.replace('_', ' ')})"
        return text

    def column(condition: str) -> tuple[str, Any]:
        return condition, lambda row: cell(row["variant"], condition)

    return markdown_table(
        variants,
        [("Agent", lambda row: display_name(row["variant"])), *(column(c) for c in conditions)],
    )


@styled
def plot(
    summary: pd.DataFrame,
    out: str | Path,
    order: Sequence[str] | None = None,
    x_label: str = "Wind power (turbulence = wind / 10)",
) -> list[Path]:
    """Return and success rate against the condition level, one line per method."""
    fig = new_figure(9.0, 3.6)
    ax_return, ax_success = fig.subplots(1, 2)
    variants, _ = _wide(summary, order)
    handles = []
    for variant in variants["variant"]:
        group = summary.loc[summary["variant"] == variant].sort_values("level")
        color = ALGORITHM_COLORS.get(variant, ALGORITHM_COLORS.get(group["agent"].iloc[0]))
        for ax, column, scale in (
            (ax_return, "mean_return", 1.0),
            (ax_success, "success_rate", 100.0),
        ):
            ax.fill_between(
                group["level"],
                group[f"{column}_ci_low"] * scale,
                group[f"{column}_ci_high"] * scale,
                color=color,
                alpha=0.15,
            )
            ax.plot(
                group["level"],
                group[column] * scale,
                color=color,
                marker="o",
                markersize=3,
                linewidth=1.6,
            )
        handles.append(
            Line2D([], [], color=color, marker="o", markersize=3, label=display_name(variant))
        )
    ax_return.axhline(SOLVED_RETURN, color=REFERENCE_COLOR, linewidth=0.6, linestyle=":")
    ax_return.set_ylabel("Mean test return")
    ax_success.set_ylabel("Success rate (%)")
    ax_success.set_ylim(-3, 103)
    levels = sorted(summary["level"].unique())
    for ax in (ax_return, ax_success):
        ax.set_xlabel(x_label)
        ax.set_xticks(levels)
    fig.legend(handles=handles, loc="outside right center", frameon=False)
    return save_figure(fig, out)
