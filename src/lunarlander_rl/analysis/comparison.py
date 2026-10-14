"""Statistical comparison of methods: the main benchmark's headline figures and tables.

Follows Agarwal et al. (NeurIPS 2021): interval estimates of aggregate metrics instead of
point estimates, performance profiles instead of a single summary number, and the
probability of improvement between pairs of methods. The score of a run is its final
policy's mean return over the 100 held-out test episodes.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from lunarlander_rl.analysis import stats
from lunarlander_rl.analysis.aggregate import OUTCOMES
from lunarlander_rl.analysis.benchmark import display_name
from lunarlander_rl.analysis.plots import (
    REFERENCE_COLOR,
    SOLVED_RETURN,
    color_of,
    new_figure,
    save_figure,
    styled,
)
from lunarlander_rl.analysis.tables import markdown_table, number, percent, with_interval

OUTCOME_COLORS = {
    "landed": "#009E73",
    "timeout": "#E69F00",
    "crashed": "#D55E00",
    "out_of_bounds": "#56B4E9",
}


def methods_in(frame: pd.DataFrame, order: Sequence[str] | None) -> list[str]:
    present = list(dict.fromkeys(frame["variant"]))
    listed = [v for v in (order or ()) if v in present]
    return listed + sorted(v for v in present if v not in listed)


def run_scores(runs: pd.DataFrame, order: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    """Each method's final test return, one value per run."""
    return {
        m: runs.loc[runs["variant"] == m, "mean_return"].to_numpy(float)
        for m in methods_in(runs, order)
    }


def interval_estimates(runs: pd.DataFrame, order: Sequence[str] | None = None) -> pd.DataFrame:
    estimates = stats.interval_estimates(run_scores(runs, order))
    renamed: pd.DataFrame = estimates.rename(columns={"method": "variant"})
    return renamed


def improvement_matrix(runs: pd.DataFrame, order: Sequence[str] | None = None) -> pd.DataFrame:
    """P(row method > column method) for every ordered pair, with intervals."""
    scores = run_scores(runs, order)
    rows: list[dict[str, Any]] = []
    for index, (x, y) in enumerate(itertools.permutations(scores, 2)):
        estimate, low, high = stats.probability_of_improvement_ci(scores[x], scores[y], seed=index)
        rows.append({"x": x, "y": y, "value": estimate, "ci_low": low, "ci_high": high})
    frame: pd.DataFrame = pd.DataFrame(rows)
    return frame


# ------------------------------------------------------------------------------ tables
def results_table(
    runs: pd.DataFrame,
    estimates: pd.DataFrame,
    summary: pd.DataFrame,
    order: Sequence[str] | None = None,
) -> str:
    """The headline table: aggregate metrics with intervals, success and cost."""
    lookup = estimates.set_index(["variant", "metric"])
    info = summary.set_index("variant")

    def metric(name: str) -> Any:
        def cell(row: pd.Series) -> str:
            r = lookup.loc[(row["variant"], name)]
            return with_interval(r["value"], r["ci_low"], r["ci_high"])

        return cell

    def spread(row: pd.Series) -> str:
        values = runs.loc[runs["variant"] == row["variant"], "mean_return"]
        return f"{values.mean():.1f} ± {values.std(ddof=1):.1f}"

    def reached(row: pd.Series) -> str:
        r = info.loc[row["variant"]]
        text = f"{r['runs_reaching_200']:.0f}/{r['seeds']:.0f}"
        if not pd.isna(r["median_steps_to_200"]):
            text += f" ({r['median_steps_to_200'] / 1e3:,.0f}k)"
        return text

    frame = pd.DataFrame({"variant": methods_in(runs, order)})
    return markdown_table(
        frame,
        [
            ("Method", lambda r: display_name(r["variant"])),
            ("IQM [95% CI]", metric("IQM")),
            ("Mean [95% CI]", metric("Mean")),
            ("Mean ± SD over seeds", spread),
            ("Optimality gap [95% CI]", metric("Optimality gap")),
            (
                "Success [95% CI]",
                lambda r: with_interval(
                    info.loc[r["variant"], "success_rate"],
                    info.loc[r["variant"], "success_rate_ci_low"],
                    info.loc[r["variant"], "success_rate_ci_high"],
                    percent,
                ),
            ),
            ("Runs reaching 200 (median steps)", reached),
            ("Training time", lambda r: f"{info.loc[r['variant'], 'training_minutes']:.0f} min"),
        ],
    )


def headline_table(
    estimates: pd.DataFrame, summary: pd.DataFrame, order: Sequence[str] | None = None
) -> str:
    """The README's compact version of :func:`results_table`."""
    lookup = estimates.set_index(["variant", "metric"])
    info = summary.set_index("variant")

    def metric(name: str) -> Any:
        def cell(row: pd.Series) -> str:
            r = lookup.loc[(row["variant"], name)]
            return with_interval(r["value"], r["ci_low"], r["ci_high"])

        return cell

    return markdown_table(
        pd.DataFrame({"variant": methods_in(estimates, order)}),
        [
            ("Method", lambda r: display_name(r["variant"])),
            ("IQM test return [95% CI]", metric("IQM")),
            ("Mean test return [95% CI]", metric("Mean")),
            ("Success", lambda r: percent(info.loc[r["variant"], "success_rate"])),
            ("Training time", lambda r: f"{info.loc[r['variant'], 'training_minutes']:.0f} min"),
        ],
    )


def improvement_table(matrix: pd.DataFrame, order: Sequence[str] | None = None) -> str:
    """P(row > column) as a square table; above 0.5 favours the row's method."""
    methods = [m for m in (order or ()) if m in set(matrix["x"])] or sorted(set(matrix["x"]))
    lookup = matrix.set_index(["x", "y"])

    def column(y: str) -> tuple[str, Any]:
        def cell(row: pd.Series) -> str:
            if row["variant"] == y:
                return "n/a"
            r = lookup.loc[(row["variant"], y)]
            return with_interval(r["value"], r["ci_low"], r["ci_high"], lambda v: number(v, 2))

        return display_name(y), cell

    return markdown_table(
        pd.DataFrame({"variant": methods}),
        [("P(row > column)", lambda r: display_name(r["variant"])), *(column(y) for y in methods)],
    )


# ----------------------------------------------------------------------------- figures
@styled
def plot_iqm_curves(
    evaluations: pd.DataFrame, out: str | Path, order: Sequence[str] | None = None
) -> list[Path]:
    """F1: IQM over seeds of the periodic evaluation return, with a 95% bootstrap band."""
    fig = new_figure(6.5, 3.6)
    ax = fig.subplots()
    handles = []
    for method in methods_in(evaluations, order):
        group = evaluations.loc[evaluations["variant"] == method]
        wide = group.pivot(index="env_step", columns="seed", values="mean_return").dropna()
        estimate, low, high = stats.curve_estimates(wide.to_numpy(float).T)
        color = color_of(method, group["agent"].iloc[0])
        ax.fill_between(wide.index, low, high, color=color, alpha=0.18, linewidth=0)
        ax.plot(wide.index, estimate, color=color)
        handles.append(Line2D([], [], color=color, label=display_name(method)))
    ax.axhline(SOLVED_RETURN, color=REFERENCE_COLOR, linewidth=0.6, linestyle=":")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("IQM evaluation return")
    ax.set_ylim(-350, 320)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1e3:.0f}k")
    handles.append(Line2D([], [], color=REFERENCE_COLOR, linewidth=0.6, linestyle=":",
                          label="Solved (200)"))  # fmt: skip
    fig.legend(handles=handles, loc="outside right center")
    return save_figure(fig, out)


@styled
def plot_aggregate_metrics(
    estimates: pd.DataFrame, out: str | Path, order: Sequence[str] | None = None
) -> list[Path]:
    """F2: point estimates and 95% intervals of four aggregate metrics, one panel each."""
    metrics = ["Median", "IQM", "Mean", "Optimality gap"]
    methods = methods_in(estimates, order)
    fig = new_figure(9.0, 0.32 * len(methods) + 1.1)
    axes = fig.subplots(1, len(metrics), sharey=True)
    positions = np.arange(len(methods))[::-1]
    lookup = estimates.set_index(["variant", "metric"])
    for ax, name in zip(axes, metrics, strict=True):
        for y, method in zip(positions, methods, strict=True):
            r = lookup.loc[(method, name)]
            color = color_of(method)
            ax.barh(y, r["ci_high"] - r["ci_low"], left=r["ci_low"], height=0.6,
                    color=color, alpha=0.75)  # fmt: skip
            ax.plot([r["value"]] * 2, [y - 0.3, y + 0.3], color="black", linewidth=1.4)
        ax.set_title(name)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Return shortfall below 200" if name == "Optimality gap" else "Test return")
    axes[0].set_yticks(positions, [display_name(m) for m in methods])
    axes[0].tick_params(axis="y", length=0)
    return save_figure(fig, out)


@styled
def plot_performance_profiles(
    episodes: pd.DataFrame, out: str | Path, order: Sequence[str] | None = None
) -> list[Path]:
    """F3: fraction of test episodes scoring above each return threshold, with 95% bands.

    A curve that lies above another for every threshold dominates it, whatever summary
    statistic one prefers."""
    taus = np.linspace(-300.0, 320.0, 125)
    fig = new_figure(6.5, 3.6)
    ax = fig.subplots()
    handles = []
    for method in methods_in(episodes, order):
        group = episodes.loc[episodes["variant"] == method]
        per_run = [g["episode_return"].to_numpy(float) for _, g in group.groupby("seed")]
        profile, low, high = stats.performance_profile(per_run, taus)
        color = color_of(method, group["agent"].iloc[0])
        ax.fill_between(taus, low, high, color=color, alpha=0.18, linewidth=0)
        ax.plot(taus, profile, color=color)
        handles.append(Line2D([], [], color=color, label=display_name(method)))
    ax.axvline(SOLVED_RETURN, color=REFERENCE_COLOR, linewidth=0.6, linestyle=":")
    ax.set_xlabel("Return threshold τ")
    ax.set_ylabel("Fraction of test episodes > τ")
    ax.set_ylim(0, 1.02)
    fig.legend(handles=handles, loc="outside right center")
    return save_figure(fig, out)


@styled
def plot_sample_efficiency(
    runs: pd.DataFrame, out: str | Path, order: Sequence[str] | None = None,
    budget: float = 1_000_000,
) -> list[Path]:  # fmt: skip
    """F4: environment steps until each run's periodic evaluation first averaged 200.

    Runs that never got there are drawn at the right edge, as crosses."""
    methods = methods_in(runs, order)
    fig = new_figure(6.5, 0.36 * len(methods) + 1.0)
    ax = fig.subplots()
    positions = np.arange(len(methods))[::-1]
    jitter = np.random.default_rng(0)
    for y, method in zip(positions, methods, strict=True):
        steps = runs.loc[runs["variant"] == method, "steps_to_200"].to_numpy(float)
        color = color_of(method)
        reached = steps[np.isfinite(steps)]
        offsets = jitter.uniform(-0.15, 0.15, size=len(steps))
        ax.scatter(reached, y + offsets[: len(reached)], s=16, color=color, alpha=0.75)
        missing = len(steps) - len(reached)
        if missing:
            ax.scatter([budget * 1.04] * missing, y + offsets[len(reached):], marker="x",
                       s=18, color=color)  # fmt: skip
        if len(reached) * 2 >= len(steps):
            median = float(np.median(np.where(np.isfinite(steps), steps, np.inf)))
            ax.plot([median] * 2, [y - 0.32, y + 0.32], color="black", linewidth=1.4)
    ax.set_yticks(positions, [display_name(m) for m in methods])
    ax.set_xlim(0, budget * 1.08)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1e3:.0f}k")
    ax.set_xlabel("Environment steps until an evaluation first averaged 200")
    ax.grid(axis="y", visible=False)
    fig.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", color="#444444", label="One run"),
            Line2D([], [], color="black", linewidth=1.4, label="Median over runs"),
            Line2D([], [], marker="x", linestyle="", color="#444444", label="Never reached"),
        ],
        loc="outside lower center",
        ncols=3,
    )
    return save_figure(fig, out)


@styled
def plot_outcomes(
    summary: pd.DataFrame, out: str | Path, order: Sequence[str] | None = None
) -> list[Path]:
    """F5: how the final policies' test episodes ended (mean over seeds)."""
    methods = methods_in(summary, order)
    info = summary.set_index("variant")
    fig = new_figure(6.5, 0.36 * len(methods) + 1.0)
    ax = fig.subplots()
    positions = np.arange(len(methods))[::-1]
    left = np.zeros(len(methods))
    for outcome in ("landed", "timeout", "crashed", "out_of_bounds"):
        assert outcome in OUTCOMES
        widths = np.array([100 * info.loc[m, f"frac_{outcome}"] for m in methods])
        ax.barh(positions, widths, left=left, height=0.62, color=OUTCOME_COLORS[outcome])
        for y, x, w in zip(positions, left, widths, strict=True):
            if w >= 6:
                ax.text(x + w / 2, y, f"{w:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if outcome != "timeout" else "black")  # fmt: skip
        left += widths
    ax.set_yticks(positions, [display_name(m) for m in methods])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of test episodes (%)")
    ax.grid(visible=False)
    ax.tick_params(axis="y", length=0)
    fig.legend(
        handles=[
            Patch(color=OUTCOME_COLORS[o], label=o.replace("_", " ").capitalize())
            for o in ("landed", "timeout", "crashed", "out_of_bounds")
        ],
        loc="outside lower center",
        ncols=4,
    )
    return save_figure(fig, out)


@styled
def plot_probability_of_improvement(
    matrix: pd.DataFrame, out: str | Path, order: Sequence[str] | None = None
) -> list[Path]:
    """P(X > Y) with 95% intervals for every pair where X is listed before Y in ``order``."""
    methods = [m for m in (order or ()) if m in set(matrix["x"])] or sorted(set(matrix["x"]))
    pairs = [(x, y) for i, x in enumerate(methods) for y in methods[i + 1 :]]
    lookup = matrix.set_index(["x", "y"])
    fig = new_figure(6.0, 0.28 * len(pairs) + 1.0)
    ax = fig.subplots()
    positions = np.arange(len(pairs))[::-1]
    for y, (a, b) in zip(positions, pairs, strict=True):
        r = lookup.loc[(a, b)]
        ax.barh(y, r["ci_high"] - r["ci_low"], left=r["ci_low"], height=0.6,
                color=color_of(a), alpha=0.75)  # fmt: skip
        ax.plot([r["value"]] * 2, [y - 0.3, y + 0.3], color="black", linewidth=1.4)
    ax.axvline(0.5, color=REFERENCE_COLOR, linewidth=0.6, linestyle=":")
    ax.set_yticks(positions, [f"{display_name(a)} > {display_name(b)}" for a, b in pairs])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability of improvement P(X > Y) over runs")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    return save_figure(fig, out)
