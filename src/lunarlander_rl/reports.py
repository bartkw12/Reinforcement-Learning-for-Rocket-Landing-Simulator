"""Every experiment of the project and how its tables and figures are built.

Each :class:`Report` names an experiment, the config file that defines its runs, and a
function that turns the finished runs into files under ``results/``. ``scripts/report.py``
builds one or all of them; ``scripts/reproduce_all.py`` runs the experiments first.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from lunarlander_rl import media
from lunarlander_rl.analysis import ablation, benchmark, comparison, replication, tuning
from lunarlander_rl.analysis import robustness as robustness_report
from lunarlander_rl.analysis.aggregate import (
    episode_returns,
    evaluation_curves,
    final_results,
    summarize_variants,
    training_curves,
)
from lunarlander_rl.robustness import RobustnessSpec, load_spec

MAIN_ORDER = ("sb3_ppo", "dqn", "sb3_dqn", "q_learning", "reinforce")
REFERENCE_V1 = Path("results/reference/v1_reported.csv")


@dataclass(frozen=True)
class Paths:
    """Where runs are read from and results are written to."""

    runs: Path = Path("runs")
    results: Path = Path("results")

    def experiment(self, name: str) -> Path:
        return self.runs / name

    @property
    def summary(self) -> Path:
        return self.results / "summary"

    @property
    def tables(self) -> Path:
        return self.results / "tables"

    def figures(self, name: str) -> Path:
        return self.results / "figures" / name


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def _aggregate(paths: Paths, name: str) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    runs = final_results(paths.experiment(name))
    summary = summarize_variants(runs)
    written = [
        _write_csv(paths.summary / f"{name}_runs.csv", runs),
        _write_csv(paths.summary / f"{name}_summary.csv", summary),
    ]
    return runs, summary, written


def _benchmark_tables(
    paths: Paths, name: str, summary: pd.DataFrame, order: Sequence[str] | None
) -> list[Path]:
    return [
        _write(paths.tables / f"{name}.md", benchmark.table(summary, order)),
        _write(paths.tables / f"{name}_outcomes.md", benchmark.outcome_table(summary, order)),
    ]


# ------------------------------------------------------------------------ builders
def build_replication(paths: Paths, name: str) -> list[Path]:
    runs, summary, written = _aggregate(paths, name)
    rows = replication.comparison(summary, pd.read_csv(REFERENCE_V1))
    out = paths.figures(name)
    return [
        *written,
        _write(paths.tables / f"{name}.md", replication.table(rows)),
        *replication.plot_learning_curves(
            training_curves(paths.experiment(name)), out / "learning_curves"
        ),
        *replication.plot_final_comparison(rows, runs, out / "final_performance"),
    ]


def build_tuning(paths: Paths, name: str) -> list[Path]:
    _, summary, written = _aggregate(paths, name)
    return [
        *written,
        _write(paths.tables / f"{name}.md", tuning.table(summary)),
        _write_csv(paths.summary / f"{name}_selection.csv", tuning.selection(summary)),
    ]


def benchmark_builder(order: Sequence[str]) -> Callable[[Paths, str], list[Path]]:
    def build(paths: Paths, name: str) -> list[Path]:
        _, summary, written = _aggregate(paths, name)
        return [
            *written,
            *_benchmark_tables(paths, name, summary, order),
            *benchmark.plot_learning_curves(
                evaluation_curves(paths.experiment(name)),
                paths.figures(name) / "learning_curves",
                order,
            ),
        ]

    return build


def build_main_benchmark(paths: Paths, name: str) -> list[Path]:
    """The benchmark report plus the statistical comparison (figures F1-F5)."""
    written = benchmark_builder(MAIN_ORDER)(paths, name)
    runs = final_results(paths.experiment(name))
    summary = summarize_variants(runs)
    estimates = comparison.interval_estimates(runs, MAIN_ORDER)
    matrix = comparison.improvement_matrix(runs, MAIN_ORDER)
    out = paths.figures(name)
    return [
        *written,
        _write_csv(paths.summary / f"{name}_interval_estimates.csv", estimates),
        _write_csv(paths.summary / f"{name}_improvement.csv", matrix),
        _write(
            paths.tables / f"{name}_results.md",
            comparison.results_table(runs, estimates, summary, MAIN_ORDER),
        ),
        _write(
            paths.tables / f"{name}_headline.md",
            comparison.headline_table(estimates, summary, MAIN_ORDER),
        ),
        _write(
            paths.tables / f"{name}_improvement.md",
            comparison.improvement_table(matrix, MAIN_ORDER),
        ),
        *comparison.plot_iqm_curves(
            evaluation_curves(paths.experiment(name)), out / "iqm_curves", MAIN_ORDER
        ),
        *comparison.plot_aggregate_metrics(estimates, out / "aggregate_metrics", MAIN_ORDER),
        *comparison.plot_performance_profiles(
            episode_returns(paths.experiment(name)), out / "performance_profiles", MAIN_ORDER
        ),
        *comparison.plot_sample_efficiency(runs, out / "sample_efficiency", MAIN_ORDER),
        *comparison.plot_outcomes(summary, out / "outcomes", MAIN_ORDER),
        *comparison.plot_probability_of_improvement(
            matrix, out / "probability_of_improvement", MAIN_ORDER
        ),
    ]


def ablation_builder(
    order: Sequence[str], metric: tuple[str, str] | None = None
) -> Callable[[Paths, str], list[Path]]:
    def build(paths: Paths, name: str) -> list[Path]:
        runs, summary, written = _aggregate(paths, name)
        out = paths.figures(name)
        written += [
            *_benchmark_tables(paths, name, summary, order),
            *ablation.plot_final_returns(runs, summary, out / "final_returns", order),
            *benchmark.plot_learning_curves(
                evaluation_curves(paths.experiment(name)),
                out / "learning_curves",
                order,
                colors=ablation.variant_colors(list(order)),
            ),
        ]
        if metric is not None:
            column, label = metric
            curves = ablation.training_metric(paths.experiment(name), column, bin_steps=20_000)
            written += ablation.plot_training_metric(curves, column, label, out / column, order)
        return written

    return build


def resolve_spec(spec: RobustnessSpec, paths: Paths) -> RobustnessSpec:
    """Point a robustness spec at the experiment of the same name under ``paths.runs``."""
    return dataclasses.replace(spec, runs=paths.experiment(spec.runs.name))


def build_robustness(paths: Paths, config: str) -> list[Path]:
    spec = resolve_spec(load_spec(config), paths)
    summary = robustness_report.summarize(robustness_report.collect(spec))
    stem = f"robustness_{spec.name}"
    return [
        _write_csv(paths.summary / f"{stem}_summary.csv", summary),
        _write(paths.tables / f"{stem}.md", robustness_report.table(summary, MAIN_ORDER)),
        _write(
            paths.tables / f"{stem}_success.md",
            robustness_report.success_table(summary, MAIN_ORDER),
        ),
        *robustness_report.plot(summary, paths.figures(stem) / "robustness", MAIN_ORDER),
    ]


def build_media(paths: Paths, config: str) -> list[Path]:
    spec = media.load_showcase(config)
    return media.build_showcase(
        spec, paths.experiment(Path(spec.runs).name), paths.results / "gifs"
    )


# ------------------------------------------------------------------------ registry
@dataclass(frozen=True)
class Report:
    name: str
    config: str
    # sweep: trains runs; robustness: re-evaluates saved runs; media: only renders.
    kind: Literal["sweep", "robustness", "media"]
    build: Callable[[Paths, str], list[Path]]

    def run(self, paths: Paths) -> list[Path]:
        # A sweep's report reads runs/<name>; the others read their spec file.
        return self.build(paths, self.name if self.kind == "sweep" else self.config)


def _ablation_order(prefix: str, grid: dict[str, Sequence[str]]) -> tuple[str, ...]:
    """Grid labels in the order the sweep expands them."""
    labels = [prefix]
    for key, values in grid.items():
        labels = [f"{label}__{key}={value}" for label in labels for value in values]
    return tuple(labels)


REPORTS: tuple[Report, ...] = (
    Report("replication_v1", "configs/experiments/replication_v1.yaml", "sweep", build_replication),
    Report(
        "baseline_check",
        "configs/experiments/baseline_check.yaml",
        "sweep",
        benchmark_builder(("dqn", "sb3_dqn", "sb3_ppo")),
    ),
    Report("tuning", "configs/experiments/tuning.yaml", "sweep", build_tuning),
    Report(
        "main_benchmark", "configs/experiments/main_benchmark.yaml", "sweep", build_main_benchmark
    ),
    Report(
        "ablation_reinforce",
        "configs/experiments/ablation_reinforce.yaml",
        "sweep",
        ablation_builder(
            _ablation_order(
                "reinforce",
                {"baseline": ["none", "normalize", "value"], "entropy_coef": ["0", "0.01"]},
            ),
            ("entropy", "Policy entropy (nats)"),
        ),
    ),
    Report(
        "ablation_dqn",
        "configs/experiments/ablation_dqn.yaml",
        "sweep",
        ablation_builder(
            (
                "dqn__double=False__loss=huber",
                "dqn__double=True__loss=huber",
                "dqn__double=False__loss=mse",
                "dqn__double=True__loss=mse",
            ),
            ("q_mean", "Mean Q estimate (replay batches)"),
        ),
    ),
    Report(
        "ablation_dqn_v1",
        "configs/experiments/ablation_dqn_v1.yaml",
        "sweep",
        ablation_builder(
            _ablation_order("dqn_v1", {"gamma": ["0.99", "0.9"], "loss": ["huber", "mse"]})
        ),
    ),
    Report("wind", "configs/robustness/wind.yaml", "robustness", build_robustness),
    Report("showcase", "configs/media/showcase.yaml", "media", build_media),
)


def report_named(name: str) -> Report:
    for report in REPORTS:
        if report.name == name:
            return report
    raise KeyError(f"unknown report {name!r}; known: {[r.name for r in REPORTS]}")
