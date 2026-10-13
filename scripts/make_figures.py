"""Generate report figures from run directories.

Examples:
    python scripts/make_figures.py replication
    python scripts/make_figures.py benchmark runs/main_benchmark
    python scripts/make_figures.py ablation runs/ablation_dqn --metric q_mean "Mean Q estimate"
    python scripts/make_figures.py robustness configs/robustness/wind.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lunarlander_rl.analysis import ablation, benchmark, replication
from lunarlander_rl.analysis import robustness as robustness_report
from lunarlander_rl.analysis.aggregate import (
    evaluation_curves,
    final_results,
    summarize_variants,
    training_curves,
)
from lunarlander_rl.robustness import load_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="report", required=True)

    rep = sub.add_parser("replication", help="v1.0 replication study (Phase 3)")
    rep.add_argument("--runs", default="runs/replication_v1")
    rep.add_argument("--reference", default="results/reference/v1_reported.csv")
    rep.add_argument("--out-dir", default="results/figures/replication_v1")

    bench = sub.add_parser("benchmark", help="agents compared at an equal step budget")
    bench.add_argument("runs", help="experiment directory, e.g. runs/main_benchmark")
    bench.add_argument("--out-dir", default=None, help="default: results/figures/<experiment>")
    bench.add_argument("--order", nargs="*", default=None, help="variant order in the legend")

    abl = sub.add_parser("ablation", help="variants of one method")
    abl.add_argument("runs", help="experiment directory, e.g. runs/ablation_dqn")
    abl.add_argument("--out-dir", default=None, help="default: results/figures/<experiment>")
    abl.add_argument("--order", nargs="*", default=None, help="variant order")
    abl.add_argument(
        "--metric", nargs=2, default=None, metavar=("COLUMN", "LABEL"),
        help="also plot a training diagnostic, e.g. q_mean 'Mean Q estimate'",
    )  # fmt: skip

    rob = sub.add_parser("robustness", help="saved policies re-evaluated under new conditions")
    rob.add_argument("spec", help="robustness spec, e.g. configs/robustness/wind.yaml")
    rob.add_argument("--out-dir", default=None, help="default: results/figures/<spec name>")
    rob.add_argument("--order", nargs="*", default=None, help="variant order in the legend")
    return parser.parse_args()


def make_replication(args: argparse.Namespace) -> list[Path]:
    runs = final_results(args.runs)
    rows = replication.comparison(summarize_variants(runs), pd.read_csv(args.reference))
    out = Path(args.out_dir)
    return [
        *replication.plot_learning_curves(training_curves(args.runs), out / "learning_curves"),
        *replication.plot_final_comparison(rows, runs, out / "final_performance"),
    ]


def make_benchmark(args: argparse.Namespace) -> list[Path]:
    out = Path(args.out_dir or Path("results/figures") / Path(args.runs).name)
    return benchmark.plot_learning_curves(
        evaluation_curves(args.runs), out / "learning_curves", args.order
    )


def make_ablation(args: argparse.Namespace) -> list[Path]:
    out = Path(args.out_dir or Path("results/figures") / Path(args.runs).name)
    runs = final_results(args.runs)
    summary = summarize_variants(runs)
    variants = benchmark.ordered(summary, args.order)["variant"].tolist()
    written = [
        *ablation.plot_final_returns(runs, summary, out / "final_returns", args.order),
        *benchmark.plot_learning_curves(
            evaluation_curves(args.runs),
            out / "learning_curves",
            args.order,
            colors=ablation.variant_colors(variants),
        ),
    ]
    if args.metric:
        column, label = args.metric
        curves = ablation.training_metric(args.runs, column, bin_steps=20_000)
        written += ablation.plot_training_metric(curves, column, label, out / column, args.order)
    return written


def make_robustness(args: argparse.Namespace) -> list[Path]:
    spec = load_spec(args.spec)
    out = Path(args.out_dir or Path("results/figures") / f"robustness_{spec.name}")
    summary = robustness_report.summarize(robustness_report.collect(spec))
    return robustness_report.plot(summary, out / "robustness", args.order)


def main() -> None:
    args = parse_args()
    make = {
        "replication": make_replication,
        "benchmark": make_benchmark,
        "ablation": make_ablation,
        "robustness": make_robustness,
    }[args.report]
    for path in make(args):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
