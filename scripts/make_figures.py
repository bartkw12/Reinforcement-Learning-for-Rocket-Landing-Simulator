"""Generate report figures from run directories.

Examples:
    python scripts/make_figures.py replication
    python scripts/make_figures.py benchmark runs/baseline_check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lunarlander_rl.analysis import benchmark, replication
from lunarlander_rl.analysis.aggregate import (
    evaluation_curves,
    final_results,
    summarize_variants,
    training_curves,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="report", required=True)

    rep = sub.add_parser("replication", help="v1.0 replication study (Phase 3)")
    rep.add_argument("--runs", default="runs/replication_v1")
    rep.add_argument("--reference", default="results/reference/v1_reported.csv")
    rep.add_argument("--out-dir", default="results/figures/replication_v1")

    bench = sub.add_parser("benchmark", help="agents compared at an equal step budget")
    bench.add_argument("runs", help="experiment directory, e.g. runs/baseline_check")
    bench.add_argument("--out-dir", default=None, help="default: results/figures/<experiment>")
    bench.add_argument("--order", nargs="*", default=None, help="variant order in the legend")
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


def main() -> None:
    args = parse_args()
    written = make_replication(args) if args.report == "replication" else make_benchmark(args)
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
