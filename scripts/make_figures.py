"""Generate report figures from run directories.

Example:
    python scripts/make_figures.py replication
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lunarlander_rl.analysis import replication
from lunarlander_rl.analysis.aggregate import final_results, summarize_variants, training_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="report", required=True)
    rep = sub.add_parser("replication", help="v1.0 replication study (Phase 3)")
    rep.add_argument("--runs", default="runs/replication_v1")
    rep.add_argument("--reference", default="results/reference/v1_reported.csv")
    rep.add_argument("--out-dir", default="results/figures/replication_v1")
    return parser.parse_args()


def make_replication(args: argparse.Namespace) -> list[Path]:
    runs = final_results(args.runs)
    rows = replication.comparison(summarize_variants(runs), pd.read_csv(args.reference))
    out = Path(args.out_dir)
    return [
        *replication.plot_learning_curves(training_curves(args.runs), out / "learning_curves"),
        *replication.plot_final_comparison(rows, runs, out / "final_performance"),
    ]


def main() -> None:
    args = parse_args()
    written = make_replication(args)
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
