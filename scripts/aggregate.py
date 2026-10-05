"""Collect an experiment's completed runs into per-run and per-variant CSV files.

Example:
    python scripts/aggregate.py runs/replication_v1
    # -> results/summary/replication_v1_runs.csv, results/summary/replication_v1_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lunarlander_rl.analysis.aggregate import final_results, summarize_variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("experiment_dir", help="e.g. runs/replication_v1")
    parser.add_argument("--out-dir", default="results/summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = Path(args.experiment_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = final_results(experiment)
    summary = summarize_variants(runs)
    runs_csv = out_dir / f"{experiment.name}_runs.csv"
    summary_csv = out_dir / f"{experiment.name}_summary.csv"
    runs.to_csv(runs_csv, index=False, lineterminator="\n")
    summary.to_csv(summary_csv, index=False, lineterminator="\n")
    print(f"{len(runs)} runs, {len(summary)} variants")
    print(f"Wrote {runs_csv} and {summary_csv}")


if __name__ == "__main__":
    main()
