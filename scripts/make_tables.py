"""Generate report tables (Markdown) from run directories.

Examples:
    python scripts/make_tables.py replication
    python scripts/make_tables.py benchmark runs/main_benchmark
    python scripts/make_tables.py tuning
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lunarlander_rl.analysis import benchmark, replication, tuning
from lunarlander_rl.analysis.aggregate import final_results, summarize_variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="report", required=True)

    rep = sub.add_parser("replication", help="v1.0 replication study (Phase 3)")
    rep.add_argument("--runs", default="runs/replication_v1")
    rep.add_argument("--reference", default="results/reference/v1_reported.csv")
    rep.add_argument("--out", default="results/tables/replication_v1.md")

    bench = sub.add_parser("benchmark", help="agents compared at an equal step budget")
    bench.add_argument("runs", help="experiment directory, e.g. runs/main_benchmark")
    bench.add_argument("--out", default=None, help="default: results/tables/<experiment>.md")
    bench.add_argument("--order", nargs="*", default=None, help="variant order of the rows")

    tune = sub.add_parser("tuning", help="hyperparameter candidates and the selection")
    tune.add_argument("--runs", default="runs/tuning")
    tune.add_argument("--out", default="results/tables/tuning.md")
    tune.add_argument("--selection", default="results/summary/tuning_selection.csv")
    return parser.parse_args()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote {path}")


def main() -> None:
    args = parse_args()
    summary = summarize_variants(final_results(args.runs))
    if args.report == "replication":
        rows = replication.comparison(summary, pd.read_csv(args.reference))
        write(Path(args.out), replication.table(rows))
    elif args.report == "benchmark":
        out = Path(args.out or Path("results/tables") / f"{Path(args.runs).name}.md")
        write(out, benchmark.table(summary, args.order))
        write(
            out.with_name(out.stem + "_outcomes.md"), benchmark.outcome_table(summary, args.order)
        )
    else:
        write(Path(args.out), tuning.table(summary))
        selected = tuning.selection(summary)
        Path(args.selection).parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(args.selection, index=False, lineterminator="\n")
        print(f"Wrote {args.selection}")
        for _, row in selected.iterrows():
            print(f"  {row['family']}: {row['variant']} ({row['mean_return']:.1f})")


if __name__ == "__main__":
    main()
