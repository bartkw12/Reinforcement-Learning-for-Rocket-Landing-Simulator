"""Generate report tables (Markdown) from run directories.

Examples:
    python scripts/make_tables.py replication
    python scripts/make_tables.py benchmark runs/main_benchmark
    python scripts/make_tables.py tuning
    python scripts/make_tables.py robustness configs/robustness/wind.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lunarlander_rl.analysis import benchmark, replication, tuning
from lunarlander_rl.analysis import robustness as robustness_report
from lunarlander_rl.analysis.aggregate import final_results, summarize_variants
from lunarlander_rl.robustness import load_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="report", required=True)

    rep = sub.add_parser("replication", help="v1.0 replication study (Phase 3)")
    rep.add_argument("--runs", default="runs/replication_v1")
    rep.add_argument("--reference", default="results/reference/v1_reported.csv")
    rep.add_argument("--out", default="results/tables/replication_v1.md")

    bench = sub.add_parser("benchmark", help="agents or ablation variants side by side")
    bench.add_argument("runs", help="experiment directory, e.g. runs/main_benchmark")
    bench.add_argument("--out", default=None, help="default: results/tables/<experiment>.md")
    bench.add_argument("--order", nargs="*", default=None, help="variant order of the rows")

    tune = sub.add_parser("tuning", help="hyperparameter candidates and the selection")
    tune.add_argument("--runs", default="runs/tuning")
    tune.add_argument("--out", default="results/tables/tuning.md")
    tune.add_argument("--selection", default="results/summary/tuning_selection.csv")

    rob = sub.add_parser("robustness", help="saved policies re-evaluated under new conditions")
    rob.add_argument("spec", help="robustness spec, e.g. configs/robustness/wind.yaml")
    rob.add_argument("--out", default=None, help="default: results/tables/robustness_<name>.md")
    rob.add_argument("--order", nargs="*", default=None, help="variant order of the rows")
    return parser.parse_args()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote {path}")


def make_robustness(args: argparse.Namespace) -> None:
    spec = load_spec(args.spec)
    results = robustness_report.collect(spec)
    summary = robustness_report.summarize(results)
    out = Path(args.out or Path("results/tables") / f"robustness_{spec.name}.md")
    write(out, robustness_report.table(summary, args.order))
    write(
        out.with_name(out.stem + "_success.md"),
        robustness_report.success_table(summary, args.order),
    )
    summary_csv = Path("results/summary") / f"robustness_{spec.name}_summary.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False, lineterminator="\n")
    print(f"Wrote {summary_csv}")


def main() -> None:
    args = parse_args()
    if args.report == "robustness":
        make_robustness(args)
        return
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
