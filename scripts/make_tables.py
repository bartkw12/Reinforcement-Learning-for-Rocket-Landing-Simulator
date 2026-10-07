"""Generate report tables (Markdown) from run directories.

Examples:
    python scripts/make_tables.py replication
    python scripts/make_tables.py benchmark runs/baseline_check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lunarlander_rl.analysis import benchmark, replication
from lunarlander_rl.analysis.aggregate import final_results, summarize_variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="report", required=True)

    rep = sub.add_parser("replication", help="v1.0 replication study (Phase 3)")
    rep.add_argument("--runs", default="runs/replication_v1")
    rep.add_argument("--reference", default="results/reference/v1_reported.csv")
    rep.add_argument("--out", default="results/tables/replication_v1.md")

    bench = sub.add_parser("benchmark", help="agents compared at an equal step budget")
    bench.add_argument("runs", help="experiment directory, e.g. runs/baseline_check")
    bench.add_argument("--out", default=None, help="default: results/tables/<experiment>.md")
    bench.add_argument("--order", nargs="*", default=None, help="variant order of the rows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_variants(final_results(args.runs))
    if args.report == "replication":
        text = replication.table(replication.comparison(summary, pd.read_csv(args.reference)))
        out = Path(args.out)
    else:
        text = benchmark.table(summary, args.order)
        out = Path(args.out or Path("results/tables") / f"{Path(args.runs).name}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
