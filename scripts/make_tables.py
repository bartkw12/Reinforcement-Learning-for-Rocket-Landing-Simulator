"""Generate report tables (Markdown) from run directories.

Example:
    python scripts/make_tables.py replication
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lunarlander_rl.analysis import replication
from lunarlander_rl.analysis.aggregate import final_results, summarize_variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="report", required=True)
    rep = sub.add_parser("replication", help="v1.0 replication study (Phase 3)")
    rep.add_argument("--runs", default="runs/replication_v1")
    rep.add_argument("--reference", default="results/reference/v1_reported.csv")
    rep.add_argument("--out", default="results/tables/replication_v1.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = replication.comparison(
        summarize_variants(final_results(args.runs)), pd.read_csv(args.reference)
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(replication.table(rows), encoding="utf-8", newline="\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
