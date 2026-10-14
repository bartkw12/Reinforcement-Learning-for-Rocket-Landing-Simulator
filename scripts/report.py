"""Build the tables and figures of one experiment, or of all of them, from finished runs.

Examples:
    python scripts/report.py main_benchmark
    python scripts/report.py all
    python scripts/report.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lunarlander_rl.reports import REPORTS, Paths, report_named


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("names", nargs="*", help="report names, or 'all'")
    parser.add_argument("--list", action="store_true", help="list the reports and exit")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--results-root", default="results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list or not args.names:
        for report in REPORTS:
            print(f"{report.name:20s} {report.config}")
        return 0
    reports = list(REPORTS) if args.names == ["all"] else [report_named(n) for n in args.names]
    paths = Paths(runs=Path(args.runs_root), results=Path(args.results_root))
    for report in reports:
        written = report.run(paths)
        print(f"{report.name}: wrote {len(written)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
