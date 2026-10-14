"""Reproduce every experiment, table and figure of the project with one command.

Experiments run in dependency order (the wind robustness study re-evaluates the main
benchmark's policies); completed runs are skipped, so the command can be restarted. The full
reproduction takes roughly 8-10 hours on a 12-core desktop CPU with 20 workers.

Examples:
    python scripts/reproduce_all.py --workers 20             # everything, then sync the docs
    python scripts/reproduce_all.py --quick --workers 4      # smoke test, a few minutes
    python scripts/reproduce_all.py --only main_benchmark wind
    python scripts/reproduce_all.py --skip-runs              # rebuild reports from runs/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from lunarlander_rl.docs_sync import default_documents, sync
from lunarlander_rl.reports import REPORTS
from lunarlander_rl.reproduce import QUICK_PATHS, StepResult, reproduce

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="tiny budgets, separate outputs")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument(
        "--only", nargs="*", default=None, choices=[r.name for r in REPORTS],
        help="run only these experiments",
    )  # fmt: skip
    parser.add_argument("--skip-runs", action="store_true", help="only rebuild the reports")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.quick:
        print(f"Quick mode: writing to {QUICK_PATHS.runs}/ and {QUICK_PATHS.results}/")
    started = time.perf_counter()

    def show(step: StepResult) -> None:
        elapsed = time.perf_counter() - started
        if step.stage != "report":
            print(
                f"[{elapsed / 60:5.1f} min] {step.report}: {step.completed} {step.stage} "
                f"completed, {step.skipped} skipped, {step.failed} failed",
                flush=True,
            )
        else:
            print(f"[{elapsed / 60:5.1f} min] {step.report}: report written "
                  f"({step.completed} files)", flush=True)  # fmt: skip

    results = reproduce(
        quick=args.quick,
        workers=args.workers,
        only=args.only,
        skip_runs=args.skip_runs,
        on_step=show,
    )
    if not args.quick:
        for document in sync(default_documents(ROOT), ROOT):
            print(f"updated {document.relative_to(ROOT)}")
    failed = sum(step.failed for step in results)
    if failed:
        print(f"{failed} run(s) failed; see the messages above", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
