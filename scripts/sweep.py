"""Run every (variant, seed) combination of an experiment file, in parallel.

Completed runs are skipped, so an interrupted sweep can simply be started again.

Example:
    python scripts/sweep.py configs/experiments/smoke.yaml --workers 4
"""

from __future__ import annotations

import argparse
import os
import sys

from lunarlander_rl.sweep import RunOutcome, load_experiment, run_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("experiment", help="experiment YAML file")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="parallel runs (default: CPU count minus one)",
    )
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--overwrite", action="store_true", help="rerun completed runs")
    parser.add_argument("--dry-run", action="store_true", help="list the runs and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs = load_experiment(args.experiment, args.runs_root)
    if args.dry_run:
        for run in runs:
            print(run.run_dir)
        print(f"{len(runs)} runs")
        return 0

    finished = 0

    def show(outcome: RunOutcome) -> None:
        nonlocal finished
        finished += 1
        line = f"[{finished}/{len(runs)}] {outcome.spec.tag}: {outcome.status}"
        if outcome.summary is not None:
            line += (
                f" in {outcome.seconds:.0f}s, mean return {outcome.summary['mean_return']:.2f}, "
                f"success rate {outcome.summary['success_rate']:.1%}"
            )
        print(line, flush=True)
        if outcome.error is not None:
            print(outcome.error, file=sys.stderr, flush=True)

    print(f"{len(runs)} runs, {args.workers} workers")
    outcomes = run_sweep(runs, max_workers=args.workers, overwrite=args.overwrite, on_result=show)
    failed = sum(outcome.status == "failed" for outcome in outcomes)
    if failed:
        print(f"{failed} run(s) failed", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
