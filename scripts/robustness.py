"""Re-evaluate an experiment's saved policies under the conditions of a robustness spec.

Example:
    python scripts/robustness.py configs/robustness/wind.yaml --workers 16
"""

from __future__ import annotations

import argparse
import os
import sys

from lunarlander_rl.robustness import EvaluationOutcome, load_spec, run_robustness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("spec", help="robustness spec YAML file")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--overwrite", action="store_true", help="redo existing evaluations")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = load_spec(args.spec)
    done = 0

    def show(outcome: EvaluationOutcome) -> None:
        nonlocal done
        done += 1
        if outcome.status != "skipped":
            run = f"{outcome.run_dir.parent.name}/{outcome.run_dir.name}"
            print(
                f"[{done}] {run} {outcome.condition}: {outcome.status} in {outcome.seconds:.0f}s",
                flush=True,
            )
        if outcome.error:
            print(outcome.error, file=sys.stderr, flush=True)

    outcomes = run_robustness(
        spec, max_workers=args.workers, overwrite=args.overwrite, on_result=show
    )
    failed = sum(o.status == "failed" for o in outcomes)
    skipped = sum(o.status == "skipped" for o in outcomes)
    print(f"{len(outcomes)} evaluations: {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
