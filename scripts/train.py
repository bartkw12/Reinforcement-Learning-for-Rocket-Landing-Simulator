"""Train one agent for one seed, or re-run a saved run and check it reproduces exactly.

Examples:
    python scripts/train.py --agent configs/agent/random.yaml --seed 0 \
        --set train.total_steps=20000 --set eval.interval_steps=5000

    # re-run a finished run from its own config.yaml and compare the results byte for byte
    python scripts/train.py --from-run runs/main_benchmark/q_learning/seed_100 --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lunarlander_rl.config import build_experiment_config, load_config
from lunarlander_rl.tracking import RunPaths, compare_runs
from lunarlander_rl.training import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--agent", help="agent YAML file")
    source.add_argument("--from-run", help="run directory whose config.yaml to train again")
    parser.add_argument("--env", default="configs/env/lunarlander.yaml", help="environment YAML")
    parser.add_argument("--name", default="adhoc", help="experiment name")
    parser.add_argument("--label", default=None, help="variant label (default: agent file name)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY.PATH=VALUE",
        help="override a config value, e.g. train.total_steps=50000 (repeatable)",
    )
    parser.add_argument(
        "--runs-root", default=None, help="default: runs (runs_verify with --from-run)"
    )
    parser.add_argument("--overwrite", action="store_true", help="replace a completed run")
    parser.add_argument(
        "--check", action="store_true",
        help="with --from-run: fail unless the results match the original run exactly",
    )  # fmt: skip
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.from_run:
        original = RunPaths(Path(args.from_run))
        config = load_config(original.config)
        runs_root = args.runs_root or "runs_verify"
    else:
        if args.check:
            raise SystemExit("--check needs --from-run")
        config = build_experiment_config(
            {
                "name": args.name,
                "label": args.label or Path(args.agent).stem,
                "seed": args.seed,
                "env": args.env,
                "agent": args.agent,
            },
            args.overrides,
        )
        runs_root = args.runs_root or "runs"
    run_dir = config.run_dir(runs_root)
    print(f"Training {config.variant} (seed {config.seed}) -> {run_dir}")
    summary = run_experiment(config, run_dir, overwrite=args.overwrite or bool(args.from_run))
    print(
        f"Final evaluation over {summary['episodes']} episodes: "
        f"mean return {summary['mean_return']:.2f} +/- {summary['std_return']:.2f}, "
        f"IQM {summary['iqm_return']:.2f}, success rate {summary['success_rate']:.1%}"
    )
    if args.check:
        matches = compare_runs(original.root, run_dir)
        for name, same in matches.items():
            print(f"  {name}: {'identical' if same else 'DIFFERENT'}")
        if not all(matches.values()):
            print("The re-run does not reproduce the original run.", file=sys.stderr)
            return 1
        print("The re-run reproduces the original run exactly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
