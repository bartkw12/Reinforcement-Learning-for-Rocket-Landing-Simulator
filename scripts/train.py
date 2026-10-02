"""Train one agent for one seed.

Example:
    python scripts/train.py --agent configs/agent/random.yaml --seed 0 \
        --set train.total_steps=20000 --set eval.interval_steps=5000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lunarlander_rl.config import build_experiment_config
from lunarlander_rl.training import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--agent", required=True, help="agent YAML file")
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
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--overwrite", action="store_true", help="replace a completed run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    run_dir = config.run_dir(args.runs_root)
    print(f"Training {config.variant} (seed {config.seed}) -> {run_dir}")
    summary = run_experiment(config, run_dir, overwrite=args.overwrite)
    print(
        f"Final evaluation over {summary['episodes']} episodes: "
        f"mean return {summary['mean_return']:.2f} +/- {summary['std_return']:.2f}, "
        f"IQM {summary['iqm_return']:.2f}, success rate {summary['success_rate']:.1%}"
    )


if __name__ == "__main__":
    main()
