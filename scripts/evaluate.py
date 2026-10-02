"""Re-evaluate a saved policy from a finished run.

Examples:
    python scripts/evaluate.py runs/adhoc/random/seed_0
    python scripts/evaluate.py runs/adhoc/random/seed_0 --checkpoint best \
        --set env.kwargs.enable_wind=true --output wind_eval.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lunarlander_rl.evaluation import evaluate_checkpoint
from lunarlander_rl.tracking import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("run_dir", help="run directory containing config.yaml and checkpoints/")
    parser.add_argument("--checkpoint", choices=("final", "best"), default="final")
    parser.add_argument("--episodes", type=int, default=None, help="default: the run's setting")
    parser.add_argument("--seed-start", type=int, default=None, help="first episode seed")
    parser.add_argument("--stochastic", action="store_true", help="sample actions from the policy")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY.PATH=VALUE",
        help="override the stored config, e.g. env.kwargs.enable_wind=true (repeatable)",
    )
    parser.add_argument("--output", default=None, help="write the full results to this JSON file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = list(args.overrides)
    if args.episodes is not None:
        overrides.append(f"eval.final_episodes={args.episodes}")
    if args.seed_start is not None:
        overrides.append(f"eval.final_seed_start={args.seed_start}")

    result = evaluate_checkpoint(
        args.run_dir,
        checkpoint=args.checkpoint,
        deterministic=False if args.stochastic else None,
        overrides=overrides,
    )
    summary = result["summary"]
    print(f"{args.checkpoint} checkpoint of {args.run_dir}, {summary['episodes']} episodes")
    print(f"  mean return   {summary['mean_return']:.2f} +/- {summary['std_return']:.2f}")
    print(f"  IQM return    {summary['iqm_return']:.2f}")
    print(
        f"  success rate  {summary['success_rate']:.1%} "
        f"(95% CI {summary['success_rate_ci_low']:.1%} to {summary['success_rate_ci_high']:.1%})"
    )
    print(f"  outcomes      {summary['outcomes']}")
    if args.output is not None:
        write_json(Path(args.output), result)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
