"""Record animated demonstrations of trained policies.

Examples:
    # every animation of the README showcase (configs/media/showcase.yaml) -> results/gifs/
    python scripts/record_gif.py showcase

    # one episode of one run, optionally under changed conditions
    python scripts/record_gif.py episode runs/main_benchmark/dqn/seed_100 --seed 10003
    python scripts/record_gif.py episode runs/main_benchmark/dqn/seed_100 \
        --set env.kwargs.enable_wind=true --out wind_demo.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lunarlander_rl import media
from lunarlander_rl.reports import Paths, report_named


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("showcase", help="all README animations")
    show.add_argument("--runs-root", default="runs")
    show.add_argument("--results-root", default="results")

    one = sub.add_parser("episode", help="one episode of one run")
    one.add_argument("run_dir")
    one.add_argument("--seed", type=int, default=10_000, help="episode seed")
    one.add_argument("--checkpoint", choices=("final", "best"), default="final")
    one.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY.PATH=VALUE",
        help="override the stored config (repeatable)",
    )
    one.add_argument("--scale", type=float, default=0.75)
    one.add_argument("--out", default=None, help="default: <run_dir>/episode_<seed>.gif")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "showcase":
        paths = Paths(runs=Path(args.runs_root), results=Path(args.results_root))
        for path in report_named("showcase").run(paths):
            print(f"Wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
        return
    recording = media.record_episode(
        args.run_dir, args.seed, checkpoint=args.checkpoint, overrides=args.overrides
    )
    frames = media.animate([recording], columns=1, scale=args.scale, frame_step=2,
                           hold_frames=38)  # fmt: skip
    out = Path(args.out or Path(args.run_dir) / f"episode_{args.seed}.gif")
    media.write_gif(frames, out, fps=25)
    print(f"{recording.label}, episode {args.seed}: {recording.outcome}, "
          f"return {recording.episode_return:.1f}, {recording.length} steps -> {out}")  # fmt: skip


if __name__ == "__main__":
    main()
