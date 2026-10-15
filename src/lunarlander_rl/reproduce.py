"""Re-run every experiment and rebuild every report, in dependency order.

The quick mode keeps the full pipeline (every experiment, report and figure) but shrinks
each run to two seeds, a few thousand steps and a handful of test episodes. It is a smoke
test of the pipeline, not a result: it writes to separate directories so it can never
overwrite the real ``runs/`` and ``results/``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from lunarlander_rl.config import ExperimentConfig, load_yaml
from lunarlander_rl.reports import REPORTS, Paths, Report, resolve_spec
from lunarlander_rl.robustness import load_spec, run_robustness
from lunarlander_rl.sweep import RunSpec, expand_experiment, run_sweep

FULL_PATHS = Paths()
QUICK_PATHS = Paths(runs=Path("runs_quick"), results=Path("results_quick"))
QUICK_SEEDS = 2
QUICK_STEPS = 4_000
QUICK_EPISODES = 4


def quick_config(config: ExperimentConfig) -> ExperimentConfig:
    """A smoke-test version of a run: same agent and protocol, tiny budgets."""
    train = config.train
    if train.total_steps is None:
        train = dataclasses.replace(train, total_episodes=QUICK_EPISODES)
    else:
        train = dataclasses.replace(train, total_steps=QUICK_STEPS, total_episodes=None)
    evaluation = dataclasses.replace(
        config.eval, interval_steps=QUICK_STEPS // 2, episodes=2, final_episodes=5
    )
    return dataclasses.replace(config, train=train, eval=evaluation)


def experiment_runs(report: Report, paths: Paths, quick: bool) -> list[RunSpec]:
    spec = load_yaml(report.config)
    runs = expand_experiment(spec, paths.runs)
    if not quick:
        return runs
    kept = set(spec["seeds"][:QUICK_SEEDS])
    return [
        RunSpec(config=quick_config(run.config), run_dir=run.run_dir)
        for run in runs
        if run.config.seed in kept
    ]


@dataclass(frozen=True)
class StepResult:
    report: str
    stage: str  # "runs", "evaluations" (robustness studies) or "report"
    completed: int = 0
    skipped: int = 0
    failed: int = 0


def reproduce(
    *,
    quick: bool = False,
    workers: int = 1,
    only: Sequence[str] | None = None,
    skip_runs: bool = False,
    on_step: Callable[[StepResult], None] | None = None,
) -> list[StepResult]:
    """Run the selected experiments (all by default), then build their reports.

    Completed runs are skipped, so an interrupted reproduction can be restarted.
    """
    paths = QUICK_PATHS if quick else FULL_PATHS
    selected = [r for r in REPORTS if only is None or r.name in only]
    unknown = set(only or ()) - {r.name for r in REPORTS}
    if unknown:
        raise KeyError(f"unknown experiments {sorted(unknown)}")
    results: list[StepResult] = []

    def record(result: StepResult) -> None:
        results.append(result)
        if on_step is not None:
            on_step(result)

    for report in selected:
        # Media only renders saved policies, so it has no runs to execute.
        if not skip_runs and report.kind != "media":
            if report.kind == "sweep":
                outcomes = run_sweep(experiment_runs(report, paths, quick), max_workers=workers)
                statuses = [o.status for o in outcomes]
            else:
                spec = resolve_spec(load_spec(report.config), paths)
                statuses = [o.status for o in run_robustness(spec, max_workers=workers)]
            record(
                StepResult(
                    report.name,
                    "runs" if report.kind == "sweep" else "evaluations",
                    completed=statuses.count("completed"),
                    skipped=statuses.count("skipped"),
                    failed=statuses.count("failed"),
                )
            )
        written = report.run(paths)
        record(StepResult(report.name, "report", completed=len(written)))
    return results
