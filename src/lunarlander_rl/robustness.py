"""Re-evaluate the saved policies of an experiment under changed conditions.

Nothing is retrained: every completed run's checkpoint is evaluated again on the run's own
held-out test episodes, once per condition, with configuration overrides applied to the
environment (for example wind). A specification file looks like::

    name: wind
    runs: runs/main_benchmark        # the experiment whose policies are re-evaluated
    checkpoint: final
    conditions:
      - label: calm                  # no overrides: reproduces each run's final evaluation
        level: 0
        overrides: {}
      - label: wind_15
        level: 15
        overrides: {env.kwargs.enable_wind: true, env.kwargs.wind_power: 15.0}

Results go to ``<run>/evaluations/<name>/<label>.json``; existing results are skipped.
"""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lunarlander_rl.analysis.aggregate import completed_runs
from lunarlander_rl.config import ConfigError, load_yaml
from lunarlander_rl.evaluation import evaluate_checkpoint
from lunarlander_rl.tracking import RunPaths, write_json

_SPEC_KEYS = {"name", "runs", "checkpoint", "conditions"}
_CONDITION_KEYS = {"label", "level", "overrides"}


@dataclass(frozen=True)
class Condition:
    label: str
    # Position of the condition on a plot's x-axis (wind power, say).
    level: float
    overrides: tuple[str, ...]


@dataclass(frozen=True)
class RobustnessSpec:
    name: str
    runs: Path
    checkpoint: str
    conditions: tuple[Condition, ...]

    def result_path(self, run_dir: Path, condition: Condition) -> Path:
        return run_dir / "evaluations" / self.name / f"{condition.label}.json"


def _as_override(key: str, value: Any) -> str:
    # JSON is valid YAML, so the value round-trips through the override parser unchanged.
    return f"{key}={json.dumps(value)}"


def parse_spec(data: Mapping[str, Any]) -> RobustnessSpec:
    unknown = sorted(set(data) - _SPEC_KEYS)
    if unknown:
        raise ConfigError(f"robustness spec: unknown key(s) {unknown}")
    for key in ("name", "runs", "conditions"):
        if key not in data:
            raise ConfigError(f"robustness spec: missing required key {key!r}")
    checkpoint = data.get("checkpoint", "final")
    if checkpoint not in ("final", "best"):
        raise ConfigError("robustness spec: checkpoint must be 'final' or 'best'")
    conditions = []
    for index, raw in enumerate(data["conditions"] or []):
        where = f"robustness spec: conditions[{index}]"
        if not isinstance(raw, Mapping) or set(raw) - _CONDITION_KEYS or "label" not in raw:
            raise ConfigError(f"{where}: needs a label, and optionally level and overrides")
        overrides = raw.get("overrides") or {}
        if not isinstance(overrides, Mapping):
            raise ConfigError(f"{where}.overrides: expected a mapping of key.path to value")
        conditions.append(
            Condition(
                label=str(raw["label"]),
                level=float(raw.get("level", index)),
                overrides=tuple(_as_override(str(k), v) for k, v in overrides.items()),
            )
        )
    labels = [c.label for c in conditions]
    if not conditions or len(set(labels)) != len(labels):
        raise ConfigError("robustness spec: conditions must be a non-empty list of unique labels")
    return RobustnessSpec(
        name=str(data["name"]),
        runs=Path(data["runs"]),
        checkpoint=checkpoint,
        conditions=tuple(conditions),
    )


def load_spec(path: str | Path) -> RobustnessSpec:
    return parse_spec(load_yaml(path))


@dataclass(frozen=True)
class EvaluationOutcome:
    run_dir: Path
    condition: str
    status: str  # "completed", "skipped" or "failed"
    seconds: float = 0.0
    error: str | None = None


def _evaluate_one(spec: RobustnessSpec, run_dir: Path, condition: Condition) -> EvaluationOutcome:
    """Top-level so that it can be sent to a worker process."""
    started = time.perf_counter()
    try:
        result = evaluate_checkpoint(
            run_dir, checkpoint=spec.checkpoint, overrides=condition.overrides
        )
        path = spec.result_path(run_dir, condition)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            path,
            {
                "condition": condition.label,
                "level": condition.level,
                "overrides": list(condition.overrides),
                "checkpoint": spec.checkpoint,
                **result,
            },
        )
    except Exception:
        return EvaluationOutcome(
            run_dir,
            condition.label,
            "failed",
            time.perf_counter() - started,
            traceback.format_exc(),
        )
    return EvaluationOutcome(run_dir, condition.label, "completed", time.perf_counter() - started)


def run_robustness(
    spec: RobustnessSpec,
    *,
    max_workers: int = 1,
    overwrite: bool = False,
    on_result: Callable[[EvaluationOutcome], None] | None = None,
) -> list[EvaluationOutcome]:
    """Evaluate every completed run of ``spec.runs`` under every condition."""
    outcomes: list[EvaluationOutcome] = []

    def record(outcome: EvaluationOutcome) -> None:
        outcomes.append(outcome)
        if on_result is not None:
            on_result(outcome)

    pending: list[tuple[Path, Condition]] = []
    for paths in completed_runs(spec.runs):
        for condition in spec.conditions:
            if not overwrite and spec.result_path(paths.root, condition).is_file():
                record(EvaluationOutcome(paths.root, condition.label, "skipped"))
            else:
                pending.append((paths.root, condition))

    if max_workers <= 1 or len(pending) <= 1:
        for run_dir, condition in pending:
            record(_evaluate_one(spec, run_dir, condition))
    else:
        with ProcessPoolExecutor(max_workers=min(max_workers, len(pending))) as pool:
            futures = [pool.submit(_evaluate_one, spec, r, c) for r, c in pending]
            for future in as_completed(futures):
                record(future.result())
    return outcomes


def result_paths(spec: RobustnessSpec) -> Sequence[tuple[RunPaths, Condition, Path]]:
    """Every expected result file, whether or not it exists yet."""
    return [
        (paths, condition, spec.result_path(paths.root, condition))
        for paths in completed_runs(spec.runs)
        for condition in spec.conditions
    ]
