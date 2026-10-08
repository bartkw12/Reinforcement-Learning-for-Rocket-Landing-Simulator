"""Expand an experiment file into runs (variants x seeds) and execute them in parallel.

An experiment file looks like::

    name: main_benchmark
    seeds: [100, 101, 102]
    base:                      # shared by every run; any ExperimentConfig section
      env: configs/env/lunarlander.yaml
      train: {total_steps: 1000000}
    variants:
      - label: dqn
        agent: configs/agent/dqn.yaml
      - label: dqn_double
        agent: configs/agent/dqn.yaml
        overrides: {agent.params.double: true}
      - label: q_learning      # a grid: one variant per combination of the listed values
        agent: configs/agent/q_learning.yaml
        grid:
          agent.params.alpha: [0.1, 0.3]
          agent.params.tiles_per_dim: [4, 6]

A grid variant expands into variants labelled ``q_learning__alpha=0.1__tiles_per_dim=4`` and
so on (the last component of each key and its value). File paths inside an experiment file
are relative to the working directory.
"""

from __future__ import annotations

import itertools
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lunarlander_rl.config import (
    ConfigError,
    ExperimentConfig,
    build_experiment_config,
    load_yaml,
    resolve_section,
    set_by_path,
)
from lunarlander_rl.tracking import RunPaths
from lunarlander_rl.training import run_experiment

_SPEC_KEYS = {"name", "seeds", "base", "variants"}
_VARIANT_KEYS = {"label", "agent", "overrides", "grid"}
_RESERVED_BASE_KEYS = {"name", "label", "seed", "agent"}


@dataclass(frozen=True)
class RunSpec:
    config: ExperimentConfig
    run_dir: Path

    @property
    def tag(self) -> str:
        return f"{self.config.variant}/seed_{self.config.seed}"


@dataclass(frozen=True)
class RunOutcome:
    spec: RunSpec
    status: str  # "completed", "skipped" or "failed"
    seconds: float = 0.0
    summary: Mapping[str, Any] | None = None
    error: str | None = None


def _check_keys(data: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}; valid keys are {sorted(allowed)}")


def expand_experiment(spec: Mapping[str, Any], runs_root: str | Path = "runs") -> list[RunSpec]:
    """Turn an experiment specification into one fully resolved config per run."""
    _check_keys(spec, _SPEC_KEYS, "experiment")
    for key in ("name", "seeds", "variants"):
        if key not in spec:
            raise ConfigError(f"experiment: missing required key {key!r}")
    seeds, variants = spec["seeds"], spec["variants"]
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ConfigError("experiment.seeds must be a non-empty list of distinct seeds")
    if not isinstance(variants, list) or not variants:
        raise ConfigError("experiment.variants must be a non-empty list")

    base = resolve_section(spec.get("base"), "experiment.base")
    reserved = sorted(set(base) & _RESERVED_BASE_KEYS)
    if reserved:
        raise ConfigError(f"experiment.base must not set {reserved}; they are set per run")

    runs: list[RunSpec] = []
    labels: set[str] = set()
    for index, variant in enumerate(variants):
        where = f"experiment.variants[{index}]"
        if not isinstance(variant, Mapping):
            raise ConfigError(f"{where}: expected a mapping")
        _check_keys(variant, _VARIANT_KEYS, where)
        if "label" not in variant or "agent" not in variant:
            raise ConfigError(f"{where}: 'label' and 'agent' are required")
        overrides = variant.get("overrides") or {}
        if not isinstance(overrides, Mapping):
            raise ConfigError(f"{where}.overrides: expected a mapping of key.path to value")

        for label, combination in _grid_points(str(variant["label"]), variant.get("grid"), where):
            if label in labels:
                raise ConfigError(f"{where}: duplicate label {label!r}")
            labels.add(label)
            for seed in seeds:
                raw: dict[str, Any] = {
                    **resolve_section(base, "experiment.base"),
                    "name": spec["name"],
                    "label": label,
                    "seed": seed,
                    "agent": resolve_section(variant["agent"], f"{where}.agent"),
                }
                if "env" in raw:
                    raw["env"] = resolve_section(raw["env"], "experiment.base.env")
                for dotted_key, value in {**overrides, **combination}.items():
                    set_by_path(raw, str(dotted_key), value)
                config = build_experiment_config(raw)
                runs.append(RunSpec(config=config, run_dir=config.run_dir(runs_root)))
    return runs


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return "-".join(_format_value(v) for v in value)
    return str(value)


def _grid_points(label: str, grid: Any, where: str) -> list[tuple[str, dict[str, Any]]]:
    """``(label, overrides)`` for every combination of a variant's grid (one point if none)."""
    if grid is None:
        return [(label, {})]
    if not isinstance(grid, Mapping) or not grid:
        raise ConfigError(f"{where}.grid: expected a non-empty mapping of key.path to values")
    for key, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ConfigError(f"{where}.grid.{key}: expected a non-empty list of values")
    keys = [str(key) for key in grid]
    points = []
    for values in itertools.product(*grid.values()):
        suffix = "".join(
            f"__{key.rsplit('.', 1)[-1]}={_format_value(value)}"
            for key, value in zip(keys, values, strict=True)
        )
        points.append((label + suffix, dict(zip(keys, values, strict=True))))
    return points


def load_experiment(path: str | Path, runs_root: str | Path = "runs") -> list[RunSpec]:
    return expand_experiment(load_yaml(path), runs_root)


def _execute(spec: RunSpec, overwrite: bool) -> RunOutcome:
    """Run one spec. Top-level so that it can be sent to a worker process."""
    started = time.perf_counter()
    try:
        summary = run_experiment(spec.config, spec.run_dir, overwrite=overwrite)
    except Exception:
        return RunOutcome(
            spec, "failed", time.perf_counter() - started, error=traceback.format_exc()
        )
    return RunOutcome(spec, "completed", time.perf_counter() - started, summary=summary)


def run_sweep(
    runs: Sequence[RunSpec],
    *,
    max_workers: int = 1,
    overwrite: bool = False,
    on_result: Callable[[RunOutcome], None] | None = None,
) -> list[RunOutcome]:
    """Execute ``runs``, skipping those already completed unless ``overwrite`` is set.

    A sweep that was interrupted can therefore simply be started again. A failing run is
    reported and does not stop the others.
    """
    outcomes: list[RunOutcome] = []

    def record(outcome: RunOutcome) -> None:
        outcomes.append(outcome)
        if on_result is not None:
            on_result(outcome)

    pending = []
    for spec in runs:
        if not overwrite and RunPaths(spec.run_dir).is_complete():
            record(RunOutcome(spec, "skipped"))
        else:
            pending.append(spec)

    if max_workers <= 1 or len(pending) <= 1:
        for spec in pending:
            record(_execute(spec, overwrite))
    else:
        with ProcessPoolExecutor(max_workers=min(max_workers, len(pending))) as pool:
            futures = [pool.submit(_execute, spec, overwrite) for spec in pending]
            for future in as_completed(futures):
                record(future.result())
    return outcomes
