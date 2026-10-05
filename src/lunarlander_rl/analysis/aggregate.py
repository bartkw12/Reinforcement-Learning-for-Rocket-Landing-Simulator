"""Collect the run directories of an experiment into tables.

An experiment directory has the layout ``<experiment>/<variant>/seed_<k>/``; only completed
runs (those with ``final_eval.json``) are read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lunarlander_rl.analysis.stats import bootstrap_ci
from lunarlander_rl.config import load_config
from lunarlander_rl.evaluation.metrics import CRASHED, LANDED, OUT_OF_BOUNDS, TIMEOUT
from lunarlander_rl.tracking import RunPaths, read_json

OUTCOMES = (LANDED, CRASHED, OUT_OF_BOUNDS, TIMEOUT)


def completed_runs(experiment_dir: str | Path) -> list[RunPaths]:
    root = Path(experiment_dir)
    runs = [RunPaths(path.parent) for path in sorted(root.glob("*/seed_*/final_eval.json"))]
    if not runs:
        raise FileNotFoundError(f"no completed runs under {root}")
    return runs


def _run_identity(paths: RunPaths) -> dict[str, Any]:
    config = load_config(paths.config)
    return {"variant": config.variant, "agent": config.agent.name, "seed": config.seed}


def final_results(experiment_dir: str | Path) -> pd.DataFrame:
    """One row per run: the final policy's evaluation, plus the best checkpoint's."""
    rows: list[dict[str, Any]] = []
    for paths in completed_runs(experiment_dir):
        report = read_json(paths.final_eval)
        meta = read_json(paths.meta)
        final = report["final"]["summary"]
        row: dict[str, Any] = {
            **_run_identity(paths),
            "env_steps": report["env_step"],
            "episodes": report["episode"],
            "training_seconds": meta.get("training_seconds"),
            "mean_return": final["mean_return"],
            "std_return": final["std_return"],
            "iqm_return": final["iqm_return"],
            "median_return": final["median_return"],
            "success_rate": final["success_rate"],
            "mean_length": final["mean_length"],
        }
        for outcome in OUTCOMES:
            row[f"frac_{outcome}"] = final["outcomes"].get(outcome, 0) / final["episodes"]
        best = report.get("best")
        row["best_env_step"] = best["env_step"] if best else np.nan
        row["best_mean_return"] = best["summary"]["mean_return"] if best else np.nan
        row["best_success_rate"] = best["summary"]["success_rate"] if best else np.nan
        rows.append(row)
    results: pd.DataFrame = pd.DataFrame(rows).sort_values(["variant", "seed"], ignore_index=True)
    return results


def summarize_variants(runs: pd.DataFrame, *, confidence: float = 0.95) -> pd.DataFrame:
    """Per variant: averages over seeds with bootstrap confidence intervals over seeds."""
    rows: list[dict[str, Any]] = []
    for variant, group in runs.groupby("variant", sort=True):
        row: dict[str, Any] = {
            "variant": variant,
            "agent": group["agent"].iloc[0],
            "seeds": len(group),
        }
        for column in ("mean_return", "success_rate", "best_mean_return"):
            values = group[column].dropna().astype(float).tolist()
            row[column] = float(np.mean(values)) if values else np.nan
            low, high = bootstrap_ci(values, confidence=confidence) if values else (np.nan, np.nan)
            row[f"{column}_ci_low"], row[f"{column}_ci_high"] = low, high
        row["seed_min_return"] = float(group["mean_return"].min())
        row["seed_max_return"] = float(group["mean_return"].max())
        row["env_steps"] = float(group["env_steps"].mean())
        row["episodes"] = float(group["episodes"].mean())
        row["training_minutes"] = float(group["training_seconds"].mean()) / 60.0
        for outcome in OUTCOMES:
            row[f"frac_{outcome}"] = float(group[f"frac_{outcome}"].mean())
        rows.append(row)
    summary: pd.DataFrame = pd.DataFrame(rows)
    return summary


def training_curves(experiment_dir: str | Path, *, window: int = 100) -> pd.DataFrame:
    """Training-episode returns of every run, with a trailing moving average.

    The moving average over the last ``window`` episodes (fewer at the start) is the
    learning curve v1.0 plotted.
    """
    frames = []
    for paths in completed_runs(experiment_dir):
        episodes = pd.read_csv(paths.train_csv, usecols=["env_step", "episode", "return"])
        episodes["moving_average"] = episodes["return"].rolling(window, min_periods=1).mean()
        for key, value in _run_identity(paths).items():
            episodes[key] = value
        frames.append(episodes)
    return pd.concat(frames, ignore_index=True)


def evaluation_curves(experiment_dir: str | Path) -> pd.DataFrame:
    """Periodic evaluations of every run (one row per evaluation)."""
    frames = []
    for paths in completed_runs(experiment_dir):
        evaluations = pd.read_csv(paths.eval_csv)
        for key, value in _run_identity(paths).items():
            evaluations[key] = value
        frames.append(evaluations)
    return pd.concat(frames, ignore_index=True)
