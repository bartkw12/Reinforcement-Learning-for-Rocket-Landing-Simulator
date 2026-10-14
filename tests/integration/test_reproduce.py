"""The one-command reproduction, checked in quick mode."""

from pathlib import Path

import pytest

from lunarlander_rl import reproduce
from lunarlander_rl.config import build_experiment_config
from lunarlander_rl.reports import REPORTS, Paths, report_named

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_quick_config_shrinks_budgets_but_keeps_the_protocol() -> None:
    steps = build_experiment_config({"agent": {"name": "random"}})
    quick = reproduce.quick_config(steps)
    assert quick.train.total_steps == reproduce.QUICK_STEPS
    assert quick.eval.final_episodes == 5
    assert quick.eval.final_seed_start == steps.eval.final_seed_start
    assert quick.agent == steps.agent

    episodes = build_experiment_config(
        {"agent": {"name": "random"}, "train": {"total_steps": None, "total_episodes": 500}}
    )
    assert reproduce.quick_config(episodes).train.total_episodes == reproduce.QUICK_EPISODES


def test_every_registered_experiment_expands(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    for report in REPORTS:
        if report.kind != "sweep":
            continue
        full = reproduce.experiment_runs(report, Paths(), quick=False)
        quick = reproduce.experiment_runs(report, Paths(), quick=True)
        assert full and quick and len(quick) < len(full), report.name
        assert {r.config.variant for r in quick} == {r.config.variant for r in full}


def test_unknown_experiment_is_rejected() -> None:
    with pytest.raises(KeyError):
        reproduce.reproduce(only=["nope"], skip_runs=True)
    with pytest.raises(KeyError):
        report_named("nope")


@pytest.mark.slow
def test_quick_reproduction_of_the_benchmark_and_wind_study(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setattr(reproduce, "QUICK_PATHS", Paths(tmp_path / "runs", tmp_path / "results"))
    steps = reproduce.reproduce(quick=True, workers=2, only=["main_benchmark", "wind"])
    assert all(step.failed == 0 for step in steps)
    results = tmp_path / "results"
    assert (results / "tables" / "main_benchmark_results.md").is_file()
    assert (results / "figures" / "main_benchmark" / "iqm_curves.png").is_file()
    assert (results / "tables" / "robustness_wind.md").is_file()
    # Restarting skips everything already done.
    again = reproduce.reproduce(quick=True, workers=2, only=["main_benchmark", "wind"])
    assert all(step.completed == 0 for step in again if step.stage != "report")
