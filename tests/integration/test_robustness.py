"""Re-evaluating saved policies under changed conditions, and the reports built from it."""

from pathlib import Path
from typing import Any

import pytest

from lunarlander_rl.analysis import robustness as report
from lunarlander_rl.config import ConfigError
from lunarlander_rl.robustness import load_spec, parse_spec, run_robustness
from lunarlander_rl.sweep import expand_experiment, run_sweep
from lunarlander_rl.tracking import read_json

REPO_ROOT = Path(__file__).resolve().parents[2]
WINDY = {"env.kwargs.enable_wind": True, "env.kwargs.wind_power": 20.0}


@pytest.fixture(scope="module")
def experiment(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("runs")
    spec = {
        "name": "bench",
        "seeds": [0, 1],
        "base": {
            "train": {"total_steps": 300},
            "eval": {"interval_steps": 300, "episodes": 1, "final_episodes": 3},
        },
        "variants": [{"label": "random", "agent": {"name": "random"}}],
    }
    assert all(o.status == "completed" for o in run_sweep(expand_experiment(spec, root)))
    return root / "bench"


def make_spec(experiment: Path, **changes: Any) -> Any:
    data: dict[str, Any] = {
        "name": "wind",
        "runs": str(experiment),
        "conditions": [
            {"label": "calm", "level": 0, "overrides": {}},
            {"label": "windy", "level": 20, "overrides": WINDY},
        ],
    }
    data.update(changes)
    return parse_spec(data)


def test_calm_reevaluation_reproduces_the_final_evaluation(experiment: Path) -> None:
    spec = make_spec(experiment)
    outcomes = run_robustness(spec)
    assert sorted(o.status for o in outcomes) == ["completed"] * 4
    for seed in (0, 1):
        run = experiment / "random" / f"seed_{seed}"
        calm = read_json(run / "evaluations" / "wind" / "calm.json")
        assert calm["episodes"] == read_json(run / "final_eval.json")["final"]["episodes"]
        windy = read_json(run / "evaluations" / "wind" / "windy.json")
        assert windy["level"] == 20.0 and windy["episodes"] != calm["episodes"]
    # Existing results are kept.
    assert {o.status for o in run_robustness(spec)} == {"skipped"}


def test_robustness_report(experiment: Path, tmp_path: Path) -> None:
    spec = make_spec(experiment)
    run_robustness(spec)
    summary = report.summarize(report.collect(spec))
    assert summary["condition"].tolist() == ["calm", "windy"]
    assert (summary["seeds"] == 2).all()
    text = report.table(summary)
    assert text.splitlines()[0] == "| Agent | calm | windy |"
    assert "Random" in report.success_table(summary)
    assert [p.suffix for p in report.plot(summary, tmp_path / "fig")] == [".png", ".pdf"]


def test_missing_results_are_reported(experiment: Path) -> None:
    spec = make_spec(experiment, name="never_run")
    with pytest.raises(FileNotFoundError, match=r"robustness\.py"):
        report.collect(spec)


@pytest.mark.parametrize(
    "changes",
    [
        {"conditions": []},
        {"conditions": [{"label": "a"}, {"label": "a"}]},
        {"conditions": [{"level": 1}]},
        {"conditions": [{"label": "a", "overrides": ["x"]}]},
        {"checkpoint": "latest"},
        {"surprise": 1},
    ],
)
def test_malformed_specs_are_rejected(tmp_path: Path, changes: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        make_spec(tmp_path, **changes)


def test_shipped_wind_spec_is_valid() -> None:
    spec = load_spec(REPO_ROOT / "configs" / "robustness" / "wind.yaml")
    assert [c.level for c in spec.conditions] == [0, 5, 10, 15, 20]
    assert spec.conditions[0].overrides == ()
    assert "env.kwargs.wind_power=15.0" in spec.conditions[3].overrides
