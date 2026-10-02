from pathlib import Path
from typing import Any

import pytest

from lunarlander_rl.config import ConfigError
from lunarlander_rl.sweep import RunOutcome, expand_experiment, load_experiment, run_sweep
from lunarlander_rl.tracking import RunPaths

REPO_ROOT = Path(__file__).resolve().parents[2]


def make_spec(**changes: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "name": "tiny",
        "seeds": [0, 1],
        "base": {
            "train": {"total_steps": 200},
            "eval": {"interval_steps": 200, "episodes": 1, "final_episodes": 2},
        },
        "variants": [{"label": "random", "agent": {"name": "random"}}],
    }
    spec.update(changes)
    return spec


def test_expand_experiment_creates_one_run_per_variant_and_seed(tmp_path: Path) -> None:
    spec = make_spec(
        variants=[
            {"label": "calm", "agent": {"name": "random"}},
            {
                "label": "windy",
                "agent": {"name": "random"},
                "overrides": {"env.kwargs.enable_wind": True, "train.total_steps": 50},
            },
        ]
    )
    runs = expand_experiment(spec, tmp_path)
    assert [run.tag for run in runs] == [
        "calm/seed_0",
        "calm/seed_1",
        "windy/seed_0",
        "windy/seed_1",
    ]
    assert runs[0].run_dir == tmp_path / "tiny" / "calm" / "seed_0"
    assert runs[0].config.env.kwargs == {} and runs[0].config.train.total_steps == 200
    assert runs[2].config.env.kwargs == {"enable_wind": True}
    assert runs[2].config.train.total_steps == 50


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"seeds": []}, "seeds"),
        ({"seeds": [0, 0]}, "seeds"),
        ({"variants": []}, "variants"),
        ({"variants": [{"label": "a"}]}, "required"),
        ({"variants": [{"label": "a", "agent": {"name": "random"}}] * 2}, "duplicate"),
        ({"variants": [{"label": "a", "agent": {"name": "random"}, "typo": 1}]}, "unknown key"),
        ({"base": {"seed": 3}}, "must not set"),
        ({"base": {"train": {"steps": 5}}}, "unknown key"),
        ({"extra": 1}, "unknown key"),
    ],
)
def test_expand_experiment_rejects_bad_specs(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        expand_experiment(make_spec(**changes))


def test_shipped_smoke_experiment_is_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    runs = load_experiment("configs/experiments/smoke.yaml")
    assert len(runs) == 2
    assert all(run.config.agent.name == "random" for run in runs)


def test_sweep_runs_then_skips_completed_runs(tmp_path: Path) -> None:
    runs = expand_experiment(make_spec(), tmp_path)

    first = run_sweep(runs)
    assert [o.status for o in first] == ["completed", "completed"]
    assert all(RunPaths(run.run_dir).is_complete() for run in runs)

    seen: list[RunOutcome] = []
    second = run_sweep(runs, on_result=seen.append)
    assert [o.status for o in second] == ["skipped", "skipped"]
    assert seen == second


def test_sweep_reports_failures_and_continues(tmp_path: Path) -> None:
    spec = make_spec(
        seeds=[0],
        variants=[
            {
                "label": "broken",
                "agent": {"name": "random"},
                "overrides": {"env.id": "NoSuchEnv-v0"},
            },
            {"label": "fine", "agent": {"name": "random"}},
        ],
    )
    outcomes = run_sweep(expand_experiment(spec, tmp_path))
    assert [o.status for o in outcomes] == ["failed", "completed"]
    assert outcomes[0].error is not None and "NoSuchEnv" in outcomes[0].error


@pytest.mark.slow
def test_parallel_sweep_matches_sequential(tmp_path: Path) -> None:
    sequential = expand_experiment(make_spec(), tmp_path / "sequential")
    parallel = expand_experiment(make_spec(), tmp_path / "parallel")
    run_sweep(sequential)
    outcomes = run_sweep(parallel, max_workers=2)
    assert sorted(o.status for o in outcomes) == ["completed", "completed"]
    for a, b in zip(sequential, parallel, strict=True):
        for name in ("train_episodes.csv", "eval.csv", "final_eval.json"):
            assert (a.run_dir / name).read_bytes() == (b.run_dir / name).read_bytes()
