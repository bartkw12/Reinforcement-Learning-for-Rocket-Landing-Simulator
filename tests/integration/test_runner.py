"""End-to-end checks of the training loop, using the random agent."""

import csv
from pathlib import Path
from typing import Any

import pytest

from lunarlander_rl.config import ExperimentConfig, build_experiment_config
from lunarlander_rl.evaluation import evaluate_checkpoint
from lunarlander_rl.tracking import RunPaths, read_json
from lunarlander_rl.training import RunExistsError, run_experiment

DETERMINISTIC_FILES = ("train_episodes.csv", "eval.csv", "final_eval.json")


def make_config(**overrides: Any) -> ExperimentConfig:
    raw: dict[str, Any] = {
        "agent": {"name": "random"},
        "seed": 0,
        "train": {"total_steps": 600},
        "eval": {"interval_steps": 300, "episodes": 2, "final_episodes": 3},
    }
    raw.update(overrides)
    return build_experiment_config(raw)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("runs") / "seed_0"
    run_experiment(make_config(), path)
    return path


def test_run_directory_contents(run_dir: Path) -> None:
    paths = RunPaths(run_dir)
    assert paths.is_complete()
    assert paths.config.is_file()
    assert paths.find_checkpoint("final").is_file()
    assert paths.find_checkpoint("best").is_file()

    meta = read_json(paths.meta)
    assert meta["status"] == "completed"
    assert meta["env_steps"] == 600
    assert "gymnasium" in meta["packages"] and "torch" in meta["packages"]
    assert {"git_commit", "git_dirty", "python", "platform", "started_at"} <= set(meta)


def test_training_log(run_dir: Path) -> None:
    rows = read_rows(run_dir / "train_episodes.csv")
    assert len(rows) >= 2
    assert [int(r["episode"]) for r in rows] == list(range(1, len(rows) + 1))
    steps = [int(r["env_step"]) for r in rows]
    assert steps == sorted(steps) and steps[-1] <= 600
    # Episode lengths account for every logged step.
    assert sum(int(r["length"]) for r in rows) == steps[-1]
    # Exactly one of the two episode-ending flags is set on every row.
    assert all(int(r["terminated"]) + int(r["truncated"]) == 1 for r in rows)


def test_evaluation_log_and_final_report(run_dir: Path) -> None:
    rows = read_rows(run_dir / "eval.csv")
    assert [int(r["env_step"]) for r in rows] == [0, 300, 600]

    final_eval = read_json(run_dir / "final_eval.json")
    assert final_eval["env_step"] == 600
    final = final_eval["final"]
    assert [e["seed"] for e in final["episodes"]] == [10_000, 10_001, 10_002]
    assert final["summary"]["episodes"] == 3
    assert sum(final["summary"]["outcomes"].values()) == 3
    assert final_eval["best"]["env_step"] in (0, 300, 600)


def test_same_seed_reproduces_the_run_exactly(run_dir: Path, tmp_path: Path) -> None:
    run_experiment(make_config(), tmp_path / "again")
    for name in DETERMINISTIC_FILES:
        assert (tmp_path / "again" / name).read_bytes() == (run_dir / name).read_bytes(), name


def test_different_seed_gives_a_different_run(run_dir: Path, tmp_path: Path) -> None:
    run_experiment(make_config(seed=1), tmp_path / "other")
    assert (tmp_path / "other" / "train_episodes.csv").read_bytes() != (
        run_dir / "train_episodes.csv"
    ).read_bytes()


def test_evaluation_schedule_does_not_change_training(run_dir: Path, tmp_path: Path) -> None:
    config = make_config(eval={"interval_steps": 100, "episodes": 1, "final_episodes": 3})
    run_experiment(config, tmp_path / "more_evals")
    assert (tmp_path / "more_evals" / "train_episodes.csv").read_bytes() == (
        run_dir / "train_episodes.csv"
    ).read_bytes()


def test_episode_budget_stops_training(tmp_path: Path) -> None:
    config = make_config(train={"total_steps": None, "total_episodes": 3})
    run_experiment(config, tmp_path / "episodes")
    assert len(read_rows(tmp_path / "episodes" / "train_episodes.csv")) == 3
    assert read_json(tmp_path / "episodes" / "final_eval.json")["episode"] == 3


def test_completed_run_is_protected(run_dir: Path) -> None:
    with pytest.raises(RunExistsError):
        run_experiment(make_config(), run_dir)


def test_overwrite_and_incomplete_runs_are_replaced(tmp_path: Path) -> None:
    target = tmp_path / "run"
    run_experiment(make_config(), target)
    run_experiment(make_config(), target, overwrite=True)

    # An interrupted run has no final_eval.json and is restarted without being asked.
    (target / "final_eval.json").unlink()
    run_experiment(make_config(), target)
    assert RunPaths(target).is_complete()


def test_refuses_to_delete_a_directory_that_is_not_a_run(tmp_path: Path) -> None:
    (tmp_path / "thesis.docx").write_text("important")
    with pytest.raises(FileExistsError):
        run_experiment(make_config(), tmp_path, overwrite=True)
    assert (tmp_path / "thesis.docx").exists()


def test_evaluate_checkpoint_reproduces_the_final_evaluation(run_dir: Path) -> None:
    assert evaluate_checkpoint(run_dir) == read_json(run_dir / "final_eval.json")["final"]


def test_evaluate_checkpoint_with_changed_conditions(run_dir: Path) -> None:
    windy = evaluate_checkpoint(
        run_dir,
        seeds=range(10_000, 10_002),
        overrides=["env.kwargs.enable_wind=true", "env.kwargs.wind_power=20.0"],
    )
    assert windy["summary"]["episodes"] == 2
    calm = read_json(run_dir / "final_eval.json")["final"]["episodes"][:2]
    assert [e["episode_return"] for e in windy["episodes"]] != [e["episode_return"] for e in calm]
