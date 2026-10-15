"""The command-line entry points that the reproducibility guide relies on."""

import subprocess
import sys
from pathlib import Path

from lunarlander_rl.config import build_experiment_config
from lunarlander_rl.training import run_experiment

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_script(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / args[0]), *args[1:]],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_train_from_run_check_reproduces_a_run(tmp_path: Path) -> None:
    config = build_experiment_config(
        {
            "agent": {"name": "random"},
            "name": "exp",
            "train": {"total_steps": 300},
            "eval": {"interval_steps": 300, "episodes": 1, "final_episodes": 2},
        }
    )
    original = config.run_dir(tmp_path / "runs")
    run_experiment(config, original)

    result = run_script("train.py", "--from-run", str(original), "--check", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "reproduces the original run exactly" in result.stdout
    assert (tmp_path / "runs_verify" / "exp" / "random" / "seed_0" / "final_eval.json").is_file()

    # A tampered original is caught.
    (original / "eval.csv").write_text("tampered\n")
    result = run_script("train.py", "--from-run", str(original), "--check", cwd=tmp_path)
    assert result.returncode == 1
    assert "eval.csv: DIFFERENT" in result.stdout


def test_sync_docs_check_passes_on_the_repository() -> None:
    result = run_script("sync_docs.py", "--check", cwd=REPO_ROOT)
    assert result.returncode == 0, result.stdout
