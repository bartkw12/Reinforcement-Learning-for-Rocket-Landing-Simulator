"""The benchmark's agent configs must be exactly the candidates tuning selected."""

from pathlib import Path

import pandas as pd
import pytest

from lunarlander_rl.agents.registry import parse_params
from lunarlander_rl.config import build_experiment_config, load_yaml
from lunarlander_rl.sweep import load_experiment

REPO_ROOT = Path(__file__).resolve().parents[2]
SELECTION = REPO_ROOT / "results" / "summary" / "tuning_selection.csv"


@pytest.mark.skipif(not SELECTION.exists(), reason="tuning has not been run")
def test_agent_configs_match_the_tuning_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    candidates = {
        run.config.variant: run.config.agent
        for run in load_experiment("configs/experiments/tuning.yaml")
    }
    selection = pd.read_csv(SELECTION)
    assert set(selection["family"]) == {"q_learning", "dqn", "reinforce"}
    for _, row in selection.iterrows():
        shipped = build_experiment_config(
            {"agent": load_yaml(f"configs/agent/{row['family']}.yaml")}
        ).agent
        assert parse_params(shipped) == parse_params(candidates[row["variant"]]), row["family"]
