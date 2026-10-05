"""Run directories in, tables and figures out: the replication report end to end."""

from pathlib import Path

import pandas as pd
import pytest

from lunarlander_rl.analysis import replication
from lunarlander_rl.analysis.aggregate import (
    evaluation_curves,
    final_results,
    summarize_variants,
    training_curves,
)
from lunarlander_rl.sweep import expand_experiment, run_sweep

VARIANTS = ["q_learning_cfg1", "dqn_cfg2", "dqn_cfg1_v1bugs"]


@pytest.fixture(scope="module")
def experiment(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("runs")
    spec = {
        "name": "replication_v1",
        "seeds": [0, 1],
        "base": {
            "train": {"total_steps": None, "total_episodes": 4},
            "eval": {"interval_steps": 200, "episodes": 1, "final_episodes": 4},
        },
        # The random agent stands in for the real ones: only the layout matters here.
        "variants": [{"label": v, "agent": {"name": "random"}} for v in VARIANTS],
    }
    outcomes = run_sweep(expand_experiment(spec, root))
    assert all(o.status == "completed" for o in outcomes)
    return root / "replication_v1"


def test_final_results_has_one_row_per_run(experiment: Path) -> None:
    runs = final_results(experiment)
    assert len(runs) == 6
    assert sorted(runs["variant"].unique()) == sorted(VARIANTS)
    assert (runs["episodes"] == 4).all()
    outcome_fractions = runs[[c for c in runs.columns if c.startswith("frac_")]].sum(axis=1)
    assert outcome_fractions.to_numpy() == pytest.approx(1.0)


def test_summary_aggregates_over_seeds(experiment: Path) -> None:
    runs = final_results(experiment)
    summary = summarize_variants(runs).set_index("variant")
    for variant in VARIANTS:
        seeds = runs.loc[runs["variant"] == variant, "mean_return"]
        row = summary.loc[variant]
        assert row["seeds"] == 2
        assert row["mean_return"] == pytest.approx(seeds.mean())
        assert row["mean_return_ci_low"] <= row["mean_return"] <= row["mean_return_ci_high"]
        assert row["seed_min_return"] == pytest.approx(seeds.min())


def test_training_curves_are_trailing_moving_averages(experiment: Path) -> None:
    curves = training_curves(experiment, window=2)
    one_run = curves[(curves["variant"] == "dqn_cfg2") & (curves["seed"] == 0)]
    returns = one_run["return"].to_numpy()
    expected = [returns[0], *((returns[1:] + returns[:-1]) / 2)]
    assert one_run["moving_average"].to_numpy() == pytest.approx(expected)
    assert len(evaluation_curves(experiment)) > 0


def test_replication_report_files(experiment: Path, tmp_path: Path) -> None:
    runs = final_results(experiment)
    reference = pd.DataFrame(
        {
            "variant": ["q_learning_cfg1", "dqn_cfg1", "dqn_cfg2"],
            "mean_return": [129.7, 194.7, 108.2],
            "std_return": [None, None, None],
            "success_rate": [0.62, 0.67, 0.31],
            "episodes": [100, 100, 100],
        }
    )
    rows = replication.comparison(summarize_variants(runs), reference)
    text = replication.table(rows)
    assert text.count("\n") == 2 + len(VARIANTS)
    assert "DQN, config 1 (v1 bugs)" in text and "194.7" in text

    first = [
        *replication.plot_learning_curves(training_curves(experiment), tmp_path / "a" / "curves"),
        *replication.plot_final_comparison(rows, runs, tmp_path / "a" / "final"),
    ]
    assert [p.suffix for p in first] == [".png", ".pdf", ".png", ".pdf"]
    assert all(p.stat().st_size > 1000 for p in first)

    # Regenerating from the same data gives byte-identical figures.
    second = replication.plot_final_comparison(rows, runs, tmp_path / "b" / "final")
    assert [p.read_bytes() for p in second] == [p.read_bytes() for p in first[2:]]
