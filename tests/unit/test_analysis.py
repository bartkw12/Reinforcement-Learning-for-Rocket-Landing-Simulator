import numpy as np
import pandas as pd
import pytest

from lunarlander_rl.analysis import replication, tuning
from lunarlander_rl.analysis.aggregate import steps_to_threshold
from lunarlander_rl.analysis.stats import bootstrap_ci
from lunarlander_rl.analysis.tables import markdown_table, number, percent, with_interval


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible() -> None:
    values = [10.0, 12.0, 9.0, 15.0, 11.0]
    low, high = bootstrap_ci(values)
    assert low < np.mean(values) < high
    assert min(values) <= low and high <= max(values)
    assert bootstrap_ci(values) == (low, high)


def test_bootstrap_ci_grows_with_spread() -> None:
    narrow = bootstrap_ci([9.0, 10.0, 11.0, 10.0])
    wide = bootstrap_ci([0.0, 10.0, 20.0, 10.0])
    assert wide[1] - wide[0] > narrow[1] - narrow[0]


def test_bootstrap_ci_edge_cases() -> None:
    assert bootstrap_ci([3.0]) == (3.0, 3.0)
    assert bootstrap_ci([2.0, 2.0, 2.0]) == (2.0, 2.0)
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_markdown_table() -> None:
    frame = pd.DataFrame(
        {"name": ["a", "b"], "value": [1.234, np.nan], "lo": [1.0, 0], "hi": [2.0, 0]}
    )
    text = markdown_table(
        frame,
        [
            ("Name", lambda r: r["name"]),
            ("Value [CI]", lambda r: with_interval(r["value"], r["lo"], r["hi"])),
        ],
    )
    assert text == ("| Name | Value [CI] |\n| :--- | ---: |\n| a | 1.2 [1.0, 2.0] |\n| b | n/a |\n")


def test_number_and_percent_formatting() -> None:
    assert number(-3.14159, 2) == "-3.14"
    assert number(None) == "n/a"
    assert percent(0.675) == "68%"
    assert percent(float("nan")) == "n/a"


def test_parse_and_describe_variants() -> None:
    assert replication.parse_variant("q_learning_cfg3") == ("q_learning", 3, False)
    assert replication.parse_variant("dqn_cfg1_v1bugs") == ("dqn", 1, True)
    assert replication.describe("reinforce_cfg2") == "REINFORCE, config 2"
    assert replication.describe("dqn_cfg1_v1bugs") == "DQN, config 1 (v1 bugs)"
    with pytest.raises(ValueError):
        replication.parse_variant("dqn")


def test_comparison_orders_rows_and_matches_v1_numbers() -> None:
    summary = pd.DataFrame(
        {
            "variant": ["reinforce_cfg1", "dqn_cfg1_v1bugs", "dqn_cfg1", "q_learning_cfg2"],
            "agent": ["reinforce", "dqn", "dqn", "q_learning"],
            "mean_return": [1.0, 2.0, 3.0, 4.0],
        }
    )
    reference = pd.DataFrame(
        {
            "variant": ["dqn_cfg1", "q_learning_cfg2", "reinforce_cfg1"],
            "mean_return": [194.7, -106.7, -35.2],
            "std_return": [np.nan, np.nan, 19.1],
            "success_rate": [0.67, 0.08, 0.0],
            "episodes": [100, 100, 100],
        }
    )
    rows = replication.comparison(summary, reference)
    assert rows["variant"].tolist() == [
        "q_learning_cfg2",
        "dqn_cfg1",
        "dqn_cfg1_v1bugs",
        "reinforce_cfg1",
    ]
    by_variant = rows.set_index("variant")
    assert by_variant.loc["dqn_cfg1", "v1_mean_return"] == 194.7
    # The bug-reproducing variant is compared against the config it copies.
    assert by_variant.loc["dqn_cfg1_v1bugs", "v1_mean_return"] == 194.7
    assert by_variant.loc["dqn_cfg1_v1bugs", "v1_success_rate"] == 0.67


def test_steps_to_threshold() -> None:
    evaluations = pd.DataFrame(
        {"env_step": [0, 10, 20, 30], "mean_return": [-100.0, 150.0, 205.0, 190.0]}
    )
    assert steps_to_threshold(evaluations, 200.0) == 20.0
    assert np.isnan(steps_to_threshold(evaluations, 300.0))


def tuning_summary() -> pd.DataFrame:
    summary: pd.DataFrame = pd.DataFrame(
        {
            "variant": [
                "dqn__zoo",
                "dqn__lr=0.0003",
                "q_learning__alpha=0.1__tiles_per_dim=4",
                "q_learning__alpha=0.3__tiles_per_dim=4",
            ],
            "mean_return": [180.0, 210.0, 50.0, 20.0],
            "seeds": [3, 3, 3, 3],
        }
    )
    return summary


def test_tuning_selection_picks_the_best_candidate_per_family() -> None:
    selected = tuning.selection(tuning_summary())
    assert selected["variant"].tolist() == [
        "dqn__lr=0.0003",
        "q_learning__alpha=0.1__tiles_per_dim=4",
    ]
    assert tuning.family("q_learning__alpha=0.1") == "q_learning"
    assert tuning.settings("q_learning__alpha=0.1__tiles_per_dim=4") == "alpha=0.1, tiles_per_dim=4"
    assert tuning.settings("dqn") == "(base)"


def test_tuning_table_marks_the_selection() -> None:
    summary = tuning_summary().assign(
        mean_return_ci_low=0.0,
        mean_return_ci_high=1.0,
        seed_min_return=0.0,
        seed_max_return=1.0,
        success_rate=0.5,
        best_mean_return=1.0,
    )
    text = tuning.table(summary)
    assert text.index("#### DQN") < text.index("#### Q-learning")
    assert "**lr=0.0003** (selected)" in text
    assert "| zoo |" in text
