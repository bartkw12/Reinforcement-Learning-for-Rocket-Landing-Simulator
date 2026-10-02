from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pytest

from lunarlander_rl.config import (
    AgentConfig,
    ConfigError,
    EvalConfig,
    ExperimentConfig,
    TrainConfig,
    apply_overrides,
    build_experiment_config,
    from_dict,
    load_config,
    save_config,
    to_dict,
)


@dataclass(frozen=True)
class Inner:
    rate: float = 0.5


@dataclass(frozen=True)
class Example:
    required: int
    lr: float = 1e-3
    flag: bool = False
    loss: Literal["huber", "mse"] = "huber"
    hidden: tuple[int, ...] = (64, 64)
    bounds: tuple[float, float] = (-1.0, 1.0)
    limit: int | None = None
    inner: Inner = field(default_factory=Inner)
    extra: dict[str, Any] = field(default_factory=dict)


def test_from_dict_builds_nested_dataclasses() -> None:
    example = from_dict(
        Example,
        {"required": 3, "hidden": [32, 16], "inner": {"rate": 1}, "limit": 7, "loss": "mse"},
    )
    assert example == Example(
        required=3, hidden=(32, 16), inner=Inner(rate=1.0), limit=7, loss="mse"
    )
    assert isinstance(example.inner.rate, float)


def test_from_dict_reads_yaml_style_exponent_strings_as_floats() -> None:
    # PyYAML parses "1e-3" (no decimal point) as a string.
    assert from_dict(Example, {"required": 1, "lr": "1e-3"}).lr == pytest.approx(0.001)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"required": 1, "learning_rate": 0.1}, "unknown key"),
        ({}, "missing required"),
        ({"required": "three"}, "expected int"),
        ({"required": True}, "expected int"),
        ({"required": 1, "flag": 1}, "expected bool"),
        ({"required": 1, "lr": "fast"}, "expected float"),
        ({"required": 1, "loss": "l1"}, "expected one of"),
        ({"required": 1, "hidden": 64}, "expected a list"),
        ({"required": 1, "hidden": [64, "big"]}, "expected int"),
        ({"required": 1, "bounds": [0.0, 1.0, 2.0]}, "expected 2 items"),
        ({"required": 1, "limit": "none"}, "does not match"),
        ({"required": 1, "inner": {"speed": 1}}, "unknown key"),
    ],
)
def test_from_dict_rejects_malformed_input(data: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        from_dict(Example, data)


def test_error_message_names_the_nested_key() -> None:
    with pytest.raises(ConfigError, match=r"inner\.rate"):
        from_dict(Example, {"required": 1, "inner": {"rate": "x"}})


def test_apply_overrides_parses_values_and_does_not_mutate_input() -> None:
    original = {"train": {"total_steps": 10}}
    result = apply_overrides(
        original,
        ["train.total_steps=500", "env.kwargs.enable_wind=true", "agent.params.lr=0.001"],
    )
    assert result == {
        "train": {"total_steps": 500},
        "env": {"kwargs": {"enable_wind": True}},
        "agent": {"params": {"lr": 0.001}},
    }
    assert original == {"train": {"total_steps": 10}}


@pytest.mark.parametrize("override", ["no_equals_sign", "=5"])
def test_apply_overrides_rejects_malformed_overrides(override: str) -> None:
    with pytest.raises(ConfigError):
        apply_overrides({}, [override])


def test_train_config_needs_a_budget() -> None:
    with pytest.raises(ConfigError):
        TrainConfig(total_steps=None, total_episodes=None)
    with pytest.raises(ConfigError):
        TrainConfig(total_steps=0)
    assert TrainConfig(total_steps=None, total_episodes=500).total_episodes == 500


def test_eval_seed_ranges_must_not_overlap() -> None:
    with pytest.raises(ConfigError, match="overlap"):
        EvalConfig(seed_start=10_050, final_seed_start=10_000, final_episodes=100)
    config = EvalConfig()
    assert set(config.seeds).isdisjoint(config.final_seeds)
    assert len(config.final_seeds) == config.final_episodes


def test_build_experiment_config_resolves_files_and_overrides(tmp_path: Path) -> None:
    agent_file = tmp_path / "agent.yaml"
    agent_file.write_text("name: random\n")
    config = build_experiment_config(
        {"agent": str(agent_file), "env": {"id": "CartPole-v1"}, "seed": 3},
        ["train.total_steps=123", "label=baseline"],
    )
    assert config.agent == AgentConfig(name="random")
    assert config.env.id == "CartPole-v1"
    assert config.train.total_steps == 123
    assert config.run_dir("out") == Path("out") / "adhoc" / "baseline" / "seed_3"


def test_run_dir_falls_back_to_agent_name() -> None:
    config = ExperimentConfig(agent=AgentConfig(name="random"), name="exp", seed=1)
    assert config.run_dir() == Path("runs") / "exp" / "random" / "seed_1"


def test_missing_config_file_is_reported() -> None:
    with pytest.raises(ConfigError, match="not found"):
        build_experiment_config({"agent": "does/not/exist.yaml"})


def test_config_round_trips_through_yaml(tmp_path: Path) -> None:
    config = build_experiment_config(
        {
            "agent": {"name": "random", "params": {}},
            "env": {"id": "LunarLander-v3", "kwargs": {"enable_wind": True}},
            "train": {"total_steps": None, "total_episodes": 50},
            "seed": 7,
        }
    )
    path = tmp_path / "config.yaml"
    save_config(config, path)
    assert load_config(path) == config
    assert to_dict(config)["env"]["kwargs"] == {"enable_wind": True}
