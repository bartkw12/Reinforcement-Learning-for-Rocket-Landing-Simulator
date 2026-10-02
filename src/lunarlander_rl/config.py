"""Typed experiment configuration.

Configs are frozen dataclasses built from YAML. Construction is strict: unknown keys,
missing required keys and wrongly typed values all raise :class:`ConfigError`, so a typo
in a config file fails before any compute is spent.
"""

from __future__ import annotations

import copy
import dataclasses
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when a configuration is malformed."""


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EnvConfig:
    """Which environment to build and how."""

    id: str = "LunarLander-v3"
    # Passed to ``gym.make``, e.g. ``{"enable_wind": true, "wind_power": 15.0}``.
    kwargs: dict[str, Any] = field(default_factory=dict)
    # ``None`` keeps the environment's registered time limit (1000 for LunarLander).
    max_episode_steps: int | None = None


@dataclass(frozen=True)
class AgentConfig:
    """Which agent to build. ``params`` is validated against the agent's own schema."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainConfig:
    """Training budget. Training stops at whichever limit is reached first."""

    total_steps: int | None = 1_000_000
    total_episodes: int | None = None
    # Keep a checkpoint of the policy with the best periodic evaluation.
    save_best: bool = True

    def __post_init__(self) -> None:
        if self.total_steps is None and self.total_episodes is None:
            raise ConfigError("train: set total_steps and/or total_episodes")
        for name in ("total_steps", "total_episodes"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ConfigError(f"train.{name} must be positive, got {value}")


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation protocol.

    Periodic evaluations (during training) and the final evaluation use disjoint,
    fixed sets of episode seeds, so that the best checkpoint is never selected on the
    episodes it is finally reported on, and every agent faces identical start states.
    """

    interval_steps: int = 10_000
    episodes: int = 10
    seed_start: int = 20_000
    final_episodes: int = 100
    final_seed_start: int = 10_000
    deterministic: bool = True
    # An episode counts as a success if it terminates with at least this return.
    success_return: float = 200.0

    def __post_init__(self) -> None:
        for name in ("interval_steps", "episodes", "final_episodes"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"eval.{name} must be positive, got {getattr(self, name)}")
        periodic = range(self.seed_start, self.seed_start + self.episodes)
        final = range(self.final_seed_start, self.final_seed_start + self.final_episodes)
        if periodic.start < final.stop and final.start < periodic.stop:
            raise ConfigError("eval: periodic and final evaluation seed ranges overlap")

    @property
    def seeds(self) -> range:
        return range(self.seed_start, self.seed_start + self.episodes)

    @property
    def final_seeds(self) -> range:
        return range(self.final_seed_start, self.final_seed_start + self.final_episodes)


@dataclass(frozen=True)
class ExperimentConfig:
    """Everything needed to reproduce one training run."""

    agent: AgentConfig
    # Experiment name and variant label; together with the seed they locate the run directory.
    name: str = "adhoc"
    label: str = ""
    seed: int = 0
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    device: str = "cpu"
    # One thread keeps CPU runs deterministic and lets sweeps use one core per run.
    torch_threads: int = 1

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ConfigError(f"seed must be non-negative, got {self.seed}")
        if self.torch_threads <= 0:
            raise ConfigError(f"torch_threads must be positive, got {self.torch_threads}")

    @property
    def variant(self) -> str:
        return self.label or self.agent.name

    def run_dir(self, runs_root: str | Path = "runs") -> Path:
        return Path(runs_root) / self.name / self.variant / f"seed_{self.seed}"


# --------------------------------------------------------------------------- #
# Strict dict -> dataclass conversion
# --------------------------------------------------------------------------- #
def _type_name(tp: Any) -> str:
    return getattr(tp, "__name__", None) or str(tp)


def _coerce(value: Any, tp: Any, where: str) -> Any:
    """Check ``value`` against the annotation ``tp`` and return it in canonical form."""
    if tp is Any:
        return value

    origin = get_origin(tp)

    if origin in (Union, types.UnionType):
        for option in get_args(tp):
            try:
                return _coerce(value, option, where)
            except ConfigError:
                continue
        raise ConfigError(f"{where}: {value!r} does not match {tp}")

    if tp is type(None):
        if value is None:
            return None
        raise ConfigError(f"{where}: expected null, got {value!r}")

    if origin is Literal:
        if value in get_args(tp):
            return value
        raise ConfigError(f"{where}: expected one of {list(get_args(tp))}, got {value!r}")

    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        if isinstance(value, tp):
            return value
        return from_dict(tp, value, where=where)

    if origin in (tuple, list):
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise ConfigError(f"{where}: expected a list, got {value!r}")
        args = get_args(tp)
        if origin is tuple and args and not (len(args) == 2 and args[1] is Ellipsis):
            if len(value) != len(args):
                raise ConfigError(f"{where}: expected {len(args)} items, got {len(value)}")
            item_types: Sequence[Any] = args
        else:
            item_types = [args[0] if args else Any] * len(value)
        # Tuples keep configs immutable; YAML and JSON write them back as lists.
        return tuple(
            _coerce(item, item_tp, f"{where}[{i}]")
            for i, (item, item_tp) in enumerate(zip(value, item_types, strict=True))
        )

    if origin is dict or tp is dict:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{where}: expected a mapping, got {value!r}")
        args = get_args(tp)
        value_tp = args[1] if len(args) == 2 else Any
        return {str(k): _coerce(v, value_tp, f"{where}.{k}") for k, v in value.items()}

    if tp is bool:
        if isinstance(value, bool):
            return value
    elif tp is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif tp is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        # PyYAML reads exponents without a decimal point (``1e-3``) as strings.
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
    elif tp is str:
        if isinstance(value, str):
            return value
    else:
        raise ConfigError(f"{where}: unsupported annotation {tp}")

    raise ConfigError(f"{where}: expected {_type_name(tp)}, got {value!r}")


def from_dict(cls: type[T], data: Any, *, where: str = "") -> T:
    """Build the dataclass ``cls`` from a mapping, rejecting unknown or mistyped keys."""
    if not (dataclasses.is_dataclass(cls) and isinstance(cls, type)):
        raise TypeError(f"{cls!r} is not a dataclass")
    prefix = f"{where}." if where else ""
    label = where or cls.__name__
    if not isinstance(data, Mapping):
        raise ConfigError(f"{label}: expected a mapping, got {data!r}")

    hints = get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls) if f.init}

    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ConfigError(f"{label}: unknown key(s) {unknown}; valid keys are {sorted(fields)}")

    missing = sorted(
        name
        for name, f in fields.items()
        if name not in data
        and f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING
    )
    if missing:
        raise ConfigError(f"{label}: missing required key(s) {missing}")

    kwargs = {name: _coerce(data[name], hints[name], f"{prefix}{name}") for name in data}
    instance: T = cls(**kwargs)
    return instance


def to_dict(config: Any) -> dict[str, Any]:
    """Plain, YAML/JSON-serialisable view of a config (tuples become lists)."""

    def plain(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        return value

    result: dict[str, Any] = plain(dataclasses.asdict(config))
    return result


# --------------------------------------------------------------------------- #
# YAML helpers
# --------------------------------------------------------------------------- #
def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def set_by_path(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``data["a"]["b"]["c"] = value`` for ``dotted_key == "a.b.c"``, creating levels."""
    *parents, leaf = dotted_key.split(".")
    node = data
    for part in parents:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise ConfigError(f"cannot set {dotted_key!r}: {part!r} is not a mapping")
        node = child
    node[leaf] = value


def apply_overrides(data: Mapping[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Return a copy of ``data`` with ``key.path=value`` overrides applied.

    Values are parsed as YAML, so ``agent.params.lr=0.001`` and ``env.kwargs.enable_wind=true``
    produce a float and a bool respectively.
    """
    result = copy.deepcopy(dict(data))
    for override in overrides:
        key, sep, raw = override.partition("=")
        if not sep or not key:
            raise ConfigError(f"override must look like key.path=value, got {override!r}")
        set_by_path(result, key.strip(), yaml.safe_load(raw))
    return result


def resolve_section(value: Any, where: str) -> dict[str, Any]:
    """A config section may be written inline or as a path to another YAML file."""
    if value is None:
        return {}
    if isinstance(value, (str, Path)):
        return load_yaml(value)
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    raise ConfigError(f"{where}: expected a mapping or a path to a YAML file, got {value!r}")


def build_experiment_config(
    raw: Mapping[str, Any], overrides: Sequence[str] = ()
) -> ExperimentConfig:
    """Resolve file references in ``raw``, apply overrides and validate."""
    data = dict(raw)
    for section in ("env", "agent"):
        if section in data:
            data[section] = resolve_section(data[section], section)
    return from_dict(ExperimentConfig, apply_overrides(data, overrides))


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(to_dict(config), handle, sort_keys=False)


def load_config(path: str | Path) -> ExperimentConfig:
    return from_dict(ExperimentConfig, load_yaml(path))
