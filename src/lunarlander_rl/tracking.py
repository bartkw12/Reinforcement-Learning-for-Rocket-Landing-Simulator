"""Run directories and metric files: the on-disk record of an experiment.

Layout of one run directory::

    config.yaml           fully resolved configuration
    meta.json             provenance (git commit, package versions, timings, status)
    train_episodes.csv    one row per training episode
    eval.csv              one row per periodic evaluation
    final_eval.json       per-episode results on the held-out evaluation seeds
    checkpoints/          final.* and best.*

A run is complete if and only if ``final_eval.json`` exists; it is written last.
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def meta(self) -> Path:
        return self.root / "meta.json"

    @property
    def train_csv(self) -> Path:
        return self.root / "train_episodes.csv"

    @property
    def eval_csv(self) -> Path:
        return self.root / "eval.csv"

    @property
    def final_eval(self) -> Path:
        return self.root / "final_eval.json"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    def checkpoint(self, name: str, suffix: str) -> Path:
        return self.checkpoints / f"{name}{suffix}"

    def find_checkpoint(self, name: str) -> Path:
        matches = sorted(self.checkpoints.glob(f"{name}.*"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected exactly one '{name}' checkpoint in {self.checkpoints}, "
                f"found {len(matches)}"
            )
        return matches[0]

    def is_complete(self) -> bool:
        return self.final_eval.is_file()


class CsvLogger:
    """Append rows to a CSV file with a fixed header.

    Rows are flushed as they are written, so a crashed run keeps what it logged. Line
    endings are always ``\\n`` so that identical runs give byte-identical files on any OS.
    """

    def __init__(self, path: Path, fieldnames: Sequence[str]) -> None:
        self._handle = path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=fieldnames, lineterminator="\n")
        self._writer.writeheader()
        self._handle.flush()

    def log(self, row: Mapping[str, Any]) -> None:
        self._writer.writerow(row)
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> CsvLogger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically, so readers never see a half-written file."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
