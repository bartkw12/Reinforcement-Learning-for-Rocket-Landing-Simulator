from pathlib import Path

import pytest

from lunarlander_rl.tracking import CsvLogger, RunPaths, read_json, write_json


def test_csv_logger_writes_lf_terminated_rows(tmp_path: Path) -> None:
    path = tmp_path / "log.csv"
    with CsvLogger(path, ("step", "value")) as log:
        log.log({"step": 1, "value": 0.5})
        log.log({"step": 2, "value": ""})
    assert path.read_bytes() == b"step,value\n1,0.5\n2,\n"


def test_csv_logger_rejects_unknown_columns(tmp_path: Path) -> None:
    with CsvLogger(tmp_path / "log.csv", ("step",)) as log, pytest.raises(ValueError):
        log.log({"step": 1, "surprise": 2})


def test_csv_logger_flushes_each_row(tmp_path: Path) -> None:
    path = tmp_path / "log.csv"
    log = CsvLogger(path, ("step",))
    log.log({"step": 1})
    # Readable before the file is closed: a crashed run keeps its rows.
    assert path.read_text() == "step\n1\n"
    log.close()


def test_write_json_round_trips_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    payload = {"b": [1, 2.5, None], "a": {"nested": True}}
    write_json(path, payload)
    assert read_json(path) == payload
    assert [p.name for p in tmp_path.iterdir()] == ["data.json"]
    assert b"\r" not in path.read_bytes()


def test_run_paths(tmp_path: Path) -> None:
    paths = RunPaths(tmp_path)
    assert not paths.is_complete()
    paths.checkpoints.mkdir()
    with pytest.raises(FileNotFoundError):
        paths.find_checkpoint("final")
    paths.checkpoint("final", ".pt").write_bytes(b"")
    assert paths.find_checkpoint("final") == tmp_path / "checkpoints" / "final.pt"
    write_json(paths.final_eval, {})
    assert paths.is_complete()
