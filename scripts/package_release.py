"""Package the run records and benchmark checkpoints as release assets.

Writes, under dist/:
  lunarlander-rl-v<version>-run-records.zip     every run's config, provenance, training log,
                                                evaluations and robustness re-evaluations
  lunarlander-rl-v<version>-benchmark-checkpoints.zip
                                                the main benchmark's final and best policies,
                                                with their configs, so they can be evaluated
  SHA256SUMS

Archives are reproducible: files are added in sorted order with fixed timestamps.

Example:
    python scripts/package_release.py
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from collections.abc import Iterable
from pathlib import Path

import lunarlander_rl
from lunarlander_rl.reports import REPORTS

RECORD_FILES = ("config.yaml", "meta.json", "train_episodes.csv", "eval.csv", "final_eval.json")
FIXED_TIME = (2026, 1, 1, 0, 0, 0)


def run_dirs(runs_root: Path, experiment: str) -> list[Path]:
    return sorted(p.parent for p in (runs_root / experiment).glob("*/seed_*/final_eval.json"))


def write_zip(path: Path, files: Iterable[Path], root: Path) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for file in sorted(files):
            info = zipfile.ZipInfo(file.relative_to(root).as_posix(), date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, file.read_bytes())
            count += 1
    return count, path.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--out-dir", default="dist")
    args = parser.parse_args()
    runs_root, out_dir = Path(args.runs_root), Path(args.out_dir)
    root = runs_root.parent
    version = lunarlander_rl.__version__

    records: list[Path] = []
    for report in REPORTS:
        if report.kind != "sweep":
            continue
        for run in run_dirs(runs_root, report.name):
            records += [run / name for name in RECORD_FILES]
            records += sorted((run / "evaluations").rglob("*.json"))
    checkpoints: list[Path] = []
    for run in run_dirs(runs_root, "main_benchmark"):
        checkpoints += [run / "config.yaml", *sorted((run / "checkpoints").iterdir())]

    archives = {
        out_dir / f"lunarlander-rl-v{version}-run-records.zip": records,
        out_dir / f"lunarlander-rl-v{version}-benchmark-checkpoints.zip": checkpoints,
    }
    sums = []
    for path, files in archives.items():
        count, size = write_zip(path, files, root)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sums.append(f"{digest}  {path.name}")
        print(f"{path}: {count} files, {size / 1e6:.1f} MB")
    (out_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8", newline="\n")
    print(f"{out_dir / 'SHA256SUMS'}")


if __name__ == "__main__":
    main()
