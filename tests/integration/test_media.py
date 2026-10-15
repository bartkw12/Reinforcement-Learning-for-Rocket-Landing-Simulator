"""Recording, drawing and writing the animated demonstrations."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from lunarlander_rl import media
from lunarlander_rl.config import ConfigError, from_dict
from lunarlander_rl.sweep import expand_experiment, run_sweep
from lunarlander_rl.tracking import write_json

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def experiment(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("runs")
    spec = {
        "name": "bench",
        "seeds": [0, 1, 2],
        "base": {
            "train": {"total_steps": 200},
            "eval": {"interval_steps": 200, "episodes": 1, "final_episodes": 3},
        },
        # The random agent stands in for trained ones; only the mechanics are tested.
        "variants": [
            {"label": "alpha", "agent": {"name": "random"}},
            {"label": "beta", "agent": {"name": "random"}},
        ],
    }
    assert all(o.status == "completed" for o in run_sweep(expand_experiment(spec, root)))
    return root / "bench"


def test_representative_run_is_the_median_one(experiment: Path, tmp_path: Path) -> None:
    # Give three fake runs distinct scores: the middle one is representative.
    for seed, score in ((0, 50.0), (1, 250.0), (2, 120.0)):
        run = tmp_path / "exp" / "m" / f"seed_{seed}"
        run.mkdir(parents=True)
        (run / "config.yaml").write_text(f"agent: {{name: random}}\nlabel: m\nseed: {seed}\n")
        write_json(run / "final_eval.json", {"final": {"summary": {"mean_return": score}}})
    assert media.representative_run(tmp_path / "exp", "m") == tmp_path / "exp" / "m" / "seed_2"
    # The random agent's evaluation does not depend on its training seed, so all three
    # runs tie and the lowest seed is chosen.
    assert media.representative_run(experiment, "alpha") == experiment / "alpha" / "seed_0"
    with pytest.raises(FileNotFoundError):
        media.representative_run(experiment, "gamma")


def test_recording_matches_the_runs_own_evaluation(experiment: Path) -> None:
    run = experiment / "alpha" / "seed_0"
    recording = media.record_episode(run, 10_000)
    media.check_against_evaluation(recording, run)
    assert len(recording.frames) == recording.length + 1 == len(recording.cumulative_returns)
    assert recording.frames[0].shape == (400, 600, 3)
    assert recording.cumulative_returns[-1] == recording.episode_return
    with pytest.raises(ValueError, match="not one of"):
        media.check_against_evaluation(media.record_episode(run, 5), run)


def test_first_failed_episode(experiment: Path) -> None:
    # A random policy essentially never lands, so its first test episode fails.
    assert media.first_failed_episode(experiment / "alpha" / "seed_0") == 10_000


def test_animation_and_files(experiment: Path, tmp_path: Path) -> None:
    recordings = [
        media.record_episode(experiment / v / "seed_0", 10_000) for v in ("alpha", "beta")
    ]
    frames = media.animate(
        recordings, columns=3, scale=0.5, frame_step=2, hold_frames=5,
        title="title", caption=("a", "b"),
    )  # fmt: skip
    longest = max(r.length for r in recordings)
    assert len(frames) == len(range(0, longest + 1, 2)) + (longest % 2 != 0) + 5
    height, width = frames[0].shape[:2]
    assert width == 900 and height > 200
    # Held frames repeat the last one.
    np.testing.assert_array_equal(frames[-1], frames[-2])

    gif = media.write_gif(frames, tmp_path / "a.gif", fps=25)
    with Image.open(gif) as image:
        assert (
            image.n_frames > 1  # type: ignore[attr-defined]
            and image.info["duration"] == 40
            and image.info["loop"] == 0
        )
    mp4 = media.write_mp4(frames, tmp_path / "a.mp4", fps=25)
    if mp4 is not None:
        assert mp4.stat().st_size > 0


def test_showcase(experiment: Path, tmp_path: Path) -> None:
    spec = from_dict(
        media.ShowcaseSpec,
        {
            "name": "demo",
            "runs": str(experiment),
            "methods": ["alpha", "beta"],
            "failure_method": "beta",
            "wind": {"env.kwargs.enable_wind": True},
        },
    )
    written = media.build_showcase(spec, experiment, tmp_path)
    names = sorted(p.name for p in written)
    assert {"landing_grid.gif", "alpha.gif", "beta.gif", "beta_failure.gif", "wind_grid.gif",
            "demo.md"} <= set(names)  # fmt: skip
    with Image.open(tmp_path / "social_preview.png") as card:
        assert card.size == (1280, 640)
    manifest = (tmp_path / "demo.md").read_text()
    assert manifest.count("\n") == 2 + 2 + 1 + 2  # header, 2 calm, 1 failure, 2 wind


def test_showcase_spec_validation(tmp_path: Path) -> None:
    good: dict[str, Any] = {"name": "x", "runs": "runs/x", "methods": ["a"], "failure_method": "a"}
    path = tmp_path / "s.yaml"
    path.write_text("name: x\nruns: runs/x\nmethods: [a]\nfailure_method: b\n")
    with pytest.raises(ConfigError, match="failure_method"):
        media.load_showcase(path)
    assert from_dict(media.ShowcaseSpec, good).episode_seed == 10_000


def test_shipped_showcase_spec_is_valid() -> None:
    spec = media.load_showcase(REPO_ROOT / "configs" / "media" / "showcase.yaml")
    assert spec.methods == ("sb3_ppo", "dqn", "sb3_dqn", "q_learning", "reinforce")
    assert spec.episode_seed == 10_000
