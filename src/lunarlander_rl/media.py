"""Animated demonstrations of trained policies (GIF, and MP4 when ffmpeg is available).

Recording replays exactly the evaluation protocol: the same environment configuration, the
same episode seed and the same deterministic policy as the run's own evaluation. A recorded
episode therefore has the return the run reported for that seed, which
:func:`check_against_evaluation` verifies.

Which policies and episodes are shown is decided by fixed rules rather than by looking at
the results (see :func:`representative_run` and :func:`first_failed_episode`), so the
animations show typical behaviour, not the best of many attempts.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from lunarlander_rl.agents import build_agent
from lunarlander_rl.analysis.aggregate import completed_runs
from lunarlander_rl.analysis.plots import ALGORITHM_NAMES
from lunarlander_rl.analysis.tables import markdown_table
from lunarlander_rl.config import (
    ConfigError,
    ExperimentConfig,
    build_experiment_config,
    from_dict,
    load_config,
    load_yaml,
    to_dict,
)
from lunarlander_rl.envs import make_env
from lunarlander_rl.evaluation.metrics import LANDED, classify_outcome
from lunarlander_rl.seeding import seed_everything
from lunarlander_rl.tracking import RunPaths, read_json

Frame = npt.NDArray[np.uint8]

OUTCOME_LABELS = {
    "landed": "LANDED",
    "crashed": "CRASHED",
    "out_of_bounds": "OUT OF BOUNDS",
    "timeout": "TIME LIMIT",
    "terminated": "ENDED",
}
OUTCOME_COLORS = {"landed": (0, 158, 115), "timeout": (230, 159, 0)}
FAILURE_COLOR = (213, 94, 0)

GRID_CAPTION = (
    "How these were chosen",
    "",
    "Each panel is the run whose final test",
    "score is its method's median over 10 seeds:",
    "a typical policy, not the best one.",
    "",
    "All panels play the same held-out test",
    "episode, in real time. A return of 200 or",
    "more conventionally counts as solved.",
)


# ------------------------------------------------------------------- choosing what to show
def representative_run(experiment_dir: str | Path, variant: str) -> Path:
    """The run of ``variant`` whose final test return is closest to the median over its
    runs (the lower seed on a tie): a typical policy of the method, not its best."""
    candidates = []
    for paths in completed_runs(experiment_dir):
        config = load_config(paths.config)
        if config.variant == variant:
            score = read_json(paths.final_eval)["final"]["summary"]["mean_return"]
            candidates.append((config.seed, float(score), paths.root))
    if not candidates:
        raise FileNotFoundError(f"no completed runs of {variant!r} in {experiment_dir}")
    median = float(np.median([score for _, score, _ in candidates]))
    _, _, root = min(candidates, key=lambda c: (abs(c[1] - median), c[0]))
    return root


def first_failed_episode(run_dir: str | Path) -> int:
    """The lowest test-episode seed on which the run's final policy did not land."""
    episodes = read_json(RunPaths(Path(run_dir)).final_eval)["final"]["episodes"]
    for episode in sorted(episodes, key=lambda e: e["seed"]):
        if episode["outcome"] != LANDED:
            return int(episode["seed"])
    raise ValueError(f"the final policy of {run_dir} landed in every test episode")


# ------------------------------------------------------------------------------ recording
@dataclass(frozen=True)
class Recording:
    label: str
    seed: int
    frames: list[Frame]
    cumulative_returns: list[float]  # return so far, one per frame (0 for the first)
    episode_return: float
    outcome: str

    @property
    def length(self) -> int:
        return len(self.frames) - 1


def _config_with(run_dir: Path, overrides: Sequence[str]) -> ExperimentConfig:
    config = load_config(RunPaths(run_dir).config)
    return build_experiment_config(to_dict(config), overrides) if overrides else config


def record_episode(
    run_dir: str | Path,
    seed: int,
    *,
    checkpoint: str = "final",
    overrides: Sequence[str] = (),
    label: str | None = None,
) -> Recording:
    """Play one episode of a saved policy and keep every rendered frame."""
    paths = RunPaths(Path(run_dir))
    config = _config_with(paths.root, overrides)
    seed_everything(config.seed, torch_threads=config.torch_threads)
    env = make_env(config.env, render_mode="rgb_array")
    try:
        policy = build_agent(
            config.agent, env.observation_space, env.action_space,
            seed=config.seed, device=config.device,
        )  # fmt: skip
        policy.load(paths.find_checkpoint(checkpoint))
        policy.seed_eval(seed)
        obs, _ = env.reset(seed=seed)
        frames: list[Frame] = [np.asarray(env.render(), dtype=np.uint8)]
        returns = [0.0]
        while True:
            action = policy.predict(obs, deterministic=config.eval.deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            returns.append(returns[-1] + float(reward))
            frames.append(np.asarray(env.render(), dtype=np.uint8))
            if terminated or truncated:
                break
        env_id = env.spec.id if env.spec is not None else None
        outcome = classify_outcome(env_id, obs, float(reward), terminated, truncated)
    finally:
        env.close()
    name = label or ALGORITHM_NAMES.get(config.variant, config.variant)
    return Recording(name, seed, frames, returns, returns[-1], outcome)


def check_against_evaluation(recording: Recording, run_dir: str | Path) -> None:
    """Fail unless the recorded episode has the return the run's own final evaluation
    reported for the same seed (only meaningful without overrides)."""
    episodes = read_json(RunPaths(Path(run_dir)).final_eval)["final"]["episodes"]
    reported = {e["seed"]: e for e in episodes}.get(recording.seed)
    if reported is None:
        raise ValueError(f"seed {recording.seed} is not one of the run's test episodes")
    if not math.isclose(reported["episode_return"], recording.episode_return, abs_tol=1e-6):
        raise AssertionError(
            f"recorded return {recording.episode_return} differs from the evaluated "
            f"{reported['episode_return']} for seed {recording.seed} of {run_dir}"
        )


# ----------------------------------------------------------------------------- drawing
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def annotate(frame: Frame, recording: Recording, index: int, scale: float) -> Frame:
    """Downscale a frame and draw the method, step and running return on it; after the
    episode ends, add a badge with the outcome and final return."""
    image = Image.fromarray(frame)
    if scale != 1.0:
        size = (round(image.width * scale), round(image.height * scale))
        image = image.resize(size, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    unit = image.height / 200
    step = min(index, recording.length)
    pad = round(6 * unit)
    draw.text((pad, pad), recording.label, fill=(255, 255, 255), font=_font(round(13 * unit)))
    draw.text(
        (pad, pad + round(16 * unit)),
        f"step {step:4d}   return {recording.cumulative_returns[step]:7.1f}",
        fill=(225, 225, 225),
        font=_font(round(10 * unit)),
    )
    if index >= recording.length:
        text = f"{OUTCOME_LABELS[recording.outcome]}  {recording.episode_return:+.0f}"
        font = _font(round(13 * unit))
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        width, height = right - left + 2 * pad, bottom - top + 2 * pad
        x, y = image.width - width - pad, pad
        color = OUTCOME_COLORS.get(recording.outcome, FAILURE_COLOR)
        draw.rounded_rectangle((x, y, x + width, y + height), radius=pad, fill=color)
        draw.text((x + pad - left, y + pad - top), text, fill=(255, 255, 255), font=font)
    return np.asarray(image)


def animate(
    recordings: Sequence[Recording],
    *,
    columns: int,
    scale: float,
    frame_step: int,
    hold_frames: int,
    title: str | None = None,
    caption: Sequence[str] = (),
) -> list[Frame]:
    """Tile recordings into a grid, one animation frame per ``frame_step`` simulation
    steps. Finished episodes keep their last frame and outcome badge; the final frame is
    held for ``hold_frames`` more frames. ``caption`` lines fill the first empty cell."""
    longest = max(r.length for r in recordings)
    indices = [*range(0, longest + 1, frame_step)]
    if indices[-1] != longest:
        indices.append(longest)
    indices += [longest] * hold_frames
    rows = math.ceil(len(recordings) / columns)
    sample = annotate(recordings[0].frames[0], recordings[0], 0, scale)
    cell_h, cell_w = sample.shape[:2]
    title_h = round(cell_h * 0.16) if title else 0
    canvas_shape = (title_h + rows * cell_h, columns * cell_w, 3)

    # Title and caption never change, so they are drawn once onto a background.
    background = Image.fromarray(np.full(canvas_shape, 255, dtype=np.uint8))
    draw = ImageDraw.Draw(background)
    if title:
        draw.text((round(cell_w * 0.03), round(title_h * 0.2)), title, fill=(30, 30, 30),
                  font=_font(round(title_h * 0.55)))  # fmt: skip
    if caption and len(recordings) < rows * columns:
        r, c = divmod(len(recordings), columns)
        x, y = c * cell_w + round(cell_w * 0.05), title_h + r * cell_h + round(cell_h * 0.08)
        size = round(cell_h * 0.058)
        for i, line in enumerate(caption):
            draw.text((x, y + round(i * size * 1.45)), line, fill=(60, 60, 60), font=_font(size))
    base = np.asarray(background)

    output = []
    for index in indices:
        canvas = base.copy()
        for k, recording in enumerate(recordings):
            r, c = divmod(k, columns)
            frame = recording.frames[min(index, recording.length)]
            tile = annotate(frame, recording, index, scale)
            canvas[title_h + r * cell_h : title_h + (r + 1) * cell_h,
                   c * cell_w : (c + 1) * cell_w] = tile  # fmt: skip
        output.append(canvas)
    return output


# ----------------------------------------------------------------------------- writing
def write_gif(frames: Sequence[Frame], path: Path, fps: float) -> Path:
    """Write an optimised, looping GIF with one shared palette."""
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(f) for f in frames]
    # LunarLander's flat-shaded graphics need few colours; one palette for all frames
    # keeps the file small and avoids flicker between frames. It is fitted to a middle and
    # the last frame together, so the outcome badges drawn at the end keep their colours.
    middle, last = images[len(images) // 2], images[-1]
    sample = Image.new("RGB", (middle.width, middle.height + last.height))
    sample.paste(middle, (0, 0))
    sample.paste(last, (0, middle.height))
    palette = sample.quantize(colors=96, method=Image.Quantize.MEDIANCUT)
    quantised = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    quantised[0].save(
        path,
        save_all=True,
        append_images=quantised[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=True,
        disposal=1,
    )
    return path


def write_mp4(frames: Sequence[Frame], path: Path, fps: float) -> Path | None:
    """Write an H.264 MP4, or return None if imageio-ffmpeg is not installed."""
    if importlib.util.find_spec("imageio_ffmpeg") is None:
        return None
    import imageio.v3 as iio

    path.parent.mkdir(parents=True, exist_ok=True)
    # H.264 needs even dimensions.
    even = [f[: f.shape[0] // 2 * 2, : f.shape[1] // 2 * 2] for f in frames]
    iio.imwrite(path, np.stack(even), fps=fps, codec="libx264", macro_block_size=1)
    return path


# -------------------------------------------------------------------------- the showcase
@dataclass(frozen=True)
class ShowcaseSpec:
    """Which animations to make; see configs/media/showcase.yaml."""

    name: str
    runs: str
    methods: tuple[str, ...]
    # Episode shown in the grids: the first held-out test episode by default.
    episode_seed: int = 10_000
    # Method whose first failed test episode is shown in the failure animation.
    failure_method: str = "reinforce"
    wind: dict[str, Any] = dataclasses.field(default_factory=dict)
    frame_step: int = 2
    scale: float = 0.5
    columns: int = 3
    hold_seconds: float = 1.5


def load_showcase(path: str | Path) -> ShowcaseSpec:
    spec = from_dict(ShowcaseSpec, load_yaml(path), where="showcase")
    if spec.failure_method not in spec.methods:
        raise ConfigError("showcase: failure_method must be one of methods")
    return spec


def _overrides(mapping: dict[str, Any]) -> list[str]:
    return [f"{key}={json.dumps(value)}" for key, value in mapping.items()]


def _gif_and_mp4(frames: Sequence[Frame], stem: Path, fps: float) -> list[Path]:
    written = [write_gif(frames, stem.with_suffix(".gif"), fps)]
    mp4 = write_mp4(frames, stem.with_suffix(".mp4"), fps)
    if mp4 is not None:
        written.append(mp4)
    return written


def build_showcase(spec: ShowcaseSpec, runs_dir: Path, out_dir: Path) -> list[Path]:
    """Make every animation of the showcase and a Markdown manifest describing them."""
    fps = 50 / spec.frame_step  # LunarLander simulates 50 steps per second.
    hold = round(spec.hold_seconds * fps)
    runs = {m: representative_run(runs_dir, m) for m in spec.methods}
    rows: list[dict[str, Any]] = []
    written: list[Path] = []

    def note(animation: str, method: str, recording: Recording, rule: str) -> None:
        rows.append(
            {
                "animation": animation,
                "method": recording.label,
                "run": f"{runs[method].parent.name}/{runs[method].name}",
                "episode": recording.seed,
                "outcome": OUTCOME_LABELS[recording.outcome].lower(),
                "return": recording.episode_return,
                "rule": rule,
            }
        )

    # Calm: every method on the same held-out test episode.
    calm = []
    for method in spec.methods:
        recording = record_episode(runs[method], spec.episode_seed)
        check_against_evaluation(recording, runs[method])
        calm.append(recording)
        note("landing_grid, " + method, method, recording, "median run, first test episode")
    title = f"Test episode {spec.episode_seed}: each method's median run, no wind"
    written += _gif_and_mp4(
        animate(calm, columns=spec.columns, scale=spec.scale, frame_step=spec.frame_step,
                hold_frames=hold, title=title, caption=GRID_CAPTION),
        out_dir / "landing_grid", fps,
    )  # fmt: skip
    for method, recording in zip(spec.methods, calm, strict=True):
        frames = animate([recording], columns=1, scale=0.75, frame_step=spec.frame_step,
                         hold_frames=hold)  # fmt: skip
        written.append(write_gif(frames, out_dir / f"{method}.gif", fps))

    # A failure: the first test episode the chosen method's median run did not land.
    method = spec.failure_method
    try:
        failed_seed: int | None = first_failed_episode(runs[method])
    except ValueError:
        failed_seed = None  # the policy landed in every test episode: nothing to show
    if failed_seed is not None:
        failure = record_episode(runs[method], failed_seed)
        check_against_evaluation(failure, runs[method])
        note(f"{method}_failure", method, failure, "median run, first test episode it failed")
        written += _gif_and_mp4(
            animate([failure], columns=1, scale=0.75, frame_step=spec.frame_step,
                    hold_frames=hold),
            out_dir / f"{method}_failure",
            fps,
        )  # fmt: skip

    # Wind the policies never trained in, same episode seed.
    if spec.wind:
        windy = [
            record_episode(runs[m], spec.episode_seed, overrides=_overrides(spec.wind))
            for m in spec.methods
        ]
        for m, recording in zip(spec.methods, windy, strict=True):
            note("wind_grid, " + m, m, recording, "median run, first test episode, wind")
        power = spec.wind.get("env.kwargs.wind_power", "?")
        written += _gif_and_mp4(
            animate(windy, columns=spec.columns, scale=spec.scale, frame_step=spec.frame_step,
                    hold_frames=hold, caption=GRID_CAPTION,
                    title=f"Test episode {spec.episode_seed} with wind {power} "
                          "(never seen in training)"),
            out_dir / "wind_grid", fps,
        )  # fmt: skip

    written.append(_write_manifest(rows, out_dir / f"{spec.name}.md"))
    return written


def _write_manifest(rows: Sequence[dict[str, Any]], path: Path) -> Path:
    table = markdown_table(
        pd.DataFrame(rows),
        [
            ("Animation", lambda r: r["animation"]),
            ("Method", lambda r: r["method"]),
            ("Run", lambda r: r["run"]),
            ("Episode seed", lambda r: str(r["episode"])),
            ("Outcome", lambda r: r["outcome"]),
            ("Return", lambda r: f"{r['return']:.1f}"),
            ("Chosen as", lambda r: r["rule"]),
        ],
        align=["left", "left", "left", "right", "left", "right", "left"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(table, encoding="utf-8", newline="\n")
    return path
