"""One driver per engine. Each `run(ctx, job)` returns a `RunResult`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..progress import Timings


@dataclass
class RunResult:
    """What a driver produced for the muxer and the uploader."""

    seed: int
    width: int
    height: int
    frames: int
    fps: int
    video_y4m: Path | None = None  # h3 / fastwan: the clip (already upscaled when asked)
    audio_wav: Path | None = None
    video_mp4: Path | None = None  # fastvideo: fv-serve's own MP4
    upscaled: tuple[int, int, int] | None = None  # (factor, width, height)
    timings: Timings = field(default_factory=Timings)
    command: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Ctx:
    """What a driver needs from the worker for one job."""

    settings: Any  # flags.EngineSettings
    knobs: dict
    sidecars: Any  # sidecar.SidecarManager
    ports: dict[str, int]
    env: dict[str, dict[str, str]]  # per engine
    job_dir: Path
    store: Any = None  # r2.ObjectStore | None
    bucket: str | None = None
    log: Any = None
    report: Any = None  # progress.Reporter
    fv_serve_bin: str = "/opt/fastvideo-rs/bin/fv-serve"
    fv_config: str = "/opt/studio/configs/fastvideo.toml"
    fv_state_dir: str = "/workspace/fv-state"
    fv_weights: str = "/workspace/w"
    fv_fake: bool = False  # tests: a fake fv-serve already listens on the port
    models: list = field(default_factory=list)  # the catalog's models (fv-serve process groups); empty in unit tests
