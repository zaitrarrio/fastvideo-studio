"""The jobs.md progress record and timings, from whatever an engine reports."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

STAGES = ("queued", "staging", "loading", "encoding", "preparing", "denoising", "decoding", "upscaling",
          "muxing", "uploading", "done", "failed")
Progress = Callable[[dict], None]


@dataclass
class Timings:
    stage_s: float = 0.0
    load_s: float = 0.0
    encode_s: float = 0.0
    denoise_s: float = 0.0
    decode_s: float = 0.0
    upscale_s: float = 0.0
    mux_s: float = 0.0
    total_s: float = 0.0
    steps_s: list[float] = field(default_factory=list)

    def to_json(self) -> dict:
        d = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items() if k != "steps_s"}
        d["steps_s"] = [round(s, 3) for s in self.steps_s]
        return d


class Reporter:
    """Sends progress records (rate-limited) and remembers the last one."""

    def __init__(self, send: Progress | None, engine: str, model: str, min_interval_s: float = 0.5):
        self.send = send
        self.engine = engine
        self.model = model
        self.t0 = time.time()
        self.last: dict = {}
        self.last_sent = 0.0
        self.min_interval_s = min_interval_s

    def __call__(self, stage: str, detail: str | None = None, *, step: int | None = None, total_steps: int | None = None,
                 frames: int | None = None, total_frames: int | None = None, force: bool = False) -> None:
        assert stage in STAGES, stage
        rec: dict[str, Any] = {"stage": stage, "elapsed_s": round(time.time() - self.t0, 1),
                               "engine": self.engine, "model": self.model}
        if detail:
            rec["detail"] = detail
        for k, v in (("step", step), ("total_steps", total_steps), ("frames", frames), ("total_frames", total_frames)):
            if v is not None:
                rec[k] = v
        changed = {k: v for k, v in rec.items() if k != "elapsed_s"} != {k: v for k, v in self.last.items() if k != "elapsed_s"}
        stage_changed = stage != self.last.get("stage")
        self.last = rec
        now = time.time()
        # A new stage always goes out; detail updates within a stage are rate-limited.
        if self.send and (force or stage_changed or (changed and now - self.last_sent >= self.min_interval_s) or now - self.last_sent >= 5):
            self.last_sent = now
            try:
                self.send(rec)
            except Exception:  # noqa: BLE001 — a lost progress update is not a failed job
                pass
