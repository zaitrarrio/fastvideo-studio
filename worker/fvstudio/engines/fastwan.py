"""fastwan-rs: `fastwan serve` for clips, `fastwan generate --causal` per
job for long videos (Self-Forcing)."""

from __future__ import annotations

import random
from pathlib import Path

from .. import flags, frames as fr
from ..catalog import Job
from ..errors import JobError
from ..sidecar import Sidecar, http_info_probe
from . import Ctx, RunResult
from .common import run_cli, run_serve_job

ENGINE = "fastwan"


def run(ctx: Ctx, job: Job) -> RunResult:
    s = ctx.settings
    seed = job.seed if job.seed is not None else random.randrange(1, 2**31)
    out = ctx.job_dir / "out.y4m"
    command = flags.command_line(flags.fastwan_generate_argv(job, s, ctx.knobs, Path("out.y4m"), seed))
    if job.model.cli_only or job.task == "long":
        ctx.sidecars.stop_all()
        ctx.report("loading", "fastwan (per-job CLI)", force=True)
        argv = flags.fastwan_generate_argv(job, s, ctx.knobs, out, seed)
        _, t = run_cli(argv, ctx.env[ENGINE], ctx.report, ctx.log or (lambda s: None), cwd=ctx.job_dir)
        if not out.is_file():
            raise JobError("engine", "fastwan wrote no out.y4m")
        info = fr.read_y4m_info(out)
        res = RunResult(seed, info.width, info.height, info.frames, job.fps_for(), video_y4m=out, timings=t, command=command)
        if job.upscale and (info.width, info.height) != (job.width, job.height):
            res.width, res.height = job.width, job.height
            res.upscaled = (job.upscale, info.width, info.height)
        return res
    port = ctx.ports[ENGINE]
    base = f"http://127.0.0.1:{port}"
    key = flags.fastwan_serve_key(job, s, ctx.knobs)

    def make() -> Sidecar:
        return Sidecar(ENGINE, key, flags.fastwan_serve_argv(job, s, ctx.knobs, port), ctx.env[ENGINE], http_info_probe(f"{base}/api/info"))

    sc, load_s = ctx.sidecars.ensure(ENGINE, key, make, job.model.vram_gib, lambda st, t: ctx.report("loading", f"{t:.0f} s"))
    body = {"prompt": job.prompt, "width": job.width, "height": job.height, "frames": job.frames, "seed": seed,
            "upscale": job.upscale}
    clip = run_serve_job(base, sc, body, ctx.job_dir, ctx.report, want_audio=False)
    ctx.report("muxing", "frames → y4m", force=True)
    fps = job.fps_for()
    fr.rgb_to_y4m(clip.rgb, clip.width, clip.height, fps, out)
    res = RunResult(seed, job.width, job.height, clip.frames, fps, video_y4m=out, timings=clip.timings, command=command)
    res.timings.load_s = load_s
    if job.upscale:
        res.upscaled = (job.upscale, clip.width, clip.height)
    return res
