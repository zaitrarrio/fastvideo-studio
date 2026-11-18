"""h3-rs: `h3 serve` for the tasks it advertises, `h3 generate` /
`h3 regen2k` per job for the rest (design §1)."""

from __future__ import annotations

import base64
import random
from pathlib import Path
from typing import Any

from .. import flags, frames as fr, httpc, media as md, mux
from ..catalog import Job
from ..errors import JobError
from ..sidecar import Sidecar, http_info_probe
from . import Ctx, RunResult
from .common import run_cli, run_serve_job

ENGINE = "h3"


def serve_supports(info: dict | None, task: str) -> bool:
    """`/api/info.tasks` (the parallel h3-rs change) lists what serve takes;
    without it, text-to-video only."""
    if task == "t2v":
        return True
    tasks = (info or {}).get("tasks")
    return isinstance(tasks, list) and flags.H3_TASK.get(task) in tasks


SERVE_BODY_MAX = 64 * 1024 * 1024  # h3 serve's cap on a body with media (inputs.max_body_bytes)
_B64_RATIO = 4 / 3


def _b64_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def serve_media_bytes(local: dict[str, Any]) -> int:
    """The decoded size of everything `serve_media_fields` would send; the
    base64 body is 4/3 of it. Over `SERVE_BODY_MAX` the job runs through the CLI."""
    n = 0
    for key in ("first_image", "last_image"):
        if local.get(key):
            n += Path(local[key]).stat().st_size
    for _k, path, audio in local.get("references", []):
        n += Path(path).stat().st_size + (Path(audio).stat().st_size if audio else 0)
    if local.get("continue"):
        y4m, wav = local["continue"]
        n += Path(y4m).stat().st_size + (Path(wav).stat().st_size if wav else 0)
    return n


def serve_fits(local: dict[str, Any]) -> bool:
    return serve_media_bytes(local) * _B64_RATIO < SERVE_BODY_MAX * 0.95


def serve_media_fields(task: str, local: dict[str, Any]) -> dict[str, Any]:
    """`h3 serve`'s per-request inputs (h3-rs serve.rs): `task`, `first_image`
    / `last_image` as {b64, mime}, `refs` as [{kind, b64, mime?, audio_b64?}]
    (video as Y4M bytes, audio as WAV), `continue` as {b64, audio_b64?}."""
    out: dict[str, Any] = {"task": flags.H3_TASK[task]}
    for key in ("first_image", "last_image"):
        if local.get(key):
            path = Path(local[key])
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            out[key] = {"b64": _b64_file(path), "mime": mime}
    refs = []
    for kind, path, audio in local.get("references", []):
        path = Path(path)
        if kind == "image":
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            refs.append({"kind": "image", "b64": _b64_file(path), "mime": mime})
        elif kind == "video":
            r = {"kind": "video", "b64": _b64_file(path), "mime": "video/x-yuv4mpeg"}
            if audio:
                r["audio_b64"] = _b64_file(Path(audio))
            refs.append(r)
        else:
            refs.append({"kind": "audio", "b64": _b64_file(path), "mime": "audio/wav"})
    if refs:
        out["refs"] = refs
    if local.get("continue"):
        y4m, wav = local["continue"]
        c = {"b64": _b64_file(Path(y4m))}
        if wav:
            c["audio_b64"] = _b64_file(Path(wav))
        out["continue"] = c
    return out


def media_kind(lm: md.LocalMedia) -> str:
    k = lm.kind
    if k in ("image", "video", "audio"):
        return k
    ext = lm.path.suffix.lower()
    if ext in (".y4m", ".mp4", ".mov", ".webm", ".mkv"):
        return "video"
    if ext in (".wav", ".mp3", ".m4a", ".flac", ".ogg"):
        return "audio"
    if ext in (".png", ".jpg", ".jpeg", ".webp", ".ppm"):
        return "image"
    raise JobError("validate", f"{lm.path.name}: cannot tell image / video / audio from {lm.mime}")


def prepare_media(ctx: Ctx, job: Job) -> dict[str, Any]:
    """Inputs → local files in the forms `h3 generate` takes."""
    out: dict[str, Any] = {}
    mdir = ctx.job_dir / "in"
    fps = job.fps_for()
    for key in ("first_image", "last_image"):
        if job.inputs.get(key):
            lm = md.fetch_media(job.inputs[key], mdir, key, ctx.store, ctx.bucket)
            if media_kind(lm) != "image":
                raise JobError("validate", f"{key} must be an image")
            out[key] = lm.path
    refs = []
    for i, m in enumerate(job.inputs.get("references", [])):
        lm = md.fetch_media(m, mdir, f"ref{i}", ctx.store, ctx.bucket)
        kind = media_kind(lm)
        if kind == "image":
            refs.append(("image", lm.path, None))
        elif kind == "video":
            y4m = lm.path if lm.path.suffix == ".y4m" else mux.to_y4m(lm.path, mdir / f"ref{i}.y4m", fps=fps)
            wav = None
            p = mux.probe(lm.path) if lm.path.suffix != ".y4m" else None
            if p and p.audio:
                wav = mux.to_wav(lm.path, mdir / f"ref{i}.wav")
            refs.append(("video", y4m, wav))
        else:
            wav = lm.path if lm.path.suffix == ".wav" else mux.to_wav(lm.path, mdir / f"ref{i}.wav")
            refs.append(("audio", wav, None))
    if refs:
        out["references"] = refs
    if job.inputs.get("video"):  # extend → --continue
        lm = md.fetch_media(job.inputs["video"], mdir, "continue", ctx.store, ctx.bucket)
        canvas = (job.width, job.height)
        y4m = mux.to_y4m(lm.path, mdir / "continue.y4m", canvas[0], canvas[1], fps, last_frames=flags.CONTINUE_FRAMES)
        wav = None
        p = mux.probe(lm.path)
        if p and p.audio:
            wav = mux.to_wav(lm.path, mdir / "continue.wav")
        out["continue"] = (y4m, wav)
    if job.inputs.get("draft"):
        lm = md.fetch_media(job.inputs["draft"], mdir, "draft", ctx.store, ctx.bucket)
        out["draft"] = lm.path if lm.path.suffix == ".y4m" else mux.to_y4m(lm.path, mdir / "draft.y4m", fps=fps)
    return out


def run(ctx: Ctx, job: Job) -> RunResult:
    s = ctx.settings
    seed = job.seed if job.seed is not None else random.randrange(1, 2**31)
    local = prepare_media(ctx, job)
    out = ctx.job_dir / "out.y4m"
    if job.task == "regen2k":
        return _run_cli(ctx, job, flags.h3_regen2k_argv(job, s, ctx.knobs, local["draft"], out, seed), out, seed)
    command = flags.command_line(flags.h3_generate_argv(job, s, ctx.knobs, local, Path("out.y4m"), seed))
    if job.model.cli_only:
        return _run_cli(ctx, job, flags.h3_generate_argv(job, s, ctx.knobs, local, out, seed), out, seed, command)
    port = ctx.ports[ENGINE]
    base = f"http://127.0.0.1:{port}"
    key = flags.h3_serve_key(job, s, ctx.knobs)

    def make() -> Sidecar:
        return Sidecar(ENGINE, key, flags.h3_serve_argv(job, s, ctx.knobs, port), ctx.env[ENGINE], http_info_probe(f"{base}/api/info"))

    # A sidecar already up for this recipe answers whether it takes the task.
    cur = ctx.sidecars.get(ENGINE)
    info = None
    if cur is not None and cur.alive() and cur.key == key:
        try:
            info = httpc.get_json(f"{base}/api/info", timeout=5)
        except Exception:  # noqa: BLE001
            info = None
    elif job.task != "t2v":
        sc, load_s = ctx.sidecars.ensure(ENGINE, key, make, job.model.vram_gib, _loading(ctx))
        ctx.report("loading", force=True)
        info = httpc.get_json(f"{base}/api/info", timeout=5)
        if not serve_supports(info, job.task):
            # serve cannot take it: stop it so the CLI has the VRAM.
            sc.stop()
            ctx.sidecars.sidecars.pop(ENGINE, None)
    if not serve_supports(info, job.task) or not serve_fits(local):
        return _run_cli(ctx, job, flags.h3_generate_argv(job, s, ctx.knobs, local, out, seed), out, seed, command)
    sc, load_s = ctx.sidecars.ensure(ENGINE, key, make, job.model.vram_gib, _loading(ctx))
    body: dict[str, Any] = {"prompt": job.prompt, "width": job.width, "height": job.height, "frames": job.frames,
                            "seed": seed, "upscale": job.upscale if job.upscale else False}
    body.update(serve_media_fields(job.task, local))
    clip = run_serve_job(base, sc, body, ctx.job_dir, ctx.report, want_audio=job.model.audio)
    ctx.report("muxing", "frames → y4m", force=True)
    fps = job.fps_for()
    fr.rgb_to_y4m(clip.rgb, clip.width, clip.height, fps, out)
    res = RunResult(seed, clip.width, clip.height, clip.frames, fps, video_y4m=out, audio_wav=clip.audio_wav,
                    timings=clip.timings, command=command)
    res.timings.load_s = load_s
    if clip.hd_jpegs:
        hd = ctx.job_dir / "out-hd.y4m"
        fr.jpegs_to_y4m(clip.hd_jpegs, fps, hd)
        info_hd = fr.read_y4m_info(hd)
        res.video_y4m = hd
        res.upscaled = (clip.up_factor or job.upscale, info_hd.width, info_hd.height)
    return res


def _loading(ctx: Ctx):
    def f(stage: str, t: float) -> None:
        ctx.report("loading", f"{t:.0f} s")
    return f


def _run_cli(ctx: Ctx, job: Job, argv: list[str], out: Path, seed: int, command: str | None = None) -> RunResult:
    # The CLI needs the whole GPU: no resident sidecar beside it.
    ctx.sidecars.stop_all()
    ctx.report("loading", "h3 (per-job CLI)", force=True)
    _, t = run_cli(argv, ctx.env[ENGINE], ctx.report, ctx.log or (lambda s: None), cwd=ctx.job_dir)
    if not out.is_file():
        raise JobError("engine", f"h3 wrote no {out.name}")
    info = fr.read_y4m_info(out)
    wav = out.with_suffix(".wav")
    res = RunResult(seed, info.width, info.height, info.frames, job.fps_for(), video_y4m=out,
                    audio_wav=wav if wav.is_file() else None, timings=t,
                    command=command or flags.command_line([a if a != str(out) else out.name for a in argv]))
    if job.upscale and (info.width, info.height) != (job.width, job.height):
        res.width, res.height = job.width, job.height
        res.upscaled = (job.upscale, info.width, info.height)
    return res
