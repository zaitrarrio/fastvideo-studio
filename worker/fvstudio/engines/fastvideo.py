"""fastvideo-rs: one resident `fv-serve` (swap = true, every variant
non-resident) driven through its native jobs API `/fv/v1/jobs`."""

from __future__ import annotations

import re
import shlex
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Any

from .. import flags, httpc, media as md
from ..catalog import Job
from ..errors import JobError
from ..progress import Timings
from ..sidecar import Sidecar, ping_probe
from . import Ctx, RunResult

ENGINE = "fastvideo"
KEY = "fv-serve"
POLL_S = 1.0
PRESIGN_S = 3600


def media_url(ctx: Ctx, m, name: str) -> str:
    """https URLs pass through; R2 uploads become presigned GETs; inline
    bytes a data URI (fv-serve takes both)."""
    if m.kind == "url":
        return m.value
    if m.kind == "r2":
        if ctx.store is None or not ctx.bucket:
            raise JobError("validate", f"{name}: r2 uploads need the Studio bucket (no R2 credentials)")
        return ctx.store.presign_get(ctx.bucket, m.value, PRESIGN_S)
    lm = md.fetch_media(m, ctx.job_dir / "in", name, ctx.store, ctx.bucket)
    return md.to_data_uri(lm)


def media_urls(ctx: Ctx, job: Job) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in job.inputs.items():
        if k == "references":
            out[k] = [media_url(ctx, m, f"ref{i}") for i, m in enumerate(v)]
        else:
            out[k] = media_url(ctx, v, k)
    return out


def process_of(model) -> str:
    """The fv-serve process a model runs in (`recipe.process`, else the
    variant alone): fv-serve refuses one process for models that need
    different process-wide settings (H3 wants FASTVIDEO_BF16_ACT=1, LTX does
    not; FastWan 2.1's VSA flag is process-wide), and serves every model of
    one family from one process with `[engine] swap = true` (fastvideo-rs's
    runpod-family-{h3,ltx,streaming}.toml)."""
    return model.recipe.get("process") or model.recipe.get("variant", "")


def group_members(models: list, variant: str) -> list[str]:
    """The variants sharing `variant`'s process, the requested one first;
    `[variant]` alone without a catalog (unit tests)."""
    out = [variant]
    mine = next((m for m in models if m.engine == "fastvideo" and m.recipe.get("variant") == variant), None)
    if mine is not None:
        process = process_of(mine)
        for m in models:
            if m.engine == "fastvideo" and process_of(m) == process:
                v = m.recipe.get("variant", "")
                if v and v not in out:
                    out.append(v)
    return out


def group_config(base: str, members: list[str], resident: str) -> str:
    """The worker's config with the process group's models only, `resident`
    marked `resident = true` (fv-serve refuses a config where nothing loads
    before readiness) and the others `resident = false` (loaded on demand,
    `[engine] swap = true` in the base)."""
    head, *blocks = base.split("[[models]]")
    keep: list[str] = []
    found = False
    for b in blocks:
        m = re.search(r'variant\s*=\s*"([^"]+)"', b)
        if not m or m.group(1) not in members:
            continue
        want = "resident = true" if m.group(1) == resident else "resident = false"
        found = found or m.group(1) == resident
        k = re.sub(r"(?m)^\s*resident\s*=.*$", want, b)
        if want not in k:
            k = k.rstrip("\n") + "\n" + want + "\n"
        keep.append("[[models]]" + k.rstrip("\n") + "\n")
    if not found:
        raise JobError("validate", f"fastvideo config lists no model {resident!r}")
    return head.rstrip("\n") + "\n\n" + "\n".join(keep)


def serve_argv(ctx: Ctx, variant: str | None = None) -> list[str]:
    if variant is None:
        return [ctx.fv_serve_bin, "--config", ctx.fv_config]
    cfg = Path(ctx.fv_state_dir) / "fastvideo.toml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(group_config(Path(ctx.fv_config).read_text(), group_members(ctx.models, variant), variant))
    return [ctx.fv_serve_bin, "--config", str(cfg)]


def run(ctx: Ctx, job: Job) -> RunResult:
    port = ctx.ports[ENGINE]
    base = f"http://127.0.0.1:{port}"
    load_s = 0.0
    variant = job.model.recipe["variant"]
    key = f"{KEY}:{process_of(job.model)}"  # one fv-serve per process group (a family: its models swap inside it)
    if not ctx.fv_fake:
        def make() -> Sidecar:
            return Sidecar(ENGINE, key, serve_argv(ctx, variant), ctx.env[ENGINE], ping_probe(f"{base}/ping"))

        sc, load_s = ctx.sidecars.ensure(ENGINE, key, make, job.model.vram_gib, lambda st, t: ctx.report("loading", f"{t:.0f} s"))
        # A model switch inside fv-serve (swap) needs the VRAM too.
        ctx.sidecars.make_room(ENGINE, job.model.vram_gib)
    else:
        sc = None
    urls = media_urls(ctx, job)
    body = flags.fastvideo_body(job, ctx.knobs, urls, job.seed)
    command = "curl -sS -X POST %s/fv/v1/jobs -H 'content-type: application/json' -d %s" % (
        base, shlex.quote(_json_short(body)))
    try:
        j = httpc.post_json(f"{base}/fv/v1/jobs", body, timeout=60)
    except httpc.HttpError as e:
        err = (e.json() or {}).get("error") if isinstance(e.json(), dict) else None
        msg = err.get("message") if isinstance(err, dict) else str(e)
        raise JobError("engine", f"fv-serve refused the job: {msg}", sc.tail() if sc else [])
    jid = j["id"]
    t0 = time.time()
    last_progress = -1.0
    while True:
        if sc is not None and not sc.alive():
            raise JobError("engine", "fv-serve died during the job", sc.tail())
        j = httpc.get_json(f"{base}/fv/v1/jobs/{jid}", timeout=30)
        st = j.get("status")
        p = float(j.get("progress") or 0.0)
        if st == "queued":
            ctx.report("queued", f"position {j.get('queue_position')}" if j.get("queue_position") else None)
        elif st == "running":
            if p != last_progress:
                last_progress = p
            steps = job.model.steps
            ctx.report("denoising", f"{int(p * 100)} %", step=int(round(p * steps)) if steps else None, total_steps=steps)
        elif st == "succeeded":
            break
        elif st in ("failed", "cancelled"):
            err = j.get("error") or {}
            raise JobError("engine", f"fv-serve: {err.get('kind', st)}: {err.get('message', '')}", sc.tail() if sc else [])
        time.sleep(POLL_S)
    out = j.get("output") or {}
    if not out.get("url"):
        raise JobError("engine", "fv-serve reported success without an output", sc.tail() if sc else [])
    ctx.report("decoding", "downloading the clip", force=True)
    mp4 = ctx.job_dir / "fv.mp4"
    ctx.job_dir.mkdir(parents=True, exist_ok=True)
    _download(out["url"], mp4)
    m = j.get("metrics") or {}
    stages = m.get("stage_durations") or {}
    t = Timings(load_s=load_s, denoise_s=float(m.get("inference_s") or 0.0), total_s=time.time() - t0)
    t.encode_s = float(stages.get("text_encoder") or stages.get("encode") or stages.get("prompt") or 0.0)
    t.decode_s = float(stages.get("vae_decode") or stages.get("decode") or stages.get("vae") or 0.0)
    return RunResult(
        seed=int(j.get("seed") if j.get("seed") is not None else (job.seed or 0)),
        width=int(j.get("width") or out.get("width") or job.width),
        height=int(j.get("height") or out.get("height") or job.height),
        frames=int(j.get("num_frames") or out.get("frames") or job.frames),
        fps=int(j.get("fps") or out.get("fps") or job.fps_for()),
        video_mp4=mp4, timings=t, command=command,
        extra={"fv_job": jid, "resolved_model": j.get("resolved_model"), "audio": out.get("audio")},
    )


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "fastvideo-studio-worker/1"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)


def _json_short(body: dict) -> str:
    import json

    short = {k: (v[:64] + "…" if isinstance(v, str) and v.startswith("data:") else v) for k, v in body.items()}
    if isinstance(short.get("reference_urls"), list):
        short["reference_urls"] = [u[:64] + "…" if u.startswith("data:") else u for u in short["reference_urls"]]
    return json.dumps(short)
