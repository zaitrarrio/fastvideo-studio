"""Shared by the h3 and fastwan drivers: the serve job loop over
`/api/generate` + `/api/jobs/<id>` (+ `/frames`, `/audio`, `/hd`), and a CLI
runner that streams `step i/n` lines into progress."""

from __future__ import annotations

import collections
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .. import frames as fr
from .. import httpc
from ..errors import JobError
from ..progress import Reporter, Timings
from ..sidecar import Sidecar

STEP_RE = re.compile(r"step (\d+)/(\d+) ([\d.]+)s")
BLOCK_RE = re.compile(r"block (\d+): denoise ([\d.]+)s\s+decode ([\d.]+)s.*?(\d+) frames \((\d+) total")
ENGINE_STAGES = {"queued": "queued", "encoding": "encoding", "preparing": "preparing", "denoising": "denoising",
                 "decoding": "decoding", "upscaling": "upscaling", "done": "done", "failed": "failed"}


@dataclass
class ServeClip:
    rgb: Path
    width: int
    height: int
    frames: int
    audio_wav: Path | None
    hd_jpegs: list[bytes]
    up_factor: int | None
    status: dict
    timings: Timings


def run_serve_job(base: str, sidecar: Sidecar, body: dict, job_dir: Path, report: Reporter, *, want_audio: bool,
                  poll_s: float = 0.5, timeout_s: float = 3600.0) -> ServeClip:
    """Submit to a resident engine and stream its frames as they decode."""
    try:
        sub = httpc.post_json(f"{base}/api/generate", body, timeout=30)
    except httpc.HttpError as e:
        j = e.json()
        msg = (j or {}).get("error") if isinstance(j, dict) else None
        raise JobError("engine", f"{sidecar.engine} refused the job: {msg or e}", sidecar.tail())
    jid = sub["id"]
    rgb_path = job_dir / "frames.rgb"
    clip = fr.RgbClip(rgb_path)
    t0 = time.time()
    status: dict = {}
    hd: list[bytes] = []
    try:
        while True:
            if not sidecar.alive():
                raise JobError("engine", f"{sidecar.engine} sidecar died during the job", sidecar.tail())
            status = httpc.get_json(f"{base}/api/jobs/{jid}", timeout=30)
            stage = ENGINE_STAGES.get(status.get("stage", ""), "denoising")
            steps = status.get("steps") or []
            total_steps = status.get("total_steps")
            got = int(status.get("frames") or 0)
            total_frames = status.get("total_frames")
            if got > clip.frames:
                chunk = fr.parse_frame_stream(httpc.get_bytes(f"{base}/api/jobs/{jid}/frames?from={clip.frames}"))
                clip.append(chunk)
            up_frames = int(status.get("up_frames") or 0)
            while len(hd) < up_frames:
                hd.append(httpc.get_bytes(f"{base}/api/jobs/{jid}/hd?frame={len(hd)}"))
            if stage == "failed":
                raise JobError("engine", f"{sidecar.engine}: {status.get('error') or 'job failed'}", sidecar.tail())
            if stage == "done":
                break
            detail = None
            if stage == "denoising" and total_steps:
                detail = f"step {len(steps)}/{total_steps}"
            elif stage == "decoding" and total_frames:
                detail = f"{got}/{total_frames} frames"
            elif stage == "upscaling" and total_frames:
                detail = f"{up_frames}/{total_frames} frames"
            report(stage, detail, step=len(steps) if steps is not None else None, total_steps=total_steps,
                   frames=got, total_frames=total_frames)
            if time.time() - t0 > timeout_s:
                raise JobError("engine", f"{sidecar.engine} job timed out after {timeout_s:.0f} s", sidecar.tail())
            time.sleep(poll_s)
    finally:
        clip.close()
    audio = None
    if want_audio and status.get("audio"):
        audio = job_dir / "audio.wav"
        audio.write_bytes(httpc.get_bytes(f"{base}/api/jobs/{jid}/audio"))
    t = Timings(
        encode_s=float(status.get("encode_s") or 0.0),
        denoise_s=float(sum(status.get("steps") or [])),
        decode_s=float(status.get("decode_s") or 0.0),
        upscale_s=float(status.get("up_s") or status.get("upscale_s") or 0.0),
        steps_s=[float(s) for s in (status.get("steps") or [])],
    )
    if status.get("up_error"):
        raise JobError("engine", f"{sidecar.engine} upscale failed: {status['up_error']}", sidecar.tail())
    return ServeClip(rgb_path, clip.width, clip.height, clip.frames, audio, hd, status.get("upscale") or None, status, t)


def run_cli(argv: list[str], env: dict[str, str], report: Reporter, log: Callable[[str], None], *,
            timeout_s: float = 7200.0, cwd: Path | None = None) -> tuple[list[str], Timings]:
    """Run an engine CLI, turning its `step i/n` / `block b:` lines into
    progress. Returns the last log lines and the timings parsed."""
    log(f"run: {' '.join(argv)}")
    t = Timings()
    tail: collections.deque = collections.deque(maxlen=400)
    t0 = time.time()
    proc = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            errors="replace", bufsize=1, cwd=str(cwd) if cwd else None, start_new_session=True)
    assert proc.stdout is not None
    report("preparing", "loading the model")
    frames_done = 0
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            tail.append(line)
            log(f"  {line}")
            m = STEP_RE.search(line)
            if m:
                i, n, secs = int(m.group(1)), int(m.group(2)), float(m.group(3))
                t.steps_s.append(secs)
                report("denoising", f"step {i}/{n}", step=i, total_steps=n, force=i == n)
                continue
            m = BLOCK_RE.search(line)
            if m:
                t.steps_s.append(float(m.group(2)))
                t.decode_s += float(m.group(3))
                frames_done = int(m.group(5))
                report("denoising", f"block {int(m.group(1)) + 1}", frames=frames_done)
                continue
            low = line.lower()
            if "decode" in low and "step" not in low and not m:
                report("decoding", line.strip()[:80])
            if time.time() - t0 > timeout_s:
                proc.kill()
                raise JobError("engine", f"{argv[0]} timed out after {timeout_s:.0f} s", list(tail)[-50:])
        code = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    if code != 0:
        raise JobError("engine", f"{os.path.basename(argv[0])} {argv[1]} exited with code {code}", list(tail)[-50:])
    t.denoise_s = sum(t.steps_s)
    t.total_s = time.time() - t0
    return list(tail), t
