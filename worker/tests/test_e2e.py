"""End to end against the real engines in `--synthetic` mode on the CPU,
a recording `progress_update` in place of the RunPod loop, a fake fv-serve
and a local R2 (STUDIO_E2E=1):

    cargo build --release -p h3-cli        (in h3-rs)
    cargo build --release -p fastwan-cli   (in fastwan-rs)

then the handler's generate path for h3, fastwan and fastvideo, producing
real MP4s through ffmpeg (skipped assertions without ffmpeg)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from fake_fv_serve import FakeFvServe

pytestmark = pytest.mark.e2e
E2E = os.environ.get("STUDIO_E2E") == "1"
H3 = Path(os.environ.get("H3_RS", "/home/user/h3-rs"))
FASTWAN = Path(os.environ.get("FASTWAN_RS", "/home/user/fastwan-rs"))


def build(repo: Path, crate: str, binary: str) -> Path:
    target = Path(os.environ.get("CARGO_TARGET_DIR_" + crate.upper().replace("-", "_"), repo / "target"))
    out = target / "release" / binary
    env = {**os.environ, "CARGO_TARGET_DIR": str(target)}
    env.pop("CARGO_BUILD_JOBS", None)
    t0 = time.time()
    subprocess.run(["cargo", "build", "--release", "-p", crate], cwd=repo, env=env, check=True)
    print(f"built {binary} in {time.time() - t0:.0f} s: {out}")
    assert out.is_file()
    return out


@pytest.fixture(scope="module")
def binaries():
    if not E2E:
        pytest.skip("STUDIO_E2E=1 runs the end-to-end test")
    return build(H3, "h3-cli", "h3"), build(FASTWAN, "fastwan-cli", "fastwan")


class ProgressRecorder:
    """What `fvstudio.rp.RunpodClient.progress_update` would post: the
    `(job id, progress object)` pairs, in order."""

    def __init__(self):
        self.updates = []

    def progress_update(self, job, data):
        self.updates.append((job["id"], data))


@pytest.fixture(scope="module")
def fake_runpod():
    return ProgressRecorder()


@pytest.fixture(scope="module")
def worker_env(tmp_path_factory, binaries, fake_runpod):
    h3, fastwan = binaries
    root = tmp_path_factory.mktemp("studio")
    bins = root / "bin"
    bins.mkdir()
    os.symlink(h3, bins / "h3")
    os.symlink(fastwan, bins / "fastwan")
    fake = FakeFvServe(root / "fvfiles").start()
    env = {
        "STUDIO_BIN_DIR": str(bins), "STUDIO_BACKEND": "cpu", "STUDIO_SYNTHETIC": "1", "STUDIO_FV_FAKE": "1",
        "STUDIO_FV_PORT": str(fake.port), "STUDIO_H3_PORT": "7881", "STUDIO_FASTWAN_PORT": "7882",
        "STUDIO_JOBS_DIR": str(root / "jobs"), "STUDIO_R2_LOCAL_DIR": str(root / "r2"), "STUDIO_R2_BUCKET": "studio",
        "STUDIO_VOLUME": str(root / "volume"), "STUDIO_VOLUME_WEIGHTS": str(root / "volume" / "weights"),
        "STUDIO_VOLUME_EXTRAS": str(root / "volume" / "h3-rs"), "STUDIO_STAGE_DIR": str(root / "stage"),
        "STUDIO_LINK_ROOT": str(root / "w"), "STUDIO_STAGE_DRY_RUN": "1", "STUDIO_IMAGE": "e2e-test",
        "STUDIO_KEEP_JOB_DIRS": "1",
    }
    # Fake volume trees so staging is a no-op link.
    for t in ("h3-8step", "h3-8step-mxfp8", "fastwan", "auxiliary/tae", "fastwan22-ti2v-5b",
              "h3-base/transformer", "h3-base/text_encoder", "h3-base/tokenizer", "h3-base/vae", "h3-base/audio_vae", "h3-ref2va"):
        (root / "volume" / "weights" / t).mkdir(parents=True)
    for f in ("auxiliary/tae/taew2_1.safetensors", "auxiliary/tae/taew2_2.safetensors"):
        (root / "volume" / "weights" / f).write_bytes(b"\x00" * 16)
    for t in ("tables", "plug4"):
        (root / "volume" / "h3-rs" / t).mkdir(parents=True)
    (root / "stage" / "fastwan").mkdir(parents=True)
    (root / "stage" / "fastwan" / ".complete").write_text("{}\n")
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    import handler as handler_mod

    h = handler_mod.make_handler(progress_update=fake_runpod.progress_update)
    yield h, fake, root
    h.worker.sidecars.stop_all()
    fake.stop()
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def run_job(h, fake_runpod, jid: str, inp: dict) -> dict:
    t0 = time.time()
    out = h({"id": jid, "input": inp})
    print(f"\n{jid}: {time.time() - t0:.1f} s → {json.dumps({k: v for k, v in out.items() if k not in ('timings', 'command')}, indent=None)[:600]}")
    if "timings" in out:
        print(f"  timings {json.dumps(out['timings'])}\n  command {out['command']}")
    return out


def check_mp4(url: str, frames: int | None, audio: bool):
    assert url.startswith("file://")
    path = Path(url[len("file://"):].split("?")[0])
    assert path.is_file() and path.stat().st_size > 1000
    if shutil.which("ffprobe"):
        from fvstudio import mux

        p = mux.probe(path)
        assert p is not None
        if frames is not None:
            assert p.frames == frames
        assert p.audio == audio
        print(f"  ffprobe: {p}")
    else:
        print("  (no ffprobe: mp4 structure not checked)")


def test_info(worker_env, fake_runpod):
    h, fake, root = worker_env
    out = run_job(h, fake_runpod, "info-1", {"op": "info"})
    assert out["catalog_version"] == 1 and "h3" in out["versions"] and out["image"] == "e2e-test"
    assert "weights/h3-8step" in out["trees"] and out["trees"]["weights/h3-8step"]["volume"] is True
    assert out["versions"]["h3"].startswith("h3 (") and out["versions"]["fastwan"].startswith("fastwan (")


def test_h3_synthetic_generate(worker_env, fake_runpod):
    h, fake, root = worker_env
    fake_runpod.updates.clear()
    out = run_job(h, fake_runpod, "h3-1", {"engine": "h3", "model": "fasth3-v2", "prompt": "a fox", "width": 480, "height": 272,
                                          "frames": 124, "seed": 7, "options": {"pv8": True, "tables": True}, "client": {"job": "c1"}})
    assert "error" not in out, out
    assert (out["engine"], out["model"], out["seed"], out["frames"], out["fps"], out["audio"]) == ("h3", "fasth3-v2", 7, 124, 24, True)
    assert out["width"] == 480 and out["height"] == 272 and out["client"] == {"job": "c1"}
    assert out["command"].startswith("h3 generate --backend cpu --variant fasth3-v2 --task t2va")
    assert out["timings"]["steps_s"] and len(out["timings"]["steps_s"]) == 8
    assert out["worker"]["image"] == "e2e-test" and out["expires_at"].endswith("Z")
    stages = [d["stage"] for _, d in fake_runpod.updates]
    for s in ("staging", "loading", "denoising", "decoding", "muxing", "uploading", "done"):
        assert s in stages, stages
    assert all(set(d) >= {"stage", "elapsed_s", "engine", "model"} for _, d in fake_runpod.updates)
    if shutil.which("ffmpeg"):
        check_mp4(out["video_url"], 124, True)
        assert out["audio_url"] and out["thumbnail_url"]
    else:
        pytest.skip("ffmpeg missing: the mux assertion is skipped")


def test_h3_fl2v_runs_through_the_resident_serve(worker_env, fake_runpod):
    """h3 serve takes first_image / last_image per request (h3-rs serve-inputs);
    the job must not fall back to the per-job CLI."""
    import base64, struct, zlib

    def png(w, h, rgb):
        raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
        def chunk(t, d):
            return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
        return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")

    h, fake, root = worker_env
    fake_runpod.updates.clear()
    first = {"b64": base64.b64encode(png(64, 32, (200, 30, 30))).decode(), "mime": "image/png"}
    last = {"b64": base64.b64encode(png(64, 32, (30, 30, 200))).decode(), "mime": "image/png"}
    out = run_job(h, fake_runpod, "h3-fl2v", {"engine": "h3", "model": "h3-plug4", "task": "fl2v", "prompt": "a fox", "width": 480, "height": 272,
                                             "frames": 124, "seed": 3, "inputs": {"first_image": first, "last_image": last}})
    assert "error" not in out, out
    assert out["task"] == "fl2v" and out["frames"] == 124
    details = [d.get("detail") or "" for _, d in fake_runpod.updates]
    assert not any("per-job CLI" in d for d in details), details
    assert "--task fl2va" in out["command"] and "--first-image" in out["command"]


def test_h3_sidecar_is_reused_and_survives_a_crash(worker_env, fake_runpod):
    h, fake, root = worker_env
    sc = h.worker.sidecars.get("h3")
    assert sc is not None and sc.alive()
    pid = sc.proc.pid
    out = run_job(h, fake_runpod, "h3-2", {"engine": "h3", "model": "fasth3-v2", "prompt": "a cat", "width": 480, "height": 272, "frames": 124})
    assert "error" not in out and h.worker.sidecars.get("h3").proc.pid == pid  # reused
    # Kill it: the next job restarts it and still succeeds.
    sc.proc.kill()
    sc.proc.wait()
    out = run_job(h, fake_runpod, "h3-3", {"engine": "h3", "model": "fasth3-v2", "prompt": "a dog", "width": 480, "height": 272, "frames": 124})
    assert "error" not in out and h.worker.sidecars.get("h3").proc.pid != pid


def test_h3_crash_mid_job_fails_with_engine_and_log_tail(worker_env, fake_runpod):
    h, fake, root = worker_env
    import threading

    sc = h.worker.sidecars.get("h3")
    assert sc is not None and sc.alive()

    def killer():
        time.sleep(3)
        sc.proc.kill()

    threading.Thread(target=killer, daemon=True).start()
    out = run_job(h, fake_runpod, "h3-4", {"engine": "h3", "model": "fasth3-v2", "prompt": "a crash", "width": 480, "height": 272, "frames": 124})
    assert out["where"] == "engine" and "died" in out["error"]
    assert isinstance(out.get("log_tail"), list) and len(out["log_tail"]) <= 50


def test_validation_error_shape(worker_env, fake_runpod):
    h, fake, root = worker_env
    out = run_job(h, fake_runpod, "bad-1", {"engine": "h3", "model": "fasth3-v2", "prompt": "x", "frames": 100})
    assert out["where"] == "validate" and "frames" in out["error"]


def test_fastwan_synthetic_generate_with_upscale(worker_env, fake_runpod):
    h, fake, root = worker_env
    fake_runpod.updates.clear()
    out = run_job(h, fake_runpod, "fw-1", {"engine": "fastwan", "model": "fastwan-1.3b", "prompt": "a fox", "width": 480, "height": 272,
                                          "frames": 17, "seed": 3, "upscale": 2})
    assert "error" not in out, out
    assert (out["engine"], out["frames"], out["fps"], out["audio"]) == ("fastwan", 17, 16, False)
    assert out["upscale"] == {"factor": 2, "width": 960, "height": 544}
    assert out["command"].startswith("fastwan generate --backend cpu")
    assert "--upscale-x 2" in out["command"] or "--upscale-x" in out["command"]
    if shutil.which("ffmpeg"):
        check_mp4(out["video_url"], 17, False)
    assert out["audio_url"] is None


def test_fastvideo_fake_generate(worker_env, fake_runpod):
    h, fake, root = worker_env
    out = run_job(h, fake_runpod, "fv-1", {"engine": "fastvideo", "model": "fastwan22-ti2v-5b", "prompt": "a fox", "width": 832,
                                          "height": 480, "frames": 49, "seed": 5, "options": {"steps": 8}})
    assert "error" not in out, out
    assert (out["engine"], out["model"], out["seed"], out["frames"]) == ("fastvideo", "fastwan22-ti2v-5b", 5, 49)
    assert fake.requests[-1]["steps"] == 8 and fake.requests[-1]["size"] == "832x480"
    assert out["timings"]["denoise_s"] == 1.5
    if shutil.which("ffmpeg"):
        check_mp4(out["video_url"], None, True)
    job_json = root / "r2" / "studio" / "jobs" / "fv-1" / "job.json"
    assert job_json.is_file() and json.loads(job_json.read_text())["output"]["model"] == "fastwan22-ti2v-5b"


def test_warm(worker_env, fake_runpod):
    h, fake, root = worker_env
    out = run_job(h, fake_runpod, "warm-1", {"op": "warm", "engine": "fastwan", "model": "fastwan-1.3b"})
    assert out["op"] == "warm" and out["sidecars"]["fastwan"]["alive"] is True
