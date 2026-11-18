"""A tiny in-process fake of fv-serve's native jobs API (`/ping`,
`/fv/v1/capabilities`, `POST /fv/v1/jobs`, `GET /fv/v1/jobs/{id}`,
`/files/…`): a job runs through queued → running (progress) → succeeded in
a few polls and serves a real MP4 (made with ffmpeg when available, else a
placeholder file) at its output URL."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def make_mp4(path: Path, width: int = 64, height: int = 48, frames: int = 9, fps: int = 24, audio: bool = True) -> bool:
    ff = shutil.which("ffmpeg")
    if not ff:
        path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
        return False
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"color=c=blue:s={width}x{height}:r={fps}:d={frames / fps:.3f}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=f=440:d={frames / fps:.3f}", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-frames:v", str(frames), "-shortest", str(path)]
    subprocess.run(cmd, check=True)
    return True


class FakeFvServe:
    def __init__(self, files_dir: Path, steps: int = 4, step_s: float = 0.6, fail_model: str | None = None):
        self.files_dir = files_dir
        self.jobs: dict[str, dict] = {}
        self.requests: list[dict] = []
        self.steps = steps
        self.step_s = step_s
        self.fail_model = fail_model
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def _json(self, code: int, body: dict):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/ping":
                    return self._json(200, {"status": "healthy"})
                if self.path == "/fv/v1/capabilities":
                    return self._json(200, {"auth": {"mode": "none"}, "protocols": ["native"], "models": []})
                if self.path.startswith("/fv/v1/jobs/"):
                    jid = self.path.rsplit("/", 1)[1]
                    j = fake.jobs.get(jid)
                    if not j:
                        return self._json(404, {"error": {"kind": "not_found", "message": "no such job"}})
                    fake._advance(j)
                    return self._json(200, fake._view(j))
                if self.path.startswith("/files/"):
                    p = fake.files_dir / self.path[len("/files/"):].split("?")[0]
                    if not p.is_file():
                        self.send_response(404); self.end_headers(); return
                    data = p.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                self.send_response(404); self.end_headers()

            def do_POST(self):
                if self.path != "/fv/v1/jobs":
                    self.send_response(404); self.end_headers(); return
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                fake.requests.append(body)
                allowed = {"model", "prompt", "negative_prompt", "seed", "size", "width", "height", "aspect_ratio", "short_edge", "seconds",
                           "num_frames", "fps", "steps", "guidance", "image_url", "last_image_url", "reference_urls", "reference_strength",
                           "reference_lora_strength", "audio_url", "video_url", "start_s", "end_s", "retake_mode", "extend_s", "extend_at",
                           "context_s", "prompt_expansion"}
                unknown = set(body) - allowed
                if unknown:
                    return self._json(400, {"error": {"kind": "invalid_request", "message": f"unknown field `{sorted(unknown)[0]}`", "param": sorted(unknown)[0]}})
                if not body.get("prompt") and not body.get("video_url") and not body.get("audio_url"):
                    return self._json(400, {"error": {"kind": "invalid_request", "message": "`prompt` must not be empty", "param": "prompt"}})
                jid = "fvjob_" + uuid.uuid4().hex
                w, h = (body.get("size") or "832x480").split("x")
                j = {"id": jid, "model": body["model"], "created": time.time(), "state": "queued", "progress": 0.0,
                     "width": int(w), "height": int(h), "num_frames": int(body.get("num_frames") or 9), "fps": int(body.get("fps") or 24),
                     "seed": int(body.get("seed") if body.get("seed") is not None else 42), "body": body}
                fake.jobs[jid] = j
                return self._json(202, fake._view(j))

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> "FakeFvServe":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()

    def _advance(self, j: dict) -> None:
        t = time.time() - j["created"]
        if j["state"] in ("succeeded", "failed"):
            return
        if j["model"] == self.fail_model:
            j["state"] = "failed"
            j["error"] = {"kind": "engine", "message": "the fake engine refuses this model"}
            return
        k = int(t / self.step_s)
        if k == 0:
            j["state"] = "queued"
        elif k <= self.steps:
            j["state"] = "running"
            j["progress"] = k / self.steps
        else:
            j["state"] = "succeeded"
            j["progress"] = 1.0
            out = self.files_dir / f"{j['id']}.mp4"
            if not out.exists():
                self.files_dir.mkdir(parents=True, exist_ok=True)
                make_mp4(out, j["width"], j["height"], min(j["num_frames"], 9), j["fps"])
            j["output"] = {"url": f"http://127.0.0.1:{self.port}/files/{out.name}?sig=x", "mime": "video/mp4", "bytes": out.stat().st_size,
                           "width": j["width"], "height": j["height"], "frames": j["num_frames"], "fps": j["fps"],
                           "audio": {"rate": 48000, "channels": 2}}
            j["metrics"] = {"inference_s": 1.5, "stage_durations": {"text_encoder": 0.2, "vae_decode": 0.3}}

    def _view(self, j: dict) -> dict:
        return {"id": j["id"], "object": "fv.job", "status": j["state"], "progress": round(j["progress"], 2), "queue_position": None,
                "model": j["model"], "resolved_model": j["model"], "task": "t2v", "width": j["width"], "height": j["height"],
                "num_frames": j["num_frames"], "fps": j["fps"], "seed": j["seed"], "error": j.get("error"), "output": j.get("output"),
                "protocol": "native", "metrics": j.get("metrics") or {"inference_s": None, "stage_durations": {}}}
