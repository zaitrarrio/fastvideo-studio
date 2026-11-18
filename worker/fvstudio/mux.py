"""ffmpeg: y4m (+ wav) → H.264 + AAC yuv420p MP4 with `+faststart`; a JPEG
thumbnail from the middle frame; a probe of an MP4's streams."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def ffprobe_path() -> str | None:
    return shutil.which("ffprobe")


@dataclass(frozen=True)
class MuxResult:
    mp4: Path
    seconds: float


def mux(video_y4m: Path, audio_wav: Path | None, out: Path, crf: int = 18, preset: str = "medium", ffmpeg: str | None = None) -> MuxResult:
    ff = ffmpeg or ffmpeg_path()
    if not ff:
        raise RuntimeError("ffmpeg is not installed")
    t0 = time.time()
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_y4m)]
    if audio_wav is not None and audio_wav.is_file():
        cmd += ["-i", str(audio_wav)]
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"]
    if audio_wav is not None and audio_wav.is_file():
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-movflags", "+faststart", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed: {r.stderr.strip()[-1200:]}")
    return MuxResult(out, time.time() - t0)


def thumbnail(mp4: Path, out: Path, frames: int, fps: int, ffmpeg: str | None = None, quality: int = 3) -> Path:
    ff = ffmpeg or ffmpeg_path()
    if not ff:
        raise RuntimeError("ffmpeg is not installed")
    mid = max(frames // 2, 0)
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp4),
           "-vf", f"select=eq(n\\,{mid})", "-frames:v", "1", "-q:v", str(quality), str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out.is_file():
        # The middle frame may be past a short/odd stream: fall back to the first.
        cmd = [ff, "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp4), "-frames:v", "1", "-q:v", str(quality), str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg thumbnail failed: {r.stderr.strip()[-800:]}")
    return out


def extract_wav(mp4: Path, out: Path, ffmpeg: str | None = None) -> Path | None:
    """The audio track as 16-bit WAV, or None when there is none."""
    ff = ffmpeg or ffmpeg_path()
    if not ff:
        return None
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp4), "-vn", "-acodec", "pcm_s16le", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out.is_file() or out.stat().st_size < 64:
        return None
    return out


@dataclass(frozen=True)
class Probe:
    width: int
    height: int
    frames: int
    fps: float
    audio: bool


def probe(mp4: Path, ffprobe: str | None = None) -> Probe | None:
    fp = ffprobe or ffprobe_path()
    if not fp:
        return None
    cmd = [fp, "-v", "error", "-show_streams", "-count_frames", "-of", "json", str(mp4)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    streams = json.loads(r.stdout or "{}").get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not v:
        return None
    n, d = (v.get("r_frame_rate") or "0/1").split("/")
    fps = int(n) / int(d) if int(d) else 0.0
    frames = int(v.get("nb_read_frames") or v.get("nb_frames") or 0)
    return Probe(int(v["width"]), int(v["height"]), frames, fps, any(s.get("codec_type") == "audio" for s in streams))


def to_y4m(src: Path, out: Path, width: int | None = None, height: int | None = None, fps: int | None = None,
           last_frames: int | None = None, ffmpeg: str | None = None) -> Path:
    """Any video → y4m (yuv420p), optionally scaled to a canvas (cover-crop)
    and trimmed to its last N frames (h3's `--continue` takes 5 / 22 / 39
    frames at the canvas)."""
    ff = ffmpeg or ffmpeg_path()
    if not ff:
        raise RuntimeError("ffmpeg is not installed")
    vf = []
    if width and height:
        vf.append(f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}")
    if fps:
        vf.append(f"fps={fps}")
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-y"]
    if last_frames:
        p = probe(src)
        if p and p.frames > last_frames and p.fps:
            start = max((p.frames - last_frames) / p.fps - 0.001, 0.0)
            cmd += ["-ss", f"{start:.4f}"]
    cmd += ["-i", str(src)]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    if last_frames:
        cmd += ["-frames:v", str(last_frames)]
    cmd += ["-pix_fmt", "yuv420p", "-f", "yuv4mpegpipe", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg →y4m failed: {r.stderr.strip()[-800:]}")
    return out


def to_wav(src: Path, out: Path, ffmpeg: str | None = None) -> Path:
    ff = ffmpeg or ffmpeg_path()
    if not ff:
        raise RuntimeError("ffmpeg is not installed")
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-vn", "-acodec", "pcm_s16le", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg →wav failed: {r.stderr.strip()[-800:]}")
    return out
