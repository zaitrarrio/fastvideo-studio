"""The engines' frame stream (`GET /api/jobs/<id>/frames?from=N`: a 16-byte
little-endian u32 header `width, height, first, count` then RGB8 frames) and
y4m files (parsed natively; written from RGB through ffmpeg's rawvideo
input, the fastest colour conversion available without numpy)."""

from __future__ import annotations

import re
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

HEADER = struct.Struct("<4I")


@dataclass
class FrameChunk:
    width: int
    height: int
    first: int
    count: int
    rgb: bytes

    @property
    def frame_bytes(self) -> int:
        return self.width * self.height * 3


def parse_frame_stream(body: bytes) -> FrameChunk:
    if len(body) < HEADER.size:
        raise ValueError(f"frame stream too short ({len(body)} bytes)")
    width, height, first, count = HEADER.unpack_from(body, 0)
    px = width * height * 3
    want = HEADER.size + px * count
    if len(body) != want:
        raise ValueError(f"frame stream: {len(body)} bytes for {count} frames of {width}x{height} (want {want})")
    return FrameChunk(width, height, first, count, body[HEADER.size:])


def encode_frame_stream(width: int, height: int, first: int, frames: list[bytes]) -> bytes:
    """The inverse, for tests and fakes."""
    return HEADER.pack(width, height, first, len(frames)) + b"".join(frames)


@dataclass
class RgbClip:
    """Frames accumulated from successive chunks, written to a raw file."""

    path: Path
    width: int = 0
    height: int = 0
    frames: int = 0
    _fh: object = field(default=None, repr=False)

    def append(self, chunk: FrameChunk) -> None:
        if chunk.count == 0:
            return
        if self.frames == 0:
            self.width, self.height = chunk.width, chunk.height
            self._fh = open(self.path, "wb")
        elif (chunk.width, chunk.height) != (self.width, self.height):
            raise ValueError("frame size changed mid-stream")
        if chunk.first != self.frames:
            raise ValueError(f"frame stream gap: have {self.frames}, chunk starts at {chunk.first}")
        self._fh.write(chunk.rgb)  # type: ignore[union-attr]
        self.frames += chunk.count

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()  # type: ignore[union-attr]
            self._fh = None


@dataclass(frozen=True)
class Y4mInfo:
    width: int
    height: int
    fps: float
    frames: int
    colorspace: str


def read_y4m_info(path: Path) -> Y4mInfo:
    """The stream header and a frame count from the file size."""
    with open(path, "rb") as f:
        header = f.readline()
    if not header.startswith(b"YUV4MPEG2 "):
        raise ValueError(f"{path}: not a y4m file")
    w = h = 0
    fps = 0.0
    cs = "420jpeg"
    for tok in header[len(b"YUV4MPEG2 "):].split():
        t = tok.decode("ascii", "replace")
        if t[0] == "W":
            w = int(t[1:])
        elif t[0] == "H":
            h = int(t[1:])
        elif t[0] == "F":
            n, d = t[1:].split(":")
            fps = int(n) / int(d)
        elif t[0] == "C":
            cs = t[1:]
    if cs.startswith("444"):
        per = w * h * 3
    elif cs.startswith("422"):
        per = w * h * 2
    elif cs == "mono":
        per = w * h
    else:
        per = w * h * 3 // 2
    size = path.stat().st_size - len(header)
    frame_block = len(b"FRAME\n") + per
    return Y4mInfo(w, h, fps, size // frame_block if frame_block else 0, cs)


def rgb_to_y4m(rgb_path: Path, width: int, height: int, fps: int, out: Path, ffmpeg: str = "ffmpeg") -> None:
    """Raw RGB8 frames → y4m (yuv420p) through ffmpeg."""
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", str(rgb_path),
        "-pix_fmt", "yuv420p", "-f", "yuv4mpegpipe", str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg rgb→y4m failed: {r.stderr.strip()[-800:]}")


def jpegs_to_y4m(jpegs: list[bytes], fps: int, out: Path, ffmpeg: str = "ffmpeg") -> None:
    """A JPEG sequence (upscaled frames) → y4m through ffmpeg's image2pipe."""
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "image2pipe", "-vcodec", "mjpeg", "-r", str(fps), "-i", "pipe:0",
        "-pix_fmt", "yuv420p", "-f", "yuv4mpegpipe", str(out),
    ]
    r = subprocess.run(cmd, input=b"".join(jpegs), capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg jpeg→y4m failed: {r.stderr.decode(errors='replace').strip()[-800:]}")


STEP_RE = re.compile(r"^step (\d+)/(\d+) ([\d.]+)s")
BLOCK_RE = re.compile(r"^block (\d+): denoise ([\d.]+)s\s+decode ([\d.]+)s.*?(\d+) frames \((\d+) total")
