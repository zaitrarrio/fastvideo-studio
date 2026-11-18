"""Job media (`{url}` / `{b64, mime}` / `{r2}`) → local files in the job dir,
and the conversions the engines' CLIs need (y4m / wav)."""

from __future__ import annotations

import base64
import mimetypes
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .catalog import Media
from .errors import JobError
from .r2 import ObjectStore

MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024
IMAGE_MIMES = ("image/png", "image/jpeg", "image/jpg", "image/webp", "image/x-portable-pixmap")
EXT = {"image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png", "image/webp": ".webp",
       "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/mpeg": ".mp3", "audio/mp4": ".m4a",
       "video/mp4": ".mp4", "video/quicktime": ".mov", "video/webm": ".webm", "video/x-yuv4mpeg": ".y4m"}


@dataclass(frozen=True)
class LocalMedia:
    path: Path
    mime: str
    source: Media

    @property
    def kind(self) -> str:
        return self.mime.split("/", 1)[0]


def fetch_media(m: Media, dest_dir: Path, name: str, store: ObjectStore | None, bucket: str | None) -> LocalMedia:
    """The media as a file `dest_dir/<name><ext>`."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if m.kind == "b64":
        mime = m.mime or "application/octet-stream"
        path = dest_dir / f"{name}{ext_of(mime, '')}"
        try:
            data = m.decode_b64()
        except Exception as e:  # noqa: BLE001
            raise JobError("validate", f"{name}: bad base64 ({e})") from None
        path.write_bytes(data)
        return LocalMedia(path, mime, m)
    if m.kind == "r2":
        if store is None or not bucket:
            raise JobError("validate", f"{name}: r2 uploads need the Studio bucket (no R2 credentials)")
        mime = guess_mime(m.value)
        path = dest_dir / f"{name}{ext_of(mime, Path(m.value).suffix)}"
        try:
            store.get_file(bucket, m.value, path)
        except Exception as e:  # noqa: BLE001
            raise JobError("validate", f"{name}: cannot read {m.value} from the bucket ({e})") from None
        return LocalMedia(path, mime, m)
    # https
    req = urllib.request.Request(m.value, headers={"User-Agent": "fastvideo-studio-worker/1"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            mime = (r.headers.get("Content-Type") or "").split(";")[0].strip() or guess_mime(m.value)
            path = dest_dir / f"{name}{ext_of(mime, Path(m.value.split('?')[0]).suffix)}"
            with open(path, "wb") as f:
                total = 0
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    total += len(buf)
                    if total > MAX_DOWNLOAD_BYTES:
                        raise JobError("validate", f"{name}: over {MAX_DOWNLOAD_BYTES >> 20} MiB")
                    f.write(buf)
    except JobError:
        raise
    except Exception as e:  # noqa: BLE001
        raise JobError("validate", f"{name}: cannot download {m.value} ({e})") from None
    return LocalMedia(path, mime, m)


def guess_mime(name: str) -> str:
    t, _ = mimetypes.guess_type(name.split("?")[0])
    return t or "application/octet-stream"


def ext_of(mime: str, fallback: str) -> str:
    return EXT.get(mime) or (mimetypes.guess_extension(mime) or fallback or ".bin")


def to_data_uri(lm: LocalMedia) -> str:
    return f"data:{lm.mime};base64," + base64.b64encode(lm.path.read_bytes()).decode("ascii")
