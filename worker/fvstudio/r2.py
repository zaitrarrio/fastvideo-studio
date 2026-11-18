"""Object storage: Cloudflare R2 through boto3, and a local-directory fake.

Both expose the same small surface: `put_file`, `get_file`, `list_keys`,
`head_size`, `presign_get`, `get_bytes`. The worker reads weights from the
`fv-weights` bucket (fastvideo-rs's layout, docs/ops/r2-weights.md:
`weights/<dest>/<rev>/<path>` with `manifests/<dest>/<rev>.json`) and
writes job outputs under `jobs/<id>/` in the Studio bucket.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol
from urllib.parse import quote


@dataclass(frozen=True)
class R2Config:
    endpoint: str
    access_key_id: str
    secret_access_key: str
    bucket: str  # the Studio bucket (outputs, uploads)

    @staticmethod
    def from_env(env: dict | None = None) -> "R2Config | None":
        e = os.environ if env is None else env
        endpoint = e.get("STUDIO_R2_ENDPOINT") or e.get("FV_R2_ENDPOINT")
        key = e.get("STUDIO_R2_ACCESS_KEY_ID") or e.get("FV_R2_ACCESS_KEY_ID")
        secret = e.get("STUDIO_R2_SECRET_ACCESS_KEY") or e.get("FV_R2_SECRET_ACCESS_KEY")
        bucket = e.get("STUDIO_R2_BUCKET") or e.get("FV_R2_BUCKET")
        if not (endpoint and key and secret and bucket):
            return None
        return R2Config(endpoint, key, secret, bucket)


class ObjectStore(Protocol):
    def put_file(self, bucket: str, key: str, path: Path, content_type: str | None = None) -> None: ...
    def get_file(self, bucket: str, key: str, path: Path) -> int: ...
    def get_bytes(self, bucket: str, key: str) -> bytes: ...
    def list_keys(self, bucket: str, prefix: str) -> Iterator[tuple[str, int]]: ...
    def head_size(self, bucket: str, key: str) -> int | None: ...
    def presign_get(self, bucket: str, key: str, expires_s: int) -> str: ...


class R2Store:
    """boto3-backed R2 (imported lazily so tests need no boto3)."""

    def __init__(self, cfg: R2Config):
        import boto3  # type: ignore
        from botocore.config import Config  # type: ignore

        self.cfg = cfg
        self.s3 = boto3.client(
            "s3",
            endpoint_url=cfg.endpoint,
            aws_access_key_id=cfg.access_key_id,
            aws_secret_access_key=cfg.secret_access_key,
            region_name="auto",
            config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
        )

    def put_file(self, bucket: str, key: str, path: Path, content_type: str | None = None) -> None:
        extra = {"ContentType": content_type or guess_type(path)}
        self.s3.upload_file(str(path), bucket, key, ExtraArgs=extra)

    def get_file(self, bucket: str, key: str, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        self.s3.download_file(bucket, key, str(tmp))
        os.replace(tmp, path)
        return path.stat().st_size

    def get_bytes(self, bucket: str, key: str) -> bytes:
        return self.s3.get_object(Bucket=bucket, Key=key)["Body"].read()

    def list_keys(self, bucket: str, prefix: str) -> Iterator[tuple[str, int]]:
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for o in page.get("Contents", []):
                yield o["Key"], int(o["Size"])

    def head_size(self, bucket: str, key: str) -> int | None:
        try:
            return int(self.s3.head_object(Bucket=bucket, Key=key)["ContentLength"])
        except self.s3.exceptions.ClientError:
            return None

    def presign_get(self, bucket: str, key: str, expires_s: int) -> str:
        return self.s3.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires_s
        )


class LocalStore:
    """A directory per bucket; presigned URLs are `file://` paths plus a
    fake query string. For tests and local runs."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _p(self, bucket: str, key: str) -> Path:
        return self.root / bucket / key

    def put_file(self, bucket: str, key: str, path: Path, content_type: str | None = None) -> None:
        dst = self._p(bucket, key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dst)

    def get_file(self, bucket: str, key: str, path: Path) -> int:
        src = self._p(bucket, key)
        if not src.is_file():
            raise FileNotFoundError(f"{bucket}/{key}")
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, path)
        return path.stat().st_size

    def get_bytes(self, bucket: str, key: str) -> bytes:
        return self._p(bucket, key).read_bytes()

    def list_keys(self, bucket: str, prefix: str) -> Iterator[tuple[str, int]]:
        base = self.root / bucket
        if not base.is_dir():
            return
        for p in sorted(base.rglob("*")):
            if p.is_file():
                key = p.relative_to(base).as_posix()
                if key.startswith(prefix):
                    yield key, p.stat().st_size

    def head_size(self, bucket: str, key: str) -> int | None:
        p = self._p(bucket, key)
        return p.stat().st_size if p.is_file() else None

    def presign_get(self, bucket: str, key: str, expires_s: int) -> str:
        exp = int(time.time()) + expires_s
        sig = hashlib.sha256(f"{bucket}/{key}/{exp}".encode()).hexdigest()[:16]
        return f"file://{self._p(bucket, key)}?X-Expires={exp}&X-Sig={sig}"


def guess_type(path: Path) -> str:
    t, _ = mimetypes.guess_type(str(path))
    if t:
        return t
    return {".y4m": "video/x-yuv4mpeg", ".mp4": "video/mp4", ".wav": "audio/wav", ".jpg": "image/jpeg"}.get(
        path.suffix, "application/octet-stream"
    )


def key_url_part(key: str) -> str:
    return quote(key, safe="/")


def read_json(store: ObjectStore, bucket: str, key: str) -> dict:
    return json.loads(store.get_bytes(bucket, key).decode("utf-8"))
