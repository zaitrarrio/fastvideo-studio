"""Object storage: Cloudflare R2 over its S3 API (stdlib SigV4, no boto3),
and a local-directory fake.

Both expose the same small surface: `put_file`, `get_file`, `list_keys`,
`head_size`, `presign_get`, `get_bytes`. The worker reads weights from the
`fv-weights` bucket (fastvideo-rs's layout, docs/ops/r2-weights.md:
`weights/<dest>/<rev>/<path>` with `manifests/<dest>/<rev>.json`) and
writes job outputs under `jobs/<id>/` in the Studio bucket.
"""

from __future__ import annotations

import hashlib
import hmac
import http.client
import json
import mimetypes
import os
import shutil
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Protocol, TypeVar
from urllib.parse import quote, urlsplit, urlunsplit

T = TypeVar("T")


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


class R2Error(Exception):
    """A non-2xx answer from the object store (status, body, url)."""

    def __init__(self, status: int, body: str, url: str):
        super().__init__(f"HTTP {status} from {url}: {body[:300]}")
        self.status = status
        self.body = body
        self.url = url


EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
UNSIGNED_PAYLOAD = "UNSIGNED-PAYLOAD"
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


def _uri_encode(s: str) -> str:
    """AWS's URI encoding: everything but unreserved characters (botocore's
    `percent_encode`, used for query keys and values)."""
    return quote(s, safe="-_.~")


def _host_header(url: str) -> str:
    """`Host` as SigV4 signs it: lowercase, no userinfo, no default port
    (botocore's `_host_from_url`)."""
    u = urlsplit(url)
    host = u.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    default = {"http": 80, "https": 443}.get(u.scheme)
    if u.port is not None and u.port != default:
        host = f"{host}:{u.port}"
    return host


class SigV4:
    """AWS Signature Version 4 for S3-style requests (stdlib only): header
    auth for the API calls and query auth for presigned GETs. The canonical
    request takes the URL path as given (S3 does not normalise it) and the
    query string's already-encoded `k=v` pairs sorted, like botocore's
    `S3SigV4Auth`; `tests/test_r2.py` holds it byte-identical to botocore."""

    def __init__(self, access_key_id: str, secret_access_key: str, region: str = "auto", service: str = "s3"):
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.region = region
        self.service = service

    def scope(self, date: str) -> str:
        return f"{date}/{self.region}/{self.service}/aws4_request"

    def signing_key(self, date: str) -> bytes:
        k = hmac.new(("AWS4" + self.secret_access_key).encode("utf-8"), date.encode(), hashlib.sha256).digest()
        for part in (self.region, self.service, "aws4_request"):
            k = hmac.new(k, part.encode(), hashlib.sha256).digest()
        return k

    @staticmethod
    def canonical_query(url: str) -> str:
        q = urlsplit(url).query
        if not q:
            return ""
        pairs = []
        for pair in q.split("&"):
            k, _, v = pair.partition("=")
            pairs.append((k, v))
        return "&".join(f"{k}={v}" for k, v in sorted(pairs))

    def _signature(self, method: str, url: str, headers: dict[str, str], payload_hash: str, amz_date: str) -> tuple[str, str]:
        """(signed header names, signature) of the canonical request built from
        `headers` (every one of them is signed)."""
        canon = sorted((k.lower(), " ".join(v.split())) for k, v in headers.items())
        signed = ";".join(k for k, _ in canon)
        canonical_request = "\n".join([
            method.upper(),
            urlsplit(url).path or "/",
            self.canonical_query(url),
            "".join(f"{k}:{v}\n" for k, v in canon),
            signed,
            payload_hash,
        ])
        date = amz_date[:8]
        string_to_sign = "\n".join([
            "AWS4-HMAC-SHA256", amz_date, self.scope(date), hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ])
        sig = hmac.new(self.signing_key(date), string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
        return signed, sig

    def sign_headers(self, method: str, url: str, headers: dict[str, str] | None, payload_hash: str,
                     now: datetime | None = None) -> dict[str, str]:
        """The request headers with `Host`, `X-Amz-Date`, `X-Amz-Content-SHA256`
        and `Authorization` added; every header in the result but
        `Authorization` is signed."""
        amz_date = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
        h = {k: v for k, v in (headers or {}).items() if k.lower() not in ("authorization", "x-amz-date", "x-amz-content-sha256", "host")}
        h["Host"] = _host_header(url)
        h["X-Amz-Date"] = amz_date
        h["X-Amz-Content-SHA256"] = payload_hash
        signed, sig = self._signature(method, url, h, payload_hash, amz_date)
        h["Authorization"] = f"AWS4-HMAC-SHA256 Credential={self.access_key_id}/{self.scope(amz_date[:8])}, SignedHeaders={signed}, Signature={sig}"
        return h

    def presign(self, method: str, url: str, expires_s: int, now: datetime | None = None) -> str:
        """Query-string auth (`X-Amz-Algorithm` … `X-Amz-Signature`) over
        `host` only with an unsigned payload, appended to `url`'s own query."""
        amz_date = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
        auth = [
            ("X-Amz-Algorithm", "AWS4-HMAC-SHA256"),
            ("X-Amz-Credential", f"{self.access_key_id}/{self.scope(amz_date[:8])}"),
            ("X-Amz-Date", amz_date),
            ("X-Amz-Expires", str(int(expires_s))),
            ("X-Amz-SignedHeaders", "host"),
        ]
        u = urlsplit(url)
        query = (u.query + "&" if u.query else "") + "&".join(f"{_uri_encode(k)}={_uri_encode(v)}" for k, v in auth)
        unsigned = urlunsplit((u.scheme, u.netloc, u.path, query, ""))
        _, sig = self._signature(method, unsigned, {"host": _host_header(url)}, UNSIGNED_PAYLOAD, amz_date)
        return f"{unsigned}&X-Amz-Signature={sig}"


class R2Store:
    """Cloudflare R2 over its S3 API with the stdlib: SigV4 header auth,
    path-style URLs (`<endpoint>/<bucket>/<key>`), region `auto`. Uploads
    stream from disk with an unsigned payload (`UNSIGNED-PAYLOAD` over TLS,
    which R2 accepts), downloads stream to a `.part` file; retried on
    connection errors and 429 / 5xx."""

    attempts = 4
    backoff_s = 0.5
    timeout_s = 120.0

    def __init__(self, cfg: R2Config, sleep: Callable[[float], None] = time.sleep):
        self.cfg = cfg
        self.endpoint = cfg.endpoint.rstrip("/")
        self.signer = SigV4(cfg.access_key_id, cfg.secret_access_key)
        self._sleep = sleep

    # ---- requests ------------------------------------------------------------

    def url(self, bucket: str, key: str = "", query: list[tuple[str, str]] | None = None) -> str:
        u = f"{self.endpoint}/{_uri_encode(bucket)}"
        if key:
            u += "/" + key_url_part(key)
        if query:
            u += "?" + "&".join(f"{_uri_encode(k)}={_uri_encode(v)}" for k, v in query)
        return u

    def _open(self, method: str, url: str, headers: dict[str, str] | None = None, body=None,
              payload_hash: str = EMPTY_SHA256, extra: dict[str, str] | None = None):
        """One signed request; the open response (2xx) or `R2Error`."""
        h = self.signer.sign_headers(method, url, headers, payload_hash)
        if extra:
            h.update(extra)  # unsigned (Content-Length)
        req = urllib.request.Request(url, data=body, method=method, headers=h)
        try:
            return urllib.request.urlopen(req, timeout=self.timeout_s)
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", "replace") if method != "HEAD" else ""
            raise R2Error(e.code, text, url) from None

    def _retry(self, op: Callable[[], T]) -> T:
        last: Exception | None = None
        for i in range(self.attempts):
            try:
                return op()
            except R2Error as e:
                if e.status not in _RETRY_STATUSES:
                    raise
                last = e
            except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
                last = e
            if i + 1 < self.attempts:
                self._sleep(self.backoff_s * (2 ** i))
        assert last is not None
        raise last

    # ---- the ObjectStore surface ----------------------------------------------

    def put_file(self, bucket: str, key: str, path: Path, content_type: str | None = None) -> None:
        path = Path(path)
        size = path.stat().st_size
        url = self.url(bucket, key)
        headers = {"Content-Type": content_type or guess_type(path)}

        def op() -> None:
            with open(path, "rb") as f:
                with self._open("PUT", url, headers, body=f, payload_hash=UNSIGNED_PAYLOAD, extra={"Content-Length": str(size)}) as r:
                    r.read()

        self._retry(op)

    def get_file(self, bucket: str, key: str, path: Path) -> int:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        url = self.url(bucket, key)

        def op() -> None:
            with self._open("GET", url) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)

        try:
            self._retry(op)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        os.replace(tmp, path)
        return path.stat().st_size

    def get_bytes(self, bucket: str, key: str) -> bytes:
        url = self.url(bucket, key)

        def op() -> bytes:
            with self._open("GET", url) as r:
                return r.read()

        return self._retry(op)

    def list_keys(self, bucket: str, prefix: str) -> Iterator[tuple[str, int]]:
        token: str | None = None
        while True:
            query = [("list-type", "2"), ("prefix", prefix)]
            if token:
                query.append(("continuation-token", token))
            url = self.url(bucket, query=query)

            def op(url=url) -> bytes:
                with self._open("GET", url) as r:
                    return r.read()

            root = ET.fromstring(self._retry(op))
            truncated = False
            token = None
            for el in root:
                tag = el.tag.rsplit("}", 1)[-1]
                if tag == "Contents":
                    fields = {c.tag.rsplit("}", 1)[-1]: (c.text or "") for c in el}
                    yield fields["Key"], int(fields.get("Size") or 0)
                elif tag == "IsTruncated":
                    truncated = (el.text or "").strip().lower() == "true"
                elif tag == "NextContinuationToken":
                    token = (el.text or "").strip() or None
            if not (truncated and token):
                return

    def head_size(self, bucket: str, key: str) -> int | None:
        url = self.url(bucket, key)

        def op() -> int | None:
            try:
                with self._open("HEAD", url) as r:
                    return int(r.headers["Content-Length"])
            except R2Error as e:
                if e.status in (404, 403):
                    return None
                raise

        return self._retry(op)

    def presign_get(self, bucket: str, key: str, expires_s: int) -> str:
        return self.signer.presign("GET", self.url(bucket, key), expires_s)


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
