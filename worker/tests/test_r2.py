"""`fvstudio.r2.R2Store`: the stdlib SigV4 signer against botocore's
(`S3SigV4Auth` / `S3SigV4QueryAuth`, when botocore is installed) for fixed
requests at a fixed timestamp, and the object operations against a tiny
in-process S3 (no signature check there)."""

from __future__ import annotations

import hashlib
import http.server
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, unquote, urlsplit

import pytest

from fvstudio.r2 import EMPTY_SHA256, UNSIGNED_PAYLOAD, R2Config, R2Error, R2Store, SigV4

NOW = datetime(2026, 11, 19, 12, 34, 56, tzinfo=timezone.utc)
ENDPOINT = "https://0123456789abcdef.r2.cloudflarestorage.com"
AK, SK = "AKIDEXAMPLE", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"


def store(endpoint: str = ENDPOINT) -> R2Store:
    return R2Store(R2Config(endpoint, AK, SK, "studio"), sleep=lambda s: None)


# ---- the signer against botocore ------------------------------------------------

# (name, method, bucket, key, query, headers, body, payload hash we send)
CASES = [
    ("get object", "GET", "studio", "jobs/abc-1/video.mp4", None, {}, None, EMPTY_SHA256),
    ("head object", "HEAD", "fv-weights", "weights/h3-8step/rev1/transformer/model-00001-of-00009.safetensors", None, {}, None, EMPTY_SHA256),
    ("key with spaces, unicode and reserved characters", "GET", "studio", "uploads/a b/ü (1)+x&y=z#!'*.png", None, {}, None, EMPTY_SHA256),
    ("list objects v2", "GET", "fv-weights", "", [("list-type", "2"), ("prefix", "manifests/h3 rs/")], {}, None, EMPTY_SHA256),
    ("list objects v2, continuation token", "GET", "fv-weights", "",
     [("list-type", "2"), ("prefix", "weights/"), ("continuation-token", "1ueGcxLPRx1Tr/XYExHnhbYLgveDs2J/wm36Hy4vbOwM=")], {}, None, EMPTY_SHA256),
    ("put bytes, signed payload", "PUT", "studio", "jobs/abc-1/job.json", None, {"Content-Type": "application/json"}, b'{"output": {"model": "fasth3-v2"}}\n',
     hashlib.sha256(b'{"output": {"model": "fasth3-v2"}}\n').hexdigest()),
    ("put file, unsigned payload", "PUT", "studio", "jobs/abc-1/video.mp4", None, {"Content-Type": "video/mp4"}, b"\x00\x00\x00\x18ftypisom", UNSIGNED_PAYLOAD),
    ("header value with inner whitespace", "PUT", "studio", "k", None, {"Content-Type": "text/plain;  charset=utf-8 "}, b"x", UNSIGNED_PAYLOAD),
]


@pytest.fixture
def botocore_clock(monkeypatch):
    botocore_auth = pytest.importorskip("botocore.auth")
    monkeypatch.setattr(botocore_auth, "get_current_datetime", lambda: NOW.replace(tzinfo=None))
    return botocore_auth


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_header_auth_matches_botocore(case, botocore_clock):
    from botocore.awsrequest import AWSRequest
    from botocore.credentials import Credentials

    _, method, bucket, key, query, headers, body, payload_hash = case
    s = store()
    url = s.url(bucket, key, query)
    ours = s.signer.sign_headers(method, url, headers, payload_hash, now=NOW)

    req = AWSRequest(method=method, url=url, headers=dict(headers), data=body)
    if payload_hash == UNSIGNED_PAYLOAD:
        req.context["client_config"] = SimpleNamespace(s3={"payload_signing_enabled": False})
    botocore_clock.S3SigV4Auth(Credentials(AK, SK), "s3", "auto").add_auth(req)

    assert req.headers["X-Amz-Date"] == ours["X-Amz-Date"] == "20261119T123456Z"
    assert req.headers["X-Amz-Content-SHA256"] == ours["X-Amz-Content-SHA256"] == payload_hash
    assert req.headers["Authorization"] == ours["Authorization"]
    # Nothing we send is unsigned, apart from Authorization itself.
    signed = ours["Authorization"].split("SignedHeaders=")[1].split(",")[0].split(";")
    assert sorted(k.lower() for k in ours if k != "Authorization") == signed


PRESIGN_CASES = [
    ("video", "studio", "jobs/abc-1/video.mp4", 86400),
    ("unicode key", "studio", "uploads/ü (1)+x&y.png", 600),
    ("weights shard", "fv-weights", "weights/fastwan21-1.3b/rev2/model.safetensors", 1),
]


@pytest.mark.parametrize("case", PRESIGN_CASES, ids=[c[0] for c in PRESIGN_CASES])
def test_presigned_get_matches_botocore(case, botocore_clock):
    from botocore.awsrequest import AWSRequest
    from botocore.credentials import Credentials

    _, bucket, key, expires = case
    s = store()
    ours = s.signer.presign("GET", s.url(bucket, key), expires, now=NOW)

    req = AWSRequest(method="GET", url=s.url(bucket, key))
    botocore_clock.S3SigV4QueryAuth(Credentials(AK, SK), "s3", "auto", expires=expires).add_auth(req)
    assert req.url == ours

    q = parse_qs(urlsplit(ours).query)
    assert q["X-Amz-Algorithm"] == ["AWS4-HMAC-SHA256"]
    assert q["X-Amz-Credential"] == [f"{AK}/20261119/auto/s3/aws4_request"]
    assert q["X-Amz-Expires"] == [str(expires)] and q["X-Amz-SignedHeaders"] == ["host"]
    assert len(q["X-Amz-Signature"][0]) == 64


def test_presign_get_is_the_query_form():
    s = store()
    url = s.presign_get("studio", "jobs/x/video.mp4", 3600)
    assert url.startswith(f"{ENDPOINT}/studio/jobs/x/video.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=")
    assert "&X-Amz-Signature=" in url and "X-Amz-Expires=3600" in url


def test_signing_key_matches_the_aws_example():
    # docs.aws.amazon.com/general/latest/gr/signature-v4-examples.html
    sig = SigV4(AK, SK, region="us-east-1", service="iam")
    assert sig.signing_key("20150830").hex() == "c4afb1cc5771d871763a393e44b703571b55cc28424d1a5e86da6ed3c154a4b9"


# ---- the operations against a fake S3 ----------------------------------------------


class FakeS3(http.server.BaseHTTPRequestHandler):
    """`PUT /b/k`, `GET /b/k`, `HEAD /b/k`, `GET /b?list-type=2&prefix=…` with
    two keys per page. Objects in `server.objects[(bucket, key)]`, every
    request's (method, path, lower-cased headers) in `server.log`."""

    page = 2

    def log_message(self, *a):  # quiet
        pass

    def _split(self):
        u = urlsplit(self.path)
        parts = u.path.lstrip("/").split("/", 1)
        return unquote(parts[0]), unquote(parts[1]) if len(parts) > 1 else "", parse_qs(u.query)

    def do_PUT(self):
        bucket, key, _ = self._split()
        n = int(self.headers["Content-Length"])
        body = self.rfile.read(n)
        self.server.log.append(("PUT", self.path, {k.lower(): v for k, v in self.headers.items()}))
        self.server.objects[(bucket, key)] = (body, self.headers.get("Content-Type"))
        self.send_response(200)
        self.send_header("ETag", '"' + hashlib.md5(body).hexdigest() + '"')
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _object(self, head: bool):
        bucket, key, _ = self._split()
        self.server.log.append(("HEAD" if head else "GET", self.path, {k.lower(): v for k, v in self.headers.items()}))
        if self.server.fail_next:
            self.server.fail_next -= 1
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        o = self.server.objects.get((bucket, key))
        if o is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body, ctype = o
        self.send_response(200)
        self.send_header("Content-Type", ctype or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not head:
            for i in range(0, len(body), 65536):
                self.wfile.write(body[i:i + 65536])

    def do_HEAD(self):
        self._object(head=True)

    def do_GET(self):
        bucket, key, q = self._split()
        if key or q.get("list-type") != ["2"]:
            return self._object(head=False)
        self.server.log.append(("GET", self.path, {k.lower(): v for k, v in self.headers.items()}))
        prefix = q.get("prefix", [""])[0]
        keys = sorted(k for (b, k) in self.server.objects if b == bucket and k.startswith(prefix))
        start = int(q.get("continuation-token", ["0"])[0])
        page = keys[start:start + self.page]
        truncated = start + self.page < len(keys)
        xml = ['<?xml version="1.0" encoding="UTF-8"?>',
               '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">',
               f"<Name>{bucket}</Name><Prefix>{prefix}</Prefix><KeyCount>{len(page)}</KeyCount><MaxKeys>{self.page}</MaxKeys>",
               f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>"]
        if truncated:
            xml.append(f"<NextContinuationToken>{start + self.page}</NextContinuationToken>")
        for k in page:
            xml.append(f"<Contents><Key>{k}</Key><Size>{len(self.server.objects[(bucket, k)][0])}</Size><StorageClass>STANDARD</StorageClass></Contents>")
        xml.append("</ListBucketResult>")
        body = "\n".join(xml).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/xml")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def fake_s3():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeS3)
    srv.objects = {}
    srv.log = []
    srv.fail_next = 0
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def test_round_trip_put_get_head_list(fake_s3, tmp_path: Path):
    s = store(f"http://127.0.0.1:{fake_s3.server_address[1]}")
    big = tmp_path / "video.mp4"
    big.write_bytes(os.urandom(3 * 1024 * 1024 + 123))
    small = tmp_path / "job.json"
    small.write_text('{"a": 1}\n')

    s.put_file("studio", "jobs/j1/video.mp4", big)
    s.put_file("studio", "jobs/j1/job.json", small, content_type="application/json")
    s.put_file("studio", "jobs/j1/ü thumb.jpg", small)
    s.put_file("studio", "uploads/x.bin", small)
    put = [e for e in fake_s3.log if e[0] == "PUT"]
    assert put[0][1] == "/studio/jobs/j1/video.mp4" and put[0][2]["content-type"] == "video/mp4"
    assert put[0][2]["x-amz-content-sha256"] == UNSIGNED_PAYLOAD and put[0][2]["content-length"] == str(big.stat().st_size)
    assert put[0][2]["authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/")
    assert put[1][2]["content-type"] == "application/json" and put[2][1] == "/studio/jobs/j1/%C3%BC%20thumb.jpg"
    assert fake_s3.objects[("studio", "jobs/j1/video.mp4")][0] == big.read_bytes()

    assert s.head_size("studio", "jobs/j1/video.mp4") == big.stat().st_size
    assert s.head_size("studio", "jobs/j1/missing.mp4") is None
    assert s.get_bytes("studio", "jobs/j1/job.json") == b'{"a": 1}\n'

    out = tmp_path / "dl" / "video.mp4"
    assert s.get_file("studio", "jobs/j1/video.mp4", out) == big.stat().st_size
    assert out.read_bytes() == big.read_bytes() and not out.with_name("video.mp4.part").exists()

    assert list(s.list_keys("studio", "jobs/j1/")) == [
        ("jobs/j1/job.json", 9), ("jobs/j1/video.mp4", big.stat().st_size), ("jobs/j1/ü thumb.jpg", 9)]
    lists = [e for e in fake_s3.log if e[0] == "GET" and "list-type=2" in e[1]]
    assert len(lists) == 2 and "continuation-token=2" in lists[1][1] and "prefix=jobs%2Fj1%2F" in lists[0][1]
    assert list(s.list_keys("studio", "nothing/")) == []


def test_get_file_404_and_retry_on_503(fake_s3, tmp_path: Path):
    s = store(f"http://127.0.0.1:{fake_s3.server_address[1]}")
    with pytest.raises(R2Error) as e:
        s.get_file("studio", "jobs/none", tmp_path / "none")
    assert e.value.status == 404 and not (tmp_path / "none.part").exists()
    assert not list(tmp_path.glob("*"))

    fake_s3.objects[("studio", "k")] = (b"hello", "text/plain")
    fake_s3.fail_next = 2
    assert s.get_bytes("studio", "k") == b"hello"
    assert len([e for e in fake_s3.log if e[1] == "/studio/k"]) == 3

    fake_s3.fail_next = 10
    with pytest.raises(R2Error) as e:
        s.get_bytes("studio", "k")
    assert e.value.status == 503
