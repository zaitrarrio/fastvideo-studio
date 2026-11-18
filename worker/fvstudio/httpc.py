"""A tiny urllib client for the sidecars on 127.0.0.1 (no `Origin` header:
`h3 serve` / `fastwan serve` refuse foreign origins)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class HttpError(Exception):
    def __init__(self, status: int, body: str, url: str):
        super().__init__(f"HTTP {status} from {url}: {body[:300]}")
        self.status = status
        self.body = body
        self.url = url

    def json(self) -> Any:
        try:
            return json.loads(self.body)
        except ValueError:
            return None


def _do(req: urllib.request.Request, timeout: float) -> tuple[int, bytes, dict]:
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        body = e.read()
        raise HttpError(e.code, body.decode("utf-8", "replace"), req.full_url) from None


def get_bytes(url: str, timeout: float = 120) -> bytes:
    return _do(urllib.request.Request(url, method="GET"), timeout)[1]


def get_json(url: str, timeout: float = 30) -> Any:
    body = get_bytes(url, timeout)
    return json.loads(body.decode("utf-8"))


def get_status(url: str, timeout: float = 10) -> tuple[int, bytes]:
    """Status and body without raising on 4xx/5xx."""
    try:
        s, b, _ = _do(urllib.request.Request(url, method="GET"), timeout)
        return s, b
    except HttpError as e:
        return e.status, e.body.encode()


def post_json(url: str, body: Any, timeout: float = 60, headers: dict | None = None) -> Any:
    data = json.dumps(body).encode("utf-8")
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=data, method="POST", headers=h)
    _, out, _ = _do(req, timeout)
    return json.loads(out.decode("utf-8")) if out.strip() else None


def delete(url: str, timeout: float = 30) -> None:
    try:
        _do(urllib.request.Request(url, method="DELETE"), timeout)
    except HttpError as e:
        if e.status != 404:
            raise
