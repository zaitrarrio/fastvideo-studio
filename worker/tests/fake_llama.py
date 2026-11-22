"""A tiny in-process fake of llama-server's HTTP API: `GET /health` (503
while "loading", 200 after) and `POST /v1/chat/completions` answering a
canned rewrite. Every request body is recorded so tests can assert the
system prompt, the model alias, the mode's wording and the image part.
`fail_status` makes completions answer that status; `hang_s` makes them
sleep (a client timeout)."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeLlama:
    def __init__(self, answer: str = "A red fox trots across fresh snow at dawn.", loading: bool = False,
                 fail_status: int | None = None, hang_s: float = 0.0):
        self.answer = answer
        self.loading = loading
        self.fail_status = fail_status
        self.hang_s = hang_s
        self.requests: list[dict] = []
        self.health_calls = 0
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
                if self.path == "/health":
                    fake.health_calls += 1
                    if fake.loading:
                        return self._json(503, {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}})
                    return self._json(200, {"status": "ok"})
                self._json(404, {"error": "not found"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                try:
                    body = json.loads(raw.decode("utf-8")) if raw else {}
                except ValueError:
                    return self._json(400, {"error": "bad json"})
                if self.path != "/v1/chat/completions":
                    return self._json(404, {"error": "not found"})
                fake.requests.append(body)
                if fake.hang_s:
                    time.sleep(fake.hang_s)
                if fake.fail_status:
                    return self._json(fake.fail_status, {"error": {"code": fake.fail_status, "message": "boom"}})
                answer = fake.answer
                if body.get("max_tokens") == 1:
                    answer = "ok"  # the readiness probe's 1-token completion
                self._json(200, {
                    "id": "chatcmpl-fake", "object": "chat.completion", "created": int(time.time()),
                    "model": body.get("model", "fake"),
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                })

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def start(self) -> "FakeLlama":
        self._thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def __enter__(self) -> "FakeLlama":
        return self.start()

    def __exit__(self, *a) -> None:
        self.stop()

    @property
    def last(self) -> dict:
        return self.requests[-1]

    def system_of(self, req: dict | None = None) -> str:
        req = req or self.last
        return next(m["content"] for m in req["messages"] if m["role"] == "system")

    def user_of(self, req: dict | None = None):
        req = req or self.last
        return next(m["content"] for m in req["messages"] if m["role"] == "user")
