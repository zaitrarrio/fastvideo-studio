"""`fvstudio.rp`: RunPod's serverless protocol against a fake RunPod
(http.server in a thread) that hands out jobs, records the takes, the
progress and done posts with their headers, and the pings."""

from __future__ import annotations

import http.server
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from fvstudio import rp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeRunpod(http.server.BaseHTTPRequestHandler):
    """`GET /v2/ep/job-take/<worker>`: the next of `server.jobs` (200) or 204;
    `server.take_statuses` forces statuses on the first takes. `POST
    /v2/ep/job-done/<worker>/<job>`: recorded; the first `server.fail_done`
    posts with a `status` other than IN_PROGRESS get a 500. `GET
    /v2/ep/ping/<worker>`: recorded."""

    def log_message(self, *a):
        pass

    def _reply(self, status: int, body: bytes = b"", ctype: str = "application/json"):
        self.send_response(status)
        if body:
            self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):
        u = urlsplit(self.path)
        q = parse_qs(u.query)
        h = {k.lower(): v for k, v in self.headers.items()}
        if "/job-take/" in u.path:
            self.server.takes.append((u.path, q, h))
            if self.server.take_statuses:
                return self._reply(self.server.take_statuses.pop(0))
            if self.server.jobs:
                job = self.server.jobs.pop(0)
                self.server.taken.append(job["id"] if isinstance(job, dict) else "<list>")
                return self._reply(200, json.dumps(job).encode())
            self.server.idle.set()
            return self._reply(204)
        if "/ping/" in u.path:
            self.server.pings.append((u.path, q, h))
            return self._reply(200, b'{"status": "ok"}')
        self._reply(404)

    def do_POST(self):
        u = urlsplit(self.path)
        q = parse_qs(u.query)
        h = {k.lower(): v for k, v in self.headers.items()}
        body = json.loads(self.rfile.read(int(h["content-length"])).decode("utf-8"))
        self.server.posts.append((u.path, q, h, body))
        if body.get("status") != "IN_PROGRESS" and self.server.fail_done > 0:
            self.server.fail_done -= 1
            return self._reply(500, b"nope", "text/plain")
        self._reply(200, b'{"status": "ok"}')


@pytest.fixture
def fake():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeRunpod)
    srv.jobs, srv.taken, srv.takes, srv.posts, srv.pings = [], [], [], [], []
    srv.take_statuses, srv.fail_done = [], 0
    srv.idle = threading.Event()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}/v2/ep1"
    srv.env = rp.RunpodEnv.from_env({
        "RUNPOD_WEBHOOK_GET_JOB": f"{base}/job-take/$ID?gpu=NVIDIA+H200",
        "RUNPOD_WEBHOOK_POST_OUTPUT": f"{base}/job-done/$RUNPOD_POD_ID/$ID?gpu=NVIDIA+H200",
        "RUNPOD_WEBHOOK_POST_STREAM": f"{base}/job-stream/$RUNPOD_POD_ID/$ID?gpu=NVIDIA+H200",
        "RUNPOD_WEBHOOK_PING": f"{base}/ping/$RUNPOD_POD_ID",
        "RUNPOD_PING_INTERVAL": "100",
        "RUNPOD_AI_API_KEY": "secret-key",
        "RUNPOD_POD_ID": "w42",
        "RUNPOD_POD_HOSTNAME": "host-1",
    })
    yield srv
    srv.shutdown()
    srv.server_close()


def client(srv, logs: list | None = None) -> rp.RunpodClient:
    c = rp.RunpodClient(srv.env, log=(logs.append if logs is not None else lambda m: None), sleep=lambda s: None)
    return c


def done_posts(srv):
    return [p for p in srv.posts if p[3].get("status") != "IN_PROGRESS"]


def progress_posts(srv):
    return [p for p in srv.posts if p[3].get("status") == "IN_PROGRESS"]


# ---- URL templates ----------------------------------------------------------------


def test_url_templates(fake):
    e = fake.env
    base = e.get_job.split("/job-take/")[0]
    assert e.take_url(False) == f"{base}/job-take/w42?gpu=NVIDIA+H200&job_in_progress=0"
    assert e.take_url(True) == f"{base}/job-take/w42?gpu=NVIDIA+H200&job_in_progress=1"
    assert e.done_url("j1") == f"{base}/job-done/w42/j1?gpu=NVIDIA+H200&isStream=false"
    assert e.ping_url("a/b") == f"{base}/ping/w42?job_id=a%2Fb&runpod_version={rp.VERSION.replace('/', '%2F')}"
    assert e.ping_url(None) == f"{base}/ping/w42?runpod_version={rp.VERSION.replace('/', '%2F')}"
    assert e.ping_interval_s == 0.1 and e.api_key == "secret-key" and e.hostname == "host-1"


def test_env_without_query_and_defaults():
    e = rp.RunpodEnv.from_env({
        "RUNPOD_WEBHOOK_GET_JOB": "https://api.runpod.ai/v2/ep/job-take/$ID",
        "RUNPOD_WEBHOOK_POST_OUTPUT": "https://api.runpod.ai/v2/ep/job-done/$RUNPOD_POD_ID/$ID",
    })
    assert e is not None
    assert e.take_url() == f"https://api.runpod.ai/v2/ep/job-take/{e.worker_id}?job_in_progress=0"
    assert e.worker_id.startswith("local-") and e.ping is None and e.ping_url("x") is None
    assert e.done_url("j") == f"https://api.runpod.ai/v2/ep/job-done/{e.worker_id}/j?isStream=false"
    assert e.ping_interval_s == 10.0 and e.api_key == ""


def test_local_mode_when_get_job_is_unset():
    assert rp.RunpodEnv.from_env({}) is None
    assert rp.RunpodEnv.from_env({"RUNPOD_WEBHOOK_GET_JOB": "  "}) is None
    with pytest.raises(RuntimeError, match="RUNPOD_WEBHOOK_POST_OUTPUT"):
        rp.RunpodEnv.from_env({"RUNPOD_WEBHOOK_GET_JOB": "https://x/job-take/$ID"})


# ---- the loop ---------------------------------------------------------------------


def test_one_job_round_trip(fake):
    fake.jobs.append({"id": "job-1", "input": {"engine": "h3", "prompt": "a fox"}, "webhook": None})
    c = client(fake)
    seen = []

    def handler(job):
        seen.append(job)
        c.progress_update(job, {"stage": "staging", "elapsed_s": 0.1})
        c.progress_update(job, {"stage": "denoising", "step": 3, "total_steps": 8, "ünïcode": "✓"})
        time.sleep(0.25)  # a ping or two while the job runs
        return {"video_url": "https://r2/v.mp4", "frames": 124}

    n = rp.serve(handler, c, max_jobs=1, idle_s=0)
    assert n == 1 and seen == [{"id": "job-1", "input": {"engine": "h3", "prompt": "a fox"}, "webhook": None}]

    # take: $ID → worker id, the template's query kept, job_in_progress appended
    path, q, h = fake.takes[0]
    assert path == "/v2/ep1/job-take/w42" and q == {"gpu": ["NVIDIA H200"], "job_in_progress": ["0"]}
    assert h["authorization"] == "secret-key"

    # progress: in order, before the result, verbatim objects
    prog = progress_posts(fake)
    assert [p[3]["output"] for p in prog] == [{"stage": "staging", "elapsed_s": 0.1},
                                             {"stage": "denoising", "step": 3, "total_steps": 8, "ünïcode": "✓"}]
    for path, q, h, body in prog:
        assert path == "/v2/ep1/job-done/w42/job-1" and q == {"gpu": ["NVIDIA H200"], "isStream": ["false"]}
        assert h["authorization"] == "secret-key" and h["x-request-id"] == "job-1"
        assert h["content-type"] == "application/x-www-form-urlencoded" and body["status"] == "IN_PROGRESS"
    assert fake.posts.index(prog[-1]) < fake.posts.index(done_posts(fake)[0])

    # done: {"output": …}, no status, isStream=false, same headers
    (path, q, h, body), = done_posts(fake)
    assert path == "/v2/ep1/job-done/w42/job-1" and q["isStream"] == ["false"] and q["gpu"] == ["NVIDIA H200"]
    assert body == {"output": {"video_url": "https://r2/v.mp4", "frames": 124}}
    assert h["authorization"] == "secret-key" and h["x-request-id"] == "job-1" and h["charset"] == "utf-8"

    # ping: $RUNPOD_POD_ID → worker id, the running job's id, runpod_version
    assert fake.pings and all(p[0] == "/v2/ep1/ping/w42" and p[2]["authorization"] == "secret-key" for p in fake.pings)
    assert any(p[1].get("job_id") == ["job-1"] for p in fake.pings), fake.pings
    assert all(p[1]["runpod_version"] == [rp.VERSION] for p in fake.pings)
    assert c.progress_sent == 2 and c.progress_failed == 0


def test_handler_error_posts_failed_with_the_error_lifted_out(fake):
    fake.jobs.append({"id": "bad-1", "input": {"frames": 100}})
    c = client(fake)
    rp.serve(lambda job: {"error": "frames 100 is not on the 17n+5 grid", "where": "validate"}, c, max_jobs=1, idle_s=0)
    (_, q, h, body), = done_posts(fake)
    assert body == {"error": "frames 100 is not on the 17n+5 grid", "output": {"where": "validate"}}
    assert q["isStream"] == ["false"] and h["x-request-id"] == "bad-1"


def test_handler_exception_is_a_failed_job_with_the_sdk_report(fake):
    fake.jobs.append({"id": "boom-1", "input": {}})
    logs = []

    def handler(job):
        raise RuntimeError("engine died")

    rp.serve(handler, client(fake, logs), max_jobs=1, idle_s=0)
    (_, _, _, body), = done_posts(fake)
    assert set(body) == {"error"}
    report = json.loads(body["error"])
    assert report["error_message"] == "engine died" and "RuntimeError" in report["error_type"]
    assert "Traceback" in report["error_traceback"] and report["worker_id"] == "w42" and report["hostname"] == "host-1"
    assert report["runpod_version"] == rp.VERSION
    assert any("boom-1 failed" in m for m in logs)


def test_run_job_shapes():
    env = rp.RunpodEnv.from_env({"RUNPOD_WEBHOOK_GET_JOB": "u/$ID", "RUNPOD_WEBHOOK_POST_OUTPUT": "u/$ID"})
    job = {"id": "j", "input": {}}
    out = {"error": "x", "where": "validate", "refresh_worker": True}
    assert rp.run_job(lambda j: out, job, env) == {"output": {"where": "validate"}, "error": "x", "stopPod": True}
    assert out == {"error": "x", "where": "validate", "refresh_worker": True}  # the handler's dict is not mutated
    assert rp.run_job(lambda j: {}, job, env) == {}
    assert rp.run_job(lambda j: {"error": ""}, job, env) == {}
    assert rp.run_job(lambda j: [1, 2], job, env) == {"output": [1, 2]}
    assert rp.run_job(lambda j: True, job, env) == {"output": True}


def test_done_post_retries_with_backoff(fake):
    fake.jobs.append({"id": "r-1", "input": {}})
    fake.fail_done = 2
    sleeps, logs = [], []
    c = rp.RunpodClient(fake.env, log=logs.append, sleep=sleeps.append)
    rp.serve(lambda job: {"ok": 1}, c, max_jobs=1, idle_s=0)
    posts = done_posts(fake)
    assert len(posts) == 3 and all(p[3] == {"output": {"ok": 1}} and p[2]["x-request-id"] == "r-1" for p in posts)
    assert sleeps == [1.0, 1.0]
    assert sum("attempt 1/3" in m for m in logs) == 1 and sum("attempt 2/3" in m for m in logs) == 1
    assert any("r-1 done" in m and "result sent" in m for m in logs)


def test_done_post_gives_up_after_three_attempts(fake):
    fake.jobs.append({"id": "r-2", "input": {}})
    fake.fail_done = 5
    sleeps, logs = [], []
    c = rp.RunpodClient(fake.env, log=logs.append, sleep=sleeps.append)
    assert rp.serve(lambda job: {"ok": 1}, c, max_jobs=1, idle_s=0) == 1
    assert len(done_posts(fake)) == 3 and sleeps == [1.0, 1.0]
    assert any("result NOT accepted" in m for m in logs)
    assert fake.fail_done == 2  # no fourth attempt


def test_take_backs_off_on_429_and_errors(fake):
    fake.take_statuses = [429, 500, 500]
    fake.jobs.append({"id": "t-1", "input": {}})
    sleeps, logs = [], []
    c = rp.RunpodClient(fake.env, log=logs.append, sleep=sleeps.append)
    rp.serve(lambda job: {"ok": 1}, c, max_jobs=1, idle_s=0)
    assert sleeps == [5.0, 1.0, 2.0] and len(fake.takes) == 4 and fake.taken == ["t-1"]
    assert any("429" in m for m in logs) and any("HTTP 500" in m for m in logs)


def test_take_ignores_malformed_jobs_and_lists(fake):
    fake.jobs.append([{"id": "l-1", "input": {"a": 1}}, {"input": {"no": "id"}}, "junk"])
    logs = []
    c = client(fake, logs)
    jobs = c.take()
    assert jobs == [{"id": "l-1", "input": {"a": 1}}] and sum("without id" in m for m in logs) == 2


def test_stop_ends_the_loop_after_the_current_job(fake):
    fake.jobs += [{"id": "s-1", "input": {}}, {"id": "s-2", "input": {}}]
    stop = threading.Event()
    c = client(fake)

    def handler(job):
        stop.set()  # as SIGTERM would, mid-job
        return {"done": job["id"]}

    assert rp.serve(handler, c, stop=stop, idle_s=0) == 1
    assert fake.taken == ["s-1"] and fake.jobs == [{"id": "s-2", "input": {}}]
    assert done_posts(fake)[0][3] == {"output": {"done": "s-1"}}


def test_signal_handlers_set_stop():
    stop = threading.Event()
    old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    logs = []
    try:
        rp.install_signal_handlers(stop, log=logs.append)
        signal.raise_signal(signal.SIGTERM)
        assert stop.wait(2) and logs == ["worker: SIGTERM received: stopping after the current job"]
    finally:
        for s, h in old.items():
            signal.signal(s, h)


def test_job_in_progress_flag_follows_the_running_job(fake):
    c = client(fake)
    assert "job_in_progress=0" in fake.env.take_url(c.current_job is not None)
    c.current_job = "x"
    assert "job_in_progress=1" in fake.env.take_url(c.current_job is not None)


# ---- local mode -----------------------------------------------------------------


def test_run_local(tmp_path: Path, capsys):
    f = tmp_path / "job.json"
    f.write_text(json.dumps({"input": {"engine": "h3", "prompt": "a fox"}}))
    logs = []
    seen = []

    def handler(job):
        seen.append(job)
        return {"frames": 124}

    assert rp.run_local(handler, str(f), log=logs.append) == 0
    assert seen == [{"id": "local_test", "input": {"engine": "h3", "prompt": "a fox"}}]
    assert json.loads(capsys.readouterr().out) == {"output": {"frames": 124}}

    f.write_text(json.dumps({"id": "j7", "input": {"op": "x"}}))
    assert rp.run_local(lambda job: {"error": "unknown op", "where": "validate"}, str(f), log=logs.append) == 1
    assert json.loads(capsys.readouterr().out) == {"output": {"where": "validate"}, "error": "unknown op"}
    assert rp.run_local(handler, str(tmp_path / "missing.json"), log=logs.append) == 2


def test_handler_main_runs_a_local_job(tmp_path: Path, monkeypatch, capsys):
    """`python handler.py --test-input job.json` without RunPod: one job
    through the real `Worker` (validation only), no SDK."""
    repo = Path(__file__).resolve().parents[2]
    for k in list(os.environ):
        if k.startswith("RUNPOD_"):
            monkeypatch.delenv(k)
    monkeypatch.setenv("STUDIO_CATALOG", str(repo / "catalog" / "catalog.json"))
    monkeypatch.setenv("STUDIO_R2_LOCAL_DIR", str(tmp_path / "r2"))
    monkeypatch.setenv("STUDIO_VOLUME", str(tmp_path / "vol"))
    monkeypatch.setenv("STUDIO_JOBS_DIR", str(tmp_path / "jobs"))
    import handler as handler_mod

    assert "runpod" not in sys.modules and "boto3" not in sys.modules
    f = tmp_path / "job.json"
    f.write_text(json.dumps({"input": {"op": "nope"}}))
    assert handler_mod.main(["--test-input", str(f)]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["output"]["where"] == "validate" and "nope" in out["error"]
