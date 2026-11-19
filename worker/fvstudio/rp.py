"""RunPod's serverless worker protocol with `urllib` only (no `runpod` SDK).

The protocol is the one runpod-python 1.7.13 speaks (`serverless/modules/
rp_job.py`, `rp_http.py`, `rp_ping.py`, `rp_progress.py`), as fastvideo-rs's
`fastvideo-deploy/src/runpod` and h3-rs's `h3 worker` do in Rust:

```text
take      GET  $RUNPOD_WEBHOOK_GET_JOB      ($ID → worker id)  &job_in_progress=0|1
            long poll; 204 / 400: no job; 429: wait 5 s; 200: {"id", "input"} or a list
progress  POST $RUNPOD_WEBHOOK_POST_OUTPUT  ($RUNPOD_POD_ID → worker id, $ID → job id) &isStream=false
            {"status": "IN_PROGRESS", "output": <progress object>}
done      POST $RUNPOD_WEBHOOK_POST_OUTPUT  (same URL)
            {"output": <handler output>} and / or {"error": <handler's error>}
            (no `status`: the platform marks the job COMPLETED or FAILED by the
            presence of `error`); 3 attempts, Fibonacci back-off 1, 1, 2 s
ping      GET  $RUNPOD_WEBHOOK_PING         ($RUNPOD_POD_ID → worker id) ?job_id=<running job>&runpod_version=…
            every $RUNPOD_PING_INTERVAL ms (default 10000)
headers   Authorization: $RUNPOD_AI_API_KEY (raw), X-Request-ID: <job id> on POSTs,
          Content-Type: application/x-www-form-urlencoded (the body is JSON text)
```

The URL templates are opaque: `$ID` / `$RUNPOD_POD_ID` are substituted and
query parameters appended with `?` or `&` as needed (they come with
`?gpu=…` already). A handler's dict output has its `error` key lifted out
beside `output` (`run_job`, like the SDK), so a `{"error", "where"}` result
is FAILED with that error and keeps `output.where`. Concurrency is 1: one
GPU, one job. `RUNPOD_WEBHOOK_GET_JOB` absent means local mode (`run_local`
runs one job from a file). SIGTERM / SIGINT stop the loop after the
current job.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from . import __version__

VERSION = f"fvstudio-worker/{__version__}"
Log = Callable[[str], None]


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def join_query(url: str, q: str) -> str:
    return f"{url}{'&' if '?' in url else '?'}{q}"


def _qenc(s: str) -> str:
    from urllib.parse import quote

    return quote(s, safe="")


@dataclass(frozen=True)
class RunpodEnv:
    """The variables RunPod injects into a serverless worker."""

    get_job: str
    post_output: str
    ping: str | None
    api_key: str
    worker_id: str
    ping_interval_s: float
    hostname: str | None

    @staticmethod
    def from_env(env: dict | None = None) -> "RunpodEnv | None":
        """`None` when `RUNPOD_WEBHOOK_GET_JOB` is unset (the SDK's local mode)."""
        e = os.environ if env is None else env

        def get(k: str) -> str | None:
            v = e.get(k)
            v = v.strip() if isinstance(v, str) else None
            return v or None

        get_job = get("RUNPOD_WEBHOOK_GET_JOB")
        if not get_job:
            return None
        post_output = get("RUNPOD_WEBHOOK_POST_OUTPUT")
        if not post_output:
            raise RuntimeError("RUNPOD_WEBHOOK_POST_OUTPUT is not set")
        try:
            ping_ms = int(get("RUNPOD_PING_INTERVAL") or 10000)
        except ValueError:
            ping_ms = 10000
        return RunpodEnv(
            get_job=get_job,
            post_output=post_output,
            ping=get("RUNPOD_WEBHOOK_PING"),
            api_key=get("RUNPOD_AI_API_KEY") or "",
            worker_id=get("RUNPOD_POD_ID") or f"local-{os.getpid()}",
            ping_interval_s=max(ping_ms, 100) / 1000.0,
            hostname=get("RUNPOD_POD_HOSTNAME") or get("HOSTNAME"),
        )

    def take_url(self, in_progress: bool = False) -> str:
        """`GET` job-take (`$ID` is the worker here)."""
        base = self.get_job.replace("$RUNPOD_POD_ID", self.worker_id).replace("$ID", self.worker_id)
        return join_query(base, f"job_in_progress={1 if in_progress else 0}")

    def done_url(self, job_id: str) -> str:
        """`POST` job-done, for progress and the result alike (`isStream=false`)."""
        base = self.post_output.replace("$RUNPOD_POD_ID", self.worker_id).replace("$ID", job_id)
        return join_query(base, "isStream=false")

    def ping_url(self, job_id: str | None) -> str | None:
        if not self.ping:
            return None
        u = self.ping.replace("$RUNPOD_POD_ID", self.worker_id)
        if job_id:
            u = join_query(u, f"job_id={_qenc(job_id)}")
        return join_query(u, f"runpod_version={_qenc(VERSION)}")


class TooManyRequests(Exception):
    pass


class RunpodClient:
    """The HTTP side: take, progress, done, ping. Progress goes out from a
    sender thread in order (the job is not held up by the platform);
    `flush_progress` drains it before the result is posted."""

    take_timeout_s = 90.0  # the SDK's job-take long poll
    post_timeout_s = 60.0
    done_attempts = 3
    done_backoff_s = (1.0, 1.0, 2.0)  # Fibonacci, like the SDK's aiohttp_retry

    def __init__(self, env: RunpodEnv, log: Log = _stderr, sleep: Callable[[float], None] = time.sleep):
        self.env = env
        self.log = log
        self.sleep = sleep
        self.current_job: str | None = None
        self._progress: queue.Queue = queue.Queue()
        self._sender: threading.Thread | None = None
        self.progress_sent = 0
        self.progress_failed = 0

    # ---- HTTP -------------------------------------------------------------------

    def _request(self, method: str, url: str, body: Any = None, job_id: str | None = None,
                 timeout: float | None = None) -> tuple[int, bytes]:
        headers = {"Authorization": self.env.api_key}
        data = None
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            headers["charset"] = "utf-8"
        if job_id is not None:
            headers["X-Request-ID"] = job_id
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.post_timeout_s) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def take(self) -> list[dict]:
        """One job-take: the jobs it returned (`[]` on 204 / 400); raises
        `TooManyRequests` on 429, `RuntimeError` on any other status and
        `urllib.error.URLError` / `OSError` on transport errors."""
        status, body = self._request("GET", self.env.take_url(self.current_job is not None), timeout=self.take_timeout_s)
        if status in (204, 400):  # 400 is expected under FlashBoot
            return []
        if status == 429:
            raise TooManyRequests()
        if not 200 <= status < 300:
            raise RuntimeError(f"job-take: HTTP {status}: {body[:200]!r}")
        if not body.strip():
            return []
        try:
            v = json.loads(body.decode("utf-8"))
        except ValueError as e:
            raise RuntimeError(f"job-take: not JSON: {e}") from None
        items = v if isinstance(v, list) else [v]
        jobs = []
        for j in items:
            if isinstance(j, dict) and "id" in j and "input" in j:
                jobs.append(j)
            else:
                self.log(f"worker: job-take returned a job without id / input: {str(j)[:200]}")
        return jobs

    def post_done(self, job_id: str, body: dict) -> bool:
        """The result, with retries; False when it was never accepted."""
        url = self.env.done_url(job_id)
        for attempt in range(self.done_attempts):
            try:
                status, text = self._request("POST", url, body, job_id=job_id)
                if 200 <= status < 300:
                    return True
                err: str = f"HTTP {status}: {text[:200]!r}"
            except (urllib.error.URLError, OSError) as e:
                err = str(e)
            self.log(f"worker: job {job_id}: posting the result failed (attempt {attempt + 1}/{self.done_attempts}): {err}")
            if attempt + 1 < self.done_attempts:
                self.sleep(self.done_backoff_s[min(attempt, len(self.done_backoff_s) - 1)])
        return False

    def ping(self) -> None:
        url = self.env.ping_url(self.current_job)
        if url is None:
            return
        try:
            status, _ = self._request("GET", url, timeout=self.env.ping_interval_s * 2)
            if not 200 <= status < 300:
                self.log(f"worker: ping: HTTP {status}")
        except (urllib.error.URLError, OSError) as e:
            self.log(f"worker: ping failed: {e}")

    # ---- progress ---------------------------------------------------------------

    def progress_update(self, job: dict, data: Any) -> None:
        """`runpod.serverless.progress_update`'s shape: the object goes out
        verbatim as `{"status": "IN_PROGRESS", "output": data}`."""
        if self._sender is None or not self._sender.is_alive():
            self._sender = threading.Thread(target=self._send_progress, name="rp-progress", daemon=True)
            self._sender.start()
        self._progress.put((str(job["id"]), data))

    def _send_progress(self) -> None:
        while True:
            job_id, data = self._progress.get()
            try:
                status, text = self._request("POST", self.env.done_url(job_id), {"status": "IN_PROGRESS", "output": data}, job_id=job_id)
                if 200 <= status < 300:
                    self.progress_sent += 1
                else:
                    self.progress_failed += 1
                    self.log(f"worker: job {job_id}: progress update: HTTP {status}: {text[:200]!r}")
            except (urllib.error.URLError, OSError) as e:
                self.progress_failed += 1
                self.log(f"worker: job {job_id}: progress update failed: {e}")
            finally:
                self._progress.task_done()

    def flush_progress(self) -> None:
        """Wait until every queued progress update has been posted."""
        self._progress.join()


# ---- jobs -------------------------------------------------------------------------


def run_job(handler: Callable[[dict], Any], job: dict, env: RunpodEnv | None = None) -> dict:
    """Runs `handler(job)` and shapes the job-done body like the SDK's
    `rp_job.run_job`: a dict output has `error` (and `refresh_worker`) lifted
    out, an empty output is dropped, an exception is `{"error": <json report>}`."""
    try:
        out = handler(job)
        if isinstance(out, dict):
            out = dict(out)
            error = out.pop("error", None)
            refresh = out.pop("refresh_worker", None)
            result: dict = {"output": out}
            if error:
                result["error"] = error
            if refresh:
                result["stopPod"] = True
        else:
            result = {"output": out}
        if result.get("output") == {}:
            result.pop("output")
        return result
    except Exception as e:  # noqa: BLE001
        report = {
            "error_type": str(type(e)),
            "error_message": str(e),
            "error_traceback": traceback.format_exc(),
            "hostname": (env.hostname if env else None) or os.environ.get("RUNPOD_POD_HOSTNAME", "unknown"),
            "worker_id": env.worker_id if env else os.environ.get("RUNPOD_POD_ID", "unknown"),
            "runpod_version": VERSION,
        }
        return {"error": json.dumps(report)}


def install_signal_handlers(stop: threading.Event, log: Log = _stderr) -> None:
    """SIGTERM / SIGINT set `stop`: the loop ends after the current job."""

    def on_signal(signum, frame):  # noqa: ARG001
        log(f"worker: {signal.Signals(signum).name} received: stopping after the current job")
        stop.set()

    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, on_signal)


def serve(handler: Callable[[dict], Any], client: RunpodClient, stop: threading.Event | None = None,
          max_jobs: int | None = None, idle_s: float = 0.2, log: Log | None = None) -> int:
    """The worker loop: take, run, post, until `stop` is set (or `max_jobs`
    ran). Returns the number of jobs run."""
    log = log or client.log
    stop = stop or threading.Event()
    env = client.env

    if env.ping:
        def ping_loop() -> None:
            while not stop.is_set():
                client.ping()
                stop.wait(env.ping_interval_s)

        threading.Thread(target=ping_loop, name="rp-ping", daemon=True).start()

    log(f"worker: {VERSION} as {env.worker_id}: waiting for jobs")
    done = 0
    backoff = 1.0
    while not stop.is_set():
        try:
            jobs = client.take()
            backoff = 1.0
        except TooManyRequests:
            log("worker: job-take: 429, waiting 5 s")
            client.sleep(5.0)
            continue
        except (urllib.error.URLError, OSError, RuntimeError) as e:
            log(f"worker: job-take failed: {e}; retrying in {backoff:.0f} s")
            client.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
            continue
        if not jobs:
            if idle_s:
                client.sleep(idle_s)
            continue
        for job in jobs:
            job_id = str(job["id"])
            client.current_job = job_id
            log(f"worker: job {job_id} started")
            t0 = time.time()
            result = run_job(handler, job, env)
            client.flush_progress()
            sent = client.post_done(job_id, result)
            client.current_job = None
            done += 1
            log(f"worker: job {job_id} {'failed' if 'error' in result else 'done'} in {time.time() - t0:.1f} s "
                f"({'result sent' if sent else 'result NOT accepted: the platform will time the job out'})")
            if max_jobs is not None and done >= max_jobs:
                stop.set()
                break
    log(f"worker: stopped after {done} job(s)")
    return done


def run_local(handler: Callable[[dict], Any], test_input: str | None, log: Log = _stderr, out=None) -> int:
    """Local mode (no `RUNPOD_WEBHOOK_GET_JOB`): one job from `test_input`
    (a file holding `{"input": …}` or the input itself; the SDK's
    `test_input.json` in the working directory when none is given), the
    result printed as JSON. Exit code 1 when it failed."""
    path = test_input or "test_input.json"
    if not os.path.isfile(path):
        log(f"worker: local mode: {path} not found (RUNPOD_WEBHOOK_GET_JOB is unset; pass --test-input FILE)")
        return 2
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    job = {"id": "local_test", "input": doc.get("input", doc) if isinstance(doc, dict) else doc}
    if isinstance(doc, dict) and "id" in doc:
        job["id"] = str(doc["id"])
    log(f"worker: local mode: running {path} as job {job['id']}")
    result = run_job(handler, job)
    print(json.dumps(result, indent=1, ensure_ascii=False), file=out or sys.stdout, flush=True)
    return 1 if "error" in result else 0
