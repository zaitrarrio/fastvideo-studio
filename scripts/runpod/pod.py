#!/usr/bin/env python3
"""The fastvideo-studio pod backend (docs/pod-mode.md) from a shell: the same
RunPod REST v2 calls the Studio Worker makes, for ops and validation without
the Worker (the page's Pod switch does the same through `/api/pod`).

    export RUNPOD_API_KEY=…  STUDIO_IMAGE=ghcr.io/zaitrarrio/fastvideo-studio-worker:sha-…
    python3 scripts/runpod/pod.py plan                       # the create body, nothing sent
    python3 scripts/runpod/pod.py up [--hours H] [--gpu ID] [--disk GB]
    python3 scripts/runpod/pod.py status                     # RunPod's view + the pod's /health
    python3 scripts/runpod/pod.py wait                       # until /health answers (the serve is up)
    python3 scripts/runpod/pod.py health | info | jobs
    python3 scripts/runpod/pod.py run '{"engine":"h3","model":"h3-plug4","prompt":"a fox"}'   # a job on the pod, polled
    python3 scripts/runpod/pod.py cancel <job id>
    python3 scripts/runpod/pod.py down [--yes]

The pod runs the worker image as `studio-worker serve` with the endpoint's
env (the account's `{{ RUNPOD_SECRET_* }}` R2 references, STUDIO_IMAGE), a
random per-pod bearer token (STUDIO_SERVE_TOKEN), the wall-clock cap
(STUDIO_POD_HOURS: the pod terminates itself through REST v2 at it) and, when
CF_ACCOUNT_ID + CF_STREAM_API_TOKEN are set, Cloudflare Stream for live
streams. The ledger is ./.runpod-pod.json (STUDIO_POD_STATE): id, url, token.
Env defaults: POD_GPU (the RTX PRO 6000 Blackwell Server Edition), POD_DC
(EU-RO-1), POD_VOLUME (bz0ofkfx45, at /runpod-volume), POD_DISK_GB (300),
POD_HOURS (4), POD_MIN_CUDA (13.0), STUDIO_R2_ENDPOINT, STUDIO_R2_BUCKET.
"""

import argparse
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REST = os.environ.get("RUNPOD_API_BASE", "https://api.runpod.io/v2")
STATE = Path(os.environ.get("STUDIO_POD_STATE", ".runpod-pod.json"))
DEFAULTS = {
    "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "dc": "EU-RO-1",
    "volume": "bz0ofkfx45",
    "disk": 300,
    "hours": 4.0,
    "min_cuda": "13.0",
}


def key() -> str:
    k = os.environ.get("RUNPOD_API_KEY", "")
    if not k:
        sys.exit("RUNPOD_API_KEY is not set")
    return k


def http(method: str, url: str, body: dict | None = None, timeout: float = 120, bearer: str | None = None, quiet: bool = False) -> dict | list | None:
    data = json.dumps(body).encode() if body is not None else None
    if not quiet:
        print(f"→ {method} {url}" + (f"\n{json.dumps(body, indent=1)}" if body is not None and method != "POST" or (body is not None and 'env' not in body) else ""), file=sys.stderr)
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {bearer or key()}", "Content-Type": "application/json", "User-Agent": "fastvideo-studio/1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        raw = e.read()
        raise RuntimeError(f"{method} {url}: HTTP {e.code}: {raw.decode(errors='replace')[:800]}") from None
    return json.loads(raw) if raw.strip() else None


def rest(method: str, path: str, body: dict | None = None, timeout: float = 120) -> dict | list | None:
    return http(method, f"{REST}{path}", body, timeout)


# ---------------------------------------------------------------- ledger

def state() -> dict:
    return json.loads(STATE.read_text()) if STATE.is_file() else {}


def save(d: dict) -> None:
    STATE.write_text(json.dumps(d, indent=1) + "\n")
    print(f"ledger → {STATE}", file=sys.stderr)


def current() -> dict:
    s = state()
    if not s.get("id"):
        sys.exit(f"no pod in {STATE}: run `up` first")
    return s


def pod_url(pod_id: str) -> str:
    return f"https://{pod_id}-8000.proxy.runpod.net"


# ---------------------------------------------------------------- the create body (the Worker's podCreateBody)

def settings(a: argparse.Namespace | None = None) -> dict:
    e = os.environ
    s = {
        "image": (a and a.image) or e.get("STUDIO_IMAGE", ""),
        "gpu": (a and a.gpu) or e.get("POD_GPU", DEFAULTS["gpu"]),
        "dc": (a and a.dc) or e.get("POD_DC", DEFAULTS["dc"]),
        "volume": (a and a.volume) or e.get("POD_VOLUME", DEFAULTS["volume"]),
        "disk": int((a and a.disk) or e.get("POD_DISK_GB", DEFAULTS["disk"])),
        "hours": float((a and a.hours) or e.get("POD_HOURS", DEFAULTS["hours"])),
        "min_cuda": e.get("POD_MIN_CUDA", DEFAULTS["min_cuda"]),
    }
    if not s["image"]:
        sys.exit("no image: pass --image or set STUDIO_IMAGE (ghcr.io/zaitrarrio/fastvideo-studio-worker:sha-…)")
    if not (0 < s["hours"] <= 24):
        sys.exit("--hours must be in (0, 24]")
    return s


def pod_env(s: dict, token: str) -> dict:
    e = os.environ
    env = {
        "FV_R2_ENDPOINT": "{{ RUNPOD_SECRET_fv_r2_endpoint }}",
        "FV_R2_ACCESS_KEY_ID": "{{ RUNPOD_SECRET_fv_r2_access_key_id }}",
        "FV_R2_SECRET_ACCESS_KEY": "{{ RUNPOD_SECRET_fv_r2_secret_access_key }}",
        "FV_R2_BUCKET": "{{ RUNPOD_SECRET_fv_r2_bucket }}",
        "STUDIO_R2_ACCESS_KEY_ID": "{{ RUNPOD_SECRET_studio_r2_access_key_id }}",
        "STUDIO_R2_SECRET_ACCESS_KEY": "{{ RUNPOD_SECRET_studio_r2_secret_access_key }}",
        "STUDIO_R2_BUCKET": e.get("STUDIO_R2_BUCKET", "fastvideo-studio"),
        "STUDIO_IMAGE": s["image"],
        "STUDIO_SERVE_TOKEN": token,
        "STUDIO_POD_HOURS": str(s["hours"]),
        "RUNPOD_API_KEY": key(),
    }
    if e.get("STUDIO_R2_ENDPOINT"):
        env["STUDIO_R2_ENDPOINT"] = e["STUDIO_R2_ENDPOINT"]
    if e.get("CF_ACCOUNT_ID") and e.get("CF_STREAM_API_TOKEN"):
        env["FV_CF_ACCOUNT_ID"] = e["CF_ACCOUNT_ID"]
        env["FV_CF_STREAM_API_TOKEN"] = e["CF_STREAM_API_TOKEN"]
    for k in ("STUDIO_ESRGAN", "STUDIO_TAEH3", "STUDIO_LOAD_TIMEOUT_S"):
        if e.get(k):
            env[k] = e[k]
    return env


def create_body(s: dict, token: str, name: str) -> dict:
    return {
        "name": name,
        "image": s["image"],
        "cloud": "SECURE",
        "gpu": {"id": s["gpu"], "count": 1, "minCudaVersion": s["min_cuda"]},
        "disk": s["disk"],
        "ports": ["8000/http"],
        "cmd": ["serve"],
        "env": pod_env(s, token),
        "mounts": {"network": [{"volumeId": s["volume"], "path": "/runpod-volume"}]},
        "dataCenterIds": [s["dc"]],
    }


def redacted(body: dict) -> dict:
    b = json.loads(json.dumps(body))
    for k in ("STUDIO_SERVE_TOKEN", "RUNPOD_API_KEY", "FV_CF_STREAM_API_TOKEN"):
        if k in b["env"]:
            b["env"][k] = "<secret>"
    return b


# ---------------------------------------------------------------- the pod's serve

def pod_call(method: str, path: str, body: dict | None = None, timeout: float = 60, quiet: bool = True) -> dict | list | None:
    s = current()
    return http(method, f"{s['url']}{path}", body, timeout, bearer=s["token"], quiet=quiet)


# ---------------------------------------------------------------- commands

def cmd_plan(a: argparse.Namespace) -> None:
    s = settings(a)
    print(json.dumps(redacted(create_body(s, "<token>", "studio-pod-plan")), indent=1))


def cmd_up(a: argparse.Namespace) -> None:
    if state().get("id"):
        sys.exit(f"{STATE} already holds pod {state()['id']}: run `down` first")
    s = settings(a)
    token = secrets.token_hex(16)
    name = f"studio-pod-{datetime.now(timezone.utc).strftime('%m%d%H%M')}"
    body = create_body(s, token, name)
    print(json.dumps(redacted(body), indent=1), file=sys.stderr)
    try:
        p = rest("POST", "/pods", body)
    except RuntimeError as e:
        sys.exit(str(e))
    if not p or not p.get("id"):
        sys.exit(f"no pod id in {p}")
    rec = {"id": p["id"], "url": pod_url(p["id"]), "token": token, "image": s["image"], "gpu": s["gpu"], "dc": s["dc"],
           "hours": s["hours"], "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "cost": p.get("cost")}
    save(rec)
    print(f"pod {p['id']} {str(p.get('status', '')).lower()} on {s['gpu']} (${p.get('cost')}/h), serve at {rec['url']} (token in {STATE}); it terminates itself after {s['hours']} h", file=sys.stderr)
    print(p["id"])


def cmd_status(a: argparse.Namespace) -> None:
    s = current()
    try:
        p = rest("GET", f"/pods/{s['id']}", timeout=30)
    except RuntimeError as e:
        p = {"error": str(e)}
    out = {"id": s["id"], "url": s["url"], "status": p.get("status"), "cost": p.get("cost"), "dc": p.get("dataCenterId"),
           "gpu": (p.get("gpu") or {}).get("id"), "created_at": s.get("created_at"), "hours": s.get("hours"), "error": p.get("error")}
    if out["status"] == "RUNNING":
        try:
            out["health"] = pod_call("GET", "/health", timeout=10)
        except Exception as e:  # noqa: BLE001
            out["health_error"] = str(e)[:300]
    print(json.dumps(out, indent=1))


def cmd_wait(a: argparse.Namespace) -> None:
    s = current()
    t0 = time.time()
    while time.time() - t0 < a.timeout:
        try:
            h = pod_call("GET", "/health", timeout=10)
            print(f"serve up after {time.time() - t0:.0f} s: {json.dumps(h)}", file=sys.stderr)
            return
        except Exception as e:  # noqa: BLE001
            print(f"  {time.time() - t0:.0f} s: {str(e)[:120]}", file=sys.stderr)
        time.sleep(10)
    sys.exit(f"pod {s['id']}: serve not up after {a.timeout} s")


def cmd_get(path: str):
    def run(a: argparse.Namespace) -> None:
        print(json.dumps(pod_call("GET", path), indent=1))
    return run


def submit(inp: dict) -> str:
    r = pod_call("POST", "/jobs", {"input": inp}, quiet=False)
    if not r or not r.get("id"):
        sys.exit(f"no job id in {r}")
    return r["id"]


def poll(jid: str) -> dict:
    last = ""
    while True:
        r = pod_call("GET", f"/jobs/{jid}", timeout=30)
        st = r.get("status")
        if st in ("IN_QUEUE", "IN_PROGRESS"):
            o = r.get("output") or {}
            line = f"{st}  {o.get('stage', '')} {o.get('detail', '')} step {o.get('step', '-')}/{o.get('total_steps', '-')} frames {o.get('frames', '-')}/{o.get('total_frames', '-')} {o.get('elapsed_s', 0)} s"
            if line != last:
                print(line, file=sys.stderr)
                last = line
            time.sleep(2)
            continue
        return r


def cmd_run(a: argparse.Namespace) -> None:
    inp = json.loads(a.input)
    jid = submit(inp)
    print(jid)
    t0 = time.time()
    r = poll(jid)
    print(f"{r.get('status')} in {time.time() - t0:.1f} s (delay {r.get('delayTime')} ms, execution {r.get('executionTime')} ms)", file=sys.stderr)
    if r.get("error"):
        print(f"error: {r['error']}", file=sys.stderr)
    print(json.dumps(r.get("output"), indent=1))


def cmd_cancel(a: argparse.Namespace) -> None:
    print(json.dumps(pod_call("POST", f"/jobs/{a.job_id}/cancel", {}), indent=1))


def cmd_down(a: argparse.Namespace) -> None:
    s = current()
    if not a.yes:
        sys.exit(f"would terminate pod {s['id']}: pass --yes")
    try:
        rest("DELETE", f"/pods/{s['id']}", timeout=60)
        print(f"pod {s['id']} terminated", file=sys.stderr)
    except RuntimeError as e:
        if "HTTP 404" not in str(e):
            sys.exit(str(e))
        print(f"pod {s['id']} already gone", file=sys.stderr)
    STATE.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, h in (("plan", "print the create body, send nothing"), ("up", "create the pod")):
        p = sub.add_parser(name, help=h)
        p.add_argument("--image", default=None)
        p.add_argument("--gpu", default=None)
        p.add_argument("--dc", default=None)
        p.add_argument("--volume", default=None)
        p.add_argument("--disk", type=int, default=None)
        p.add_argument("--hours", type=float, default=None)
        p.set_defaults(fn=cmd_plan if name == "plan" else cmd_up)
    p = sub.add_parser("status", help="RunPod's view of the pod and its /health"); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("wait", help="until the pod's serve answers /health"); p.add_argument("--timeout", type=float, default=900); p.set_defaults(fn=cmd_wait)
    for name, path in (("health", "/health"), ("info", "/info"), ("jobs", "/jobs")):
        p = sub.add_parser(name, help=f"GET {path} on the pod"); p.set_defaults(fn=cmd_get(path))
    p = sub.add_parser("run", help="submit one job to the pod and poll it"); p.add_argument("input"); p.set_defaults(fn=cmd_run)
    p = sub.add_parser("cancel", help="cancel a job"); p.add_argument("job_id"); p.set_defaults(fn=cmd_cancel)
    p = sub.add_parser("down", help="terminate the pod"); p.add_argument("--yes", action="store_true"); p.set_defaults(fn=cmd_down)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
