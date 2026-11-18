#!/usr/bin/env python3
"""The fastvideo-studio RunPod queue endpoint, through the REST API v1
(rest.runpod.io/v1) and the job API (api.runpod.ai/v2). Bodies follow what
fastvideo-rs tests live (crates/fastvideo-deploy/src/provider/runpod/serverless.rs,
scripts/serve/runpod-endpoint.sh) with the same money guards: workers 0 / 1,
idle timeout 5 s, FlashBoot off, a wall-clock backstop that deletes the
endpoint. Every request body is printed before it is sent.

    export RUNPOD_API_KEY=…  STUDIO_IMAGE=ghcr.io/zaitrarrio/fastvideo-studio-worker:sha-…
    python3 scripts/runpod/endpoint.py template          # create the template, print its id
    python3 scripts/runpod/endpoint.py up                # template + endpoint; writes .studio-endpoint.json
    python3 scripts/runpod/endpoint.py status | health
    python3 scripts/runpod/endpoint.py run '{"engine":"h3","model":"fasth3-v2","prompt":"a fox"}'
    python3 scripts/runpod/endpoint.py cancel <job id>
    python3 scripts/runpod/endpoint.py smoke             # op info + one 480p draft per engine, timed
    python3 scripts/runpod/endpoint.py backstop 90       # delete the endpoint after 90 minutes (detached)
    python3 scripts/runpod/endpoint.py down

Env: RUNPOD_API_KEY (required); STUDIO_IMAGE; RUNPOD_VOLUME_ID (default
bz0ofkfx45, fv-weights-h3-ltx-500, EU-RO-1: its data centre pins the
endpoint); RUNPOD_GPU_TYPES (comma list, default the RTX PRO 6000 Server
Edition); STUDIO_ENDPOINT_PREFIX (names, default fastvideo-studio);
STUDIO_CONTAINER_DISK_GB (default 350: the staged trees live there);
STUDIO_IDLE_TIMEOUT_S (5); STUDIO_EXEC_TIMEOUT_S (1800); STUDIO_R2_BUCKET
(the Studio bucket, default the fv_r2_bucket secret); STUDIO_STATE
(the state file, default ./.studio-endpoint.json).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REST = os.environ.get("RUNPOD_API_BASE", "https://rest.runpod.io/v1")
QUEUE = os.environ.get("RUNPOD_QUEUE_API", "https://api.runpod.ai/v2")
STATE = Path(os.environ.get("STUDIO_STATE", ".studio-endpoint.json"))
DEFAULT_GPUS = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
DEFAULT_VOLUME = "bz0ofkfx45"
DEFAULT_DC = "EU-RO-1"


def key() -> str:
    k = os.environ.get("RUNPOD_API_KEY")
    if not k:
        sys.exit("RUNPOD_API_KEY is not set")
    return k


def call(method: str, url: str, body: dict | None = None, timeout: float = 120) -> dict | list | None:
    data = json.dumps(body).encode() if body is not None else None
    if body is not None:
        print(f"→ {method} {url}\n{json.dumps(body, indent=1)}", file=sys.stderr)
    else:
        print(f"→ {method} {url}", file=sys.stderr)
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {key()}", "Content-Type": "application/json", "User-Agent": "fastvideo-studio/1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        raw = e.read()
        sys.exit(f"{method} {url}: HTTP {e.code}: {raw.decode(errors='replace')[:800]}")
    return json.loads(raw) if raw.strip() else None


def state() -> dict:
    return json.loads(STATE.read_text()) if STATE.is_file() else {}


def save(d: dict) -> None:
    STATE.write_text(json.dumps(d, indent=1) + "\n")
    print(f"state → {STATE}", file=sys.stderr)


def name() -> str:
    return os.environ.get("STUDIO_ENDPOINT_PREFIX", "fastvideo-studio")


def template_body() -> dict:
    image = os.environ.get("STUDIO_IMAGE")
    if not image:
        sys.exit("STUDIO_IMAGE is not set (ghcr.io/zaitrarrio/fastvideo-studio-worker:sha-…)")
    env = {
        # The Studio bucket and the weights store share the account's R2 key (fastvideo-rs secrets).
        "FV_R2_ENDPOINT": "{{ RUNPOD_SECRET_fv_r2_endpoint }}",
        "FV_R2_ACCESS_KEY_ID": "{{ RUNPOD_SECRET_fv_r2_access_key_id }}",
        "FV_R2_SECRET_ACCESS_KEY": "{{ RUNPOD_SECRET_fv_r2_secret_access_key }}",
        "FV_R2_BUCKET": "{{ RUNPOD_SECRET_fv_r2_bucket }}",
        "STUDIO_IMAGE": image,
    }
    if os.environ.get("STUDIO_R2_BUCKET"):
        env["STUDIO_R2_BUCKET"] = os.environ["STUDIO_R2_BUCKET"]
    for k in ("STUDIO_ESRGAN", "STUDIO_TAEH3", "STUDIO_LOAD_TIMEOUT_S"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return {
        "name": f"{name()}-{int(time.time())}",
        "imageName": image,
        "isServerless": True,
        "containerDiskInGb": int(os.environ.get("STUDIO_CONTAINER_DISK_GB", "350")),
        "volumeInGb": 0,
        "dockerEntrypoint": ["/opt/studio/venv/bin/python", "/opt/studio/handler.py"],
        "dockerStartCmd": [],
        "env": env,
    }


def endpoint_body(template_id: str) -> dict:
    gpus = [g.strip() for g in os.environ.get("RUNPOD_GPU_TYPES", DEFAULT_GPUS).split(",") if g.strip()]
    return {
        "name": f"{name()}-{int(time.time())}",
        "templateId": template_id,
        "computeType": "GPU",
        "gpuTypeIds": gpus,
        "gpuCount": 1,
        "networkVolumeId": os.environ.get("RUNPOD_VOLUME_ID", DEFAULT_VOLUME),
        "dataCenterIds": [os.environ.get("RUNPOD_DC", DEFAULT_DC)],
        "workersMin": 0,
        "workersMax": 1,
        "idleTimeout": int(os.environ.get("STUDIO_IDLE_TIMEOUT_S", "5")),
        "flashboot": False,
        "executionTimeoutMs": int(os.environ.get("STUDIO_EXEC_TIMEOUT_S", "1800")) * 1000,
        "scalerType": "QUEUE_DELAY",
        "scalerValue": 4,
        "minCudaVersion": "13.0",
    }


def cmd_template() -> str:
    t = call("POST", f"{REST}/templates", template_body())
    assert isinstance(t, dict)
    tid = t["id"]
    print(tid)
    save({**state(), "template": tid})
    return tid


def cmd_up() -> None:
    tid = cmd_template()
    e = call("POST", f"{REST}/endpoints", endpoint_body(tid))
    assert isinstance(e, dict)
    eid = e["id"]
    save({**state(), "template": tid, "endpoint": eid, "created": time.time()})
    print(f"{eid} {tid}")
    print(f"job API: {QUEUE}/{eid}/run    (delete with: endpoint.py down)", file=sys.stderr)


def endpoint_id() -> str:
    eid = os.environ.get("STUDIO_ENDPOINT_ID") or state().get("endpoint")
    if not eid:
        sys.exit("no endpoint: run `up` or set STUDIO_ENDPOINT_ID")
    return eid


def cmd_down() -> None:
    st = state()
    eid = os.environ.get("STUDIO_ENDPOINT_ID") or st.get("endpoint")
    tid = st.get("template")
    if eid:
        call("PATCH", f"{REST}/endpoints/{eid}", {"workersMin": 0, "workersMax": 0})
        for i in range(1, 7):
            try:
                call("DELETE", f"{REST}/endpoints/{eid}")
                break
            except SystemExit as e:
                print(f"delete attempt {i}: {e}", file=sys.stderr)
                time.sleep(i * 5)
    if tid:
        call("DELETE", f"{REST}/templates/{tid}")
    if STATE.is_file():
        STATE.unlink()
    print("down")


def cmd_status() -> None:
    print(json.dumps(call("GET", f"{REST}/endpoints/{endpoint_id()}"), indent=1))


def cmd_health() -> None:
    print(json.dumps(call("GET", f"{QUEUE}/{endpoint_id()}/health"), indent=1))


def submit(inp: dict) -> str:
    r = call("POST", f"{QUEUE}/{endpoint_id()}/run", {"input": inp})
    assert isinstance(r, dict)
    return r["id"]


def poll(job_id: str, quiet: bool = False) -> dict:
    last = None
    while True:
        r = call("GET", f"{QUEUE}/{endpoint_id()}/status/{job_id}") if not quiet else _get_quiet(f"{QUEUE}/{endpoint_id()}/status/{job_id}")
        assert isinstance(r, dict)
        st = r.get("status")
        line = f"{st}"
        if r.get("output") and st == "IN_PROGRESS":
            p = r["output"]
            line += f"  {p.get('stage')} {p.get('detail') or ''} step {p.get('step', '-')}/{p.get('total_steps', '-')} frames {p.get('frames', '-')}/{p.get('total_frames', '-')} {p.get('elapsed_s', 0)} s"
        if line != last:
            print(line, file=sys.stderr)
            last = line
        if st in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            return r
        time.sleep(2)


def _get_quiet(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key()}", "User-Agent": "fastvideo-studio/1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def cmd_run(arg: str) -> None:
    inp = json.loads(arg)
    jid = submit(inp)
    print(jid)
    t0 = time.time()
    r = poll(jid, quiet=True)
    print(f"{r.get('status')} in {time.time() - t0:.1f} s (delay {r.get('delayTime')} ms, execution {r.get('executionTime')} ms)", file=sys.stderr)
    print(json.dumps(r.get("output"), indent=1))


def cmd_cancel(job_id: str) -> None:
    print(json.dumps(call("POST", f"{QUEUE}/{endpoint_id()}/cancel/{job_id}", {}), indent=1))


SMOKE_PROMPT = "A red fox trots through fresh snow at dawn in a quiet pine forest; snow crunches softly under its paws."
SMOKE_JOBS = [
    ("h3 draft 480p", {"engine": "h3", "model": "h3-plug4", "task": "t2v", "prompt": SMOKE_PROMPT, "width": 832, "height": 480, "frames": 124, "seed": 7}),
    ("fastwan 480p", {"engine": "fastwan", "model": "fastwan-1.3b", "task": "t2v", "prompt": SMOKE_PROMPT, "width": 832, "height": 480, "frames": 81, "seed": 7}),
    ("fastvideo h3 draft 480p", {"engine": "fastvideo", "model": "fasth3-4step-vsa-480p-taeh3", "task": "t2v", "prompt": SMOKE_PROMPT, "width": 832, "height": 480, "frames": 124, "seed": 7}),
]


def cmd_smoke() -> None:
    results = []
    t0 = time.time()
    r = poll(submit({"op": "info"}), quiet=True)
    print(json.dumps(r.get("output"), indent=1))
    results.append(("info", r.get("status"), time.time() - t0, r.get("executionTime")))
    for label, inp in SMOKE_JOBS:
        t1 = time.time()
        r = poll(submit(inp), quiet=True)
        out = r.get("output") or {}
        ok = r.get("status") == "COMPLETED" and "error" not in out
        print(json.dumps({k: v for k, v in out.items() if k not in ("command",)}, indent=1))
        if out.get("command"):
            print(f"command: {out['command']}", file=sys.stderr)
        results.append((label, "ok" if ok else f"{r.get('status')}: {out.get('error', '')[:120]}", time.time() - t1, r.get("executionTime")))
    print("\nsmoke summary")
    for label, st, wall, exe in results:
        print(f"  {label:28s} {st:40s} wall {wall:7.1f} s  execution {exe or '-'} ms")


def cmd_backstop(minutes: str) -> None:
    """A detached timer that deletes the endpoint (and template) after N
    minutes: a session that dies takes a timer in it along, so this one is a
    new session (`setsid`), and the pod-side backstop in the worker is the
    second net (handler: STUDIO_LOAD_TIMEOUT_S, the endpoint's executionTimeoutMs)."""
    n = float(minutes)
    script = f"sleep {int(n * 60)}; cd {os.getcwd()!s}; {sys.executable} {os.path.abspath(__file__)} down >> backstop.log 2>&1"
    p = subprocess.Popen(["setsid", "nohup", "bash", "-c", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    print(f"backstop pid {p.pid}: `down` in {n:g} minutes")


def main(argv: list[str]) -> None:
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, args = argv[1], argv[2:]
    if cmd == "template":
        cmd_template()
    elif cmd == "up":
        cmd_up()
    elif cmd == "down":
        cmd_down()
    elif cmd == "status":
        cmd_status()
    elif cmd == "health":
        cmd_health()
    elif cmd == "run":
        cmd_run(args[0])
    elif cmd == "cancel":
        cmd_cancel(args[0])
    elif cmd == "smoke":
        cmd_smoke()
    elif cmd == "backstop":
        cmd_backstop(args[0] if args else "120")
    elif cmd == "plan":
        print(json.dumps({"template": template_body(), "endpoint": endpoint_body("<template>")}, indent=1))
    else:
        sys.exit(f"unknown command {cmd}; see --help")


if __name__ == "__main__":
    main(sys.argv)
