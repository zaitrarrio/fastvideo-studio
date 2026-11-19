#!/usr/bin/env python3
"""The fastvideo-studio RunPod queue endpoint, through the REST API v2
(api.runpod.io/v2: `rest.runpod.io` no longer resolves) and the job API
(api.runpod.ai/v2). v2 endpoints take the image directly, so there is no
template step; a GPU is chosen by **pool** (`GET /v2/catalog/gpus` → `pool`,
e.g. BLACKWELL_96 for the RTX PRO 6000 family) minus excluded type ids, and
a private ghcr image needs a container-registry credential id (`registry`,
`GET /v2/registries`). The money guards are fastvideo-rs's: workers 0 / 1,
idle timeout 5 s, FlashBoot off, execution timeout 30 min, a wall-clock
backstop that deletes the endpoint. Every request body is printed before it
is sent.

    export RUNPOD_API_KEY=…  STUDIO_IMAGE=ghcr.io/zaitrarrio/fastvideo-studio-worker:sha-…
    python3 scripts/runpod/endpoint.py gpus                       # catalog: id, pool, memory, serverless $/h
    python3 scripts/runpod/endpoint.py plan [up flags]            # print the create body, send nothing
    python3 scripts/runpod/endpoint.py up --name studio --registry-name 'fv-ghcr-*'
    python3 scripts/runpod/endpoint.py status | health            # from --id or .runpod-endpoint.json
    python3 scripts/runpod/endpoint.py run '{"engine":"h3","model":"fasth3-v2","prompt":"a fox"}'
    python3 scripts/runpod/endpoint.py cancel <job id>
    python3 scripts/runpod/endpoint.py smoke                      # op info + one 480p draft per engine, timed
    python3 scripts/runpod/endpoint.py backstop 90                # `down --yes` after 90 minutes (detached)
    python3 scripts/runpod/endpoint.py down --yes

`up` flags (and their env defaults): --name (studio) --image (STUDIO_IMAGE)
--pool (BLACKWELL_96) --exclude TYPE_ID (repeatable; default the Max-Q
Workstation Edition) --gpu-count (1) --disk GB (STUDIO_CONTAINER_DISK_GB,
300: the staged trees live there) --volume (RUNPOD_VOLUME_ID, bz0ofkfx45 =
fv-weights-h3-ltx-500, mounted at /runpod-volume) --dc (RUNPOD_DC, EU-RO-1:
the volume's data centre) --workers-max (1; more needs --i-know) --idle s
(STUDIO_IDLE_TIMEOUT_S, 5) --timeout-s (STUDIO_EXEC_TIMEOUT_S, 1800)
--flashboot --registry ID | --registry-name GLOB (the newest credential
whose name matches, e.g. 'fv-ghcr-*') --min-cuda (13.0) --env K=V
(repeatable). The container env always carries the `{{ RUNPOD_SECRET_fv_r2_* }}`
references and STUDIO_IMAGE; STUDIO_R2_BUCKET, STUDIO_ESRGAN, STUDIO_TAEH3
and STUDIO_LOAD_TIMEOUT_S are passed through when set. The ledger is
./.runpod-endpoint.json (STUDIO_STATE); `--id` or STUDIO_ENDPOINT_ID
overrides it for status / health / down.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REST = os.environ.get("RUNPOD_API_BASE", "https://api.runpod.io/v2")
GRAPHQL = os.environ.get("RUNPOD_GRAPHQL", "https://api.runpod.io/graphql")
QUEUE = os.environ.get("RUNPOD_QUEUE_API", "https://api.runpod.ai/v2")
STATE = Path(os.environ.get("STUDIO_STATE", ".runpod-endpoint.json"))
DEFAULT_POOL = "BLACKWELL_96"
DEFAULT_EXCLUDE = ["NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"]
DEFAULT_VOLUME = "bz0ofkfx45"
DEFAULT_DC = "EU-RO-1"
DEFAULT_MIN_CUDA = "13.0"


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


# ---------------------------------------------------------------- ledger

def state() -> dict:
    return json.loads(STATE.read_text()) if STATE.is_file() else {}


def save(d: dict) -> None:
    STATE.write_text(json.dumps(d, indent=1) + "\n")
    print(f"ledger → {STATE}", file=sys.stderr)


def endpoint_id(explicit: str | None = None) -> str:
    eid = explicit or os.environ.get("STUDIO_ENDPOINT_ID") or state().get("id")
    if not eid:
        sys.exit("no endpoint: run `up`, pass --id or set STUDIO_ENDPOINT_ID")
    return eid


# ---------------------------------------------------------------- the create body

@dataclass
class UpSpec:
    """Everything `up` / `plan` need, resolved from flags and the environment."""
    name: str = "studio"
    image: str = ""
    pool: str = DEFAULT_POOL
    exclude: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE))
    gpu_count: int = 1
    disk: int = 300
    volume: str = DEFAULT_VOLUME
    dc: str = DEFAULT_DC
    workers_max: int = 1
    idle: int = 5
    timeout_s: int = 1800
    flashboot: bool = False
    registry: str | None = None
    registry_name: str | None = None
    min_cuda: str = DEFAULT_MIN_CUDA
    env: dict = field(default_factory=dict)
    i_know: bool = False

    @classmethod
    def from_args(cls, a: argparse.Namespace, environ: dict | None = None) -> "UpSpec":
        e = os.environ if environ is None else environ
        image = a.image or e.get("STUDIO_IMAGE", "")
        if not image:
            sys.exit("no image: pass --image or set STUDIO_IMAGE (ghcr.io/zaitrarrio/fastvideo-studio-worker:sha-…)")
        extra: dict = {}
        for kv in a.env or []:
            if "=" not in kv:
                sys.exit(f"--env takes K=V, got {kv!r}")
            k, v = kv.split("=", 1)
            extra[k] = v
        spec = cls(
            name=a.name,
            image=image,
            pool=a.pool,
            exclude=list(DEFAULT_EXCLUDE) if a.exclude is None else [x for x in a.exclude if x],
            gpu_count=a.gpu_count,
            disk=a.disk if a.disk is not None else int(e.get("STUDIO_CONTAINER_DISK_GB", "300")),
            volume=a.volume or e.get("RUNPOD_VOLUME_ID", DEFAULT_VOLUME),
            dc=a.dc or e.get("RUNPOD_DC", DEFAULT_DC),
            workers_max=a.workers_max,
            idle=a.idle if a.idle is not None else int(e.get("STUDIO_IDLE_TIMEOUT_S", "5")),
            timeout_s=a.timeout_s if a.timeout_s is not None else int(e.get("STUDIO_EXEC_TIMEOUT_S", "1800")),
            flashboot=bool(a.flashboot),
            registry=a.registry,
            registry_name=a.registry_name,
            min_cuda=a.min_cuda,
            env=extra,
            i_know=bool(a.i_know),
        )
        spec.guard()
        return spec

    def guard(self) -> None:
        """The money guards: one worker at most, unless the caller says they know."""
        if self.workers_max > 1 and not self.i_know:
            sys.exit(f"--workers-max {self.workers_max} > 1 costs {self.workers_max} GPUs at once; pass --i-know to mean it")
        if self.workers_max < 0 or self.gpu_count < 1:
            sys.exit("--workers-max must be ≥ 0 and --gpu-count ≥ 1")


def container_env(spec: UpSpec, environ: dict | None = None) -> dict:
    """The worker's env: the account's R2 secrets by reference, the image for
    `op: info`, the Studio knobs passed through, then the caller's --env."""
    e = os.environ if environ is None else environ
    env = {
        # The Studio bucket and the weights store share the account's R2 key (fastvideo-rs secrets).
        "FV_R2_ENDPOINT": "{{ RUNPOD_SECRET_fv_r2_endpoint }}",
        "FV_R2_ACCESS_KEY_ID": "{{ RUNPOD_SECRET_fv_r2_access_key_id }}",
        "FV_R2_SECRET_ACCESS_KEY": "{{ RUNPOD_SECRET_fv_r2_secret_access_key }}",
        "FV_R2_BUCKET": "{{ RUNPOD_SECRET_fv_r2_bucket }}",
        "STUDIO_IMAGE": spec.image,
    }
    for k in ("STUDIO_R2_BUCKET", "STUDIO_ESRGAN", "STUDIO_TAEH3", "STUDIO_LOAD_TIMEOUT_S"):
        if e.get(k):
            env[k] = e[k]
    env.update(spec.env)
    return env


def endpoint_body(spec: UpSpec, environ: dict | None = None) -> dict:
    """`POST /v2/serverless` for a queue endpoint on one GPU pool."""
    body = {
        "name": spec.name,
        "type": "QUEUE",
        "scaling": {"type": "QUEUE_DELAY", "queueDelay": 4},
        "image": spec.image,
        "disk": spec.disk,
        "env": container_env(spec, environ),
        "gpu": {
            "pools": [spec.pool],
            "count": spec.gpu_count,
            "excludedTypes": list(spec.exclude),
            "allowedCudaVersions": [],
            "minCudaVersion": spec.min_cuda,
        },
        "workers": {"min": 0, "max": spec.workers_max, "idleTimeout": spec.idle},
        "timeout": spec.timeout_s * 1000,
        "flashboot": "FLASHBOOT" if spec.flashboot else "OFF",
        "mounts": {"network": [{"volumeId": spec.volume}]},
        "dataCenterIds": [spec.dc],
    }
    if spec.registry:
        body["registry"] = spec.registry
    return body


# ---------------------------------------------------------------- catalog / registry / name checks

def pool_gpus(catalog: dict | list, pool: str) -> list[dict]:
    """The catalog entries of one pool. `catalog` is `GET /v2/catalog/gpus`'s
    body (`{"gpus": […]}`) or its list; an unknown pool is an error that names
    the pools that exist."""
    gpus = catalog.get("gpus", []) if isinstance(catalog, dict) else list(catalog)
    hit = [g for g in gpus if g.get("pool") == pool]
    if not hit:
        pools = sorted({g["pool"] for g in gpus if g.get("pool")})
        sys.exit(f"GPU pool {pool!r} is not in GET {REST}/catalog/gpus; pools: {', '.join(pools)}")
    return hit


def describe_pool(gpus: list[dict], exclude: list[str]) -> str:
    lines = []
    for g in sorted(gpus, key=lambda g: g.get("id", "")):
        price = (g.get("price") or {}).get("serverless")
        tag = "  (excluded)" if g.get("id") in exclude else ""
        lines.append(f"  {g.get('id', '?'):60s} {g.get('memory', '?'):>4} GB  serverless ${price if price is not None else '?'}/h{tag}")
    stray = [x for x in exclude if x not in {g.get("id") for g in gpus}]
    if stray:
        lines.append(f"  note: excluded type(s) not in this pool: {', '.join(stray)}")
    return "\n".join(lines)


def check_pool(spec: UpSpec) -> None:
    gpus = pool_gpus(call("GET", f"{REST}/catalog/gpus") or {}, spec.pool)
    print(f"pool {spec.pool}:\n{describe_pool(gpus, spec.exclude)}", file=sys.stderr)
    if not [g for g in gpus if g.get("id") not in spec.exclude]:
        sys.exit(f"every GPU of pool {spec.pool} is excluded")


def pick_registry(registries: list[dict], pattern: str) -> str:
    """The newest credential whose name matches `pattern` (fnmatch; a bare
    prefix matches as `prefix*`). Names end in a unix timestamp
    (`fv-ghcr-1790305046`), so newest is the largest trailing number, else
    the last one listed."""
    if not any(c in pattern for c in "*?["):
        pattern = pattern + "*"
    hits = [r for r in registries if fnmatch.fnmatchcase(r.get("name", ""), pattern)]
    if not hits:
        names = ", ".join(r.get("name", "?") for r in registries) or "none"
        sys.exit(f"no container-registry credential matches {pattern!r}; have: {names}")

    def stamp(r: dict) -> int:
        m = re.search(r"(\d+)$", r.get("name", ""))
        return int(m.group(1)) if m else -1

    best = max(enumerate(hits), key=lambda ih: (stamp(ih[1]), ih[0]))[1]
    return best["id"]


def list_registries() -> list[dict]:
    """`GET /v2/registries` (`{"registries": [{id, name}]}`); the GraphQL
    `myself.containerRegistryCreds` query is the same list, kept as the
    fallback."""
    try:
        r = call("GET", f"{REST}/registries")
        if isinstance(r, dict) and isinstance(r.get("registries"), list):
            return r["registries"]
    except SystemExit as e:
        print(f"registries via REST failed ({e}); trying GraphQL", file=sys.stderr)
    r = call("POST", GRAPHQL, {"query": "{ myself { containerRegistryCreds { id name } } }"})
    assert isinstance(r, dict)
    return r.get("data", {}).get("myself", {}).get("containerRegistryCreds", []) or []


def resolve_registry(spec: UpSpec) -> None:
    if spec.registry_name and not spec.registry:
        spec.registry = pick_registry(list_registries(), spec.registry_name)
        print(f"registry {spec.registry_name!r} → {spec.registry}", file=sys.stderr)


def list_endpoints() -> list[dict]:
    r = call("GET", f"{REST}/serverless")
    return r.get("endpoints", []) if isinstance(r, dict) else []


def check_name_free(name: str) -> None:
    for e in list_endpoints():
        if e.get("name") == name:
            sys.exit(f"an endpoint named {name!r} already exists: {e.get('id')} (down it, or pick another --name)")


# ---------------------------------------------------------------- commands

def cmd_up(a: argparse.Namespace) -> None:
    spec = UpSpec.from_args(a)
    check_pool(spec)
    check_name_free(spec.name)
    resolve_registry(spec)
    e = call("POST", f"{REST}/serverless", endpoint_body(spec))
    assert isinstance(e, dict) and e.get("id"), f"unexpected create response: {e}"
    eid = e["id"]
    save({"id": eid, "name": e.get("name", spec.name), "createdAt": e.get("createdAt") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "image": e.get("image", spec.image), "requestUrls": e.get("requestUrls", {})})
    print(eid)
    print(json.dumps({"id": eid, "name": e.get("name"), "requestUrls": e.get("requestUrls")}, indent=1), file=sys.stderr)
    print(f"job API: {QUEUE}/{eid}/run    (delete with: endpoint.py down --yes)", file=sys.stderr)


def cmd_plan(a: argparse.Namespace) -> None:
    spec = UpSpec.from_args(a)
    if spec.registry_name and not spec.registry:
        spec.registry = f"<newest registry matching {spec.registry_name}>"
    print(json.dumps(endpoint_body(spec), indent=1))


def cmd_gpus(a: argparse.Namespace) -> None:
    cat = call("GET", f"{REST}/catalog/gpus") or {}
    gpus = cat.get("gpus", []) if isinstance(cat, dict) else cat
    if a.pool:
        gpus = [g for g in gpus if g.get("pool") == a.pool]
    gpus = sorted(gpus, key=lambda g: (g.get("pool") or "~", g.get("id", "")))
    print(f"{'id':60s} {'pool':14s} {'mem':>5}  serverless $/h")
    for g in gpus:
        price = (g.get("price") or {}).get("serverless")
        print(f"{g.get('id', '?'):60s} {g.get('pool') or '-':14s} {g.get('memory', '?'):>5}  {price if price is not None else '-'}")


def cmd_down(a: argparse.Namespace) -> None:
    eid = endpoint_id(a.id)
    if not a.yes:
        sys.exit(f"would delete endpoint {eid}; pass --yes")
    try:
        call("PATCH", f"{REST}/serverless/{eid}", {"workers": {"min": 0, "max": 0}})
    except SystemExit as e:
        print(f"scale-to-zero before delete failed (going on): {e}", file=sys.stderr)
    for i in range(1, 7):
        try:
            call("DELETE", f"{REST}/serverless/{eid}")
            break
        except SystemExit as e:
            print(f"delete attempt {i}: {e}", file=sys.stderr)
            if i == 6:
                sys.exit(f"could not delete {eid}; delete it in the console")
            time.sleep(i * 5)
    if STATE.is_file() and state().get("id") == eid:
        STATE.unlink()
    print("down")


def cmd_status(a: argparse.Namespace) -> None:
    print(json.dumps(call("GET", f"{REST}/serverless/{endpoint_id(a.id)}"), indent=1))


def cmd_health(a: argparse.Namespace) -> None:
    print(json.dumps(call("GET", f"{QUEUE}/{endpoint_id(a.id)}/health"), indent=1))


def cmd_list(a: argparse.Namespace) -> None:
    for e in list_endpoints():
        w = e.get("workers") or {}
        print(f"{e.get('id')}  {e.get('name', ''):24s} {','.join((e.get('gpu') or {}).get('pools') or []):16s} "
              f"workers {w.get('min', '?')}/{w.get('max', '?')}  {e.get('image', '')}")


# ---------------------------------------------------------------- jobs (api.runpod.ai/v2, unchanged)

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


def cmd_run(a: argparse.Namespace) -> None:
    inp = json.loads(a.input)
    jid = submit(inp)
    print(jid)
    t0 = time.time()
    r = poll(jid, quiet=True)
    print(f"{r.get('status')} in {time.time() - t0:.1f} s (delay {r.get('delayTime')} ms, execution {r.get('executionTime')} ms)", file=sys.stderr)
    if r.get("error"):
        # RunPod lifts a handler's `error` key out of the output into the status.
        print(f"error: {r['error']}", file=sys.stderr)
    print(json.dumps(r.get("output"), indent=1))


def cmd_cancel(a: argparse.Namespace) -> None:
    print(json.dumps(call("POST", f"{QUEUE}/{endpoint_id()}/cancel/{a.job_id}", {}), indent=1))


SMOKE_PROMPT = "A red fox trots through fresh snow at dawn in a quiet pine forest; snow crunches softly under its paws."
SMOKE_JOBS = [
    ("h3 draft 480p", {"engine": "h3", "model": "h3-plug4", "task": "t2v", "prompt": SMOKE_PROMPT, "width": 832, "height": 480, "frames": 124, "seed": 7}),
    ("fastwan 480p", {"engine": "fastwan", "model": "fastwan-1.3b", "task": "t2v", "prompt": SMOKE_PROMPT, "width": 832, "height": 480, "frames": 81, "seed": 7}),
    ("fastvideo h3 draft 480p", {"engine": "fastvideo", "model": "fasth3-4step-vsa-480p-taeh3", "task": "t2v", "prompt": SMOKE_PROMPT, "width": 832, "height": 480, "frames": 124, "seed": 7}),
]


def cmd_smoke(a: argparse.Namespace) -> None:
    results = []
    t0 = time.time()
    r = poll(submit({"op": "info"}), quiet=True)
    print(json.dumps(r.get("output"), indent=1))
    results.append(("info", r.get("status"), time.time() - t0, r.get("executionTime")))
    for label, inp in SMOKE_JOBS:
        t1 = time.time()
        jid = submit(inp)
        print(f"{label}: job {jid}", file=sys.stderr)
        r = poll(jid, quiet=True)
        out = r.get("output") or {}
        err = r.get("error") or out.get("error") or ""
        ok = r.get("status") == "COMPLETED" and not err
        if err:
            print(f"error: {err}", file=sys.stderr)
        print(json.dumps({k: v for k, v in out.items() if k not in ("command",)}, indent=1))
        if out.get("command"):
            print(f"command: {out['command']}", file=sys.stderr)
        results.append((label, "ok" if ok else f"{r.get('status')}: {str(err)[:120]}", time.time() - t1, r.get("executionTime")))
    print("\nsmoke summary")
    for label, st, wall, exe in results:
        print(f"  {label:28s} {st:40s} wall {wall:7.1f} s  execution {exe or '-'} ms")


def cmd_backstop(a: argparse.Namespace) -> None:
    """A detached timer that deletes the endpoint after N minutes: a session
    that dies takes a timer in it along, so this one is a new session
    (`setsid`), and the pod-side backstop in the worker is the second net
    (handler: STUDIO_LOAD_TIMEOUT_S, the endpoint's `timeout`)."""
    n = float(a.minutes if a.minutes is not None else a.minutes_opt)
    eid = endpoint_id(a.id)
    script = (f"sleep {int(n * 60)}; cd {os.getcwd()!s}; {sys.executable} {os.path.abspath(__file__)} down --yes --id {eid}"
              f" >> backstop.log 2>&1")
    p = subprocess.Popen(["setsid", "nohup", "bash", "-c", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    print(f"backstop pid {p.pid}: `down --yes --id {eid}` in {n:g} minutes")


def cmd_template(a: argparse.Namespace) -> None:
    print("template: no-op on REST v2 (an endpoint takes the image directly; `up` creates it in one call)", file=sys.stderr)


# ---------------------------------------------------------------- CLI

def up_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--name", default=os.environ.get("STUDIO_ENDPOINT_NAME", "studio"), help="endpoint name (must be free)")
    p.add_argument("--image", default=None, help="worker image (default STUDIO_IMAGE)")
    p.add_argument("--pool", default=DEFAULT_POOL, help="GPU pool id from `gpus` (default %(default)s)")
    p.add_argument("--exclude", action="append", default=None, metavar="TYPE_ID",
                   help="GPU type id to exclude from the pool, repeatable (default: the Max-Q Workstation Edition; pass --exclude '' for none)")
    p.add_argument("--gpu-count", type=int, default=1)
    p.add_argument("--disk", type=int, default=None, help="container disk GB (default STUDIO_CONTAINER_DISK_GB or 300)")
    p.add_argument("--volume", default=None, help="network volume id (default RUNPOD_VOLUME_ID or bz0ofkfx45)")
    p.add_argument("--dc", default=None, help="data centre id (default RUNPOD_DC or EU-RO-1)")
    p.add_argument("--workers-max", type=int, default=1, help="max workers (default 1; > 1 needs --i-know)")
    p.add_argument("--idle", type=int, default=None, help="idle timeout s (default STUDIO_IDLE_TIMEOUT_S or 5)")
    p.add_argument("--timeout-s", type=int, default=None, help="per-request timeout s (default STUDIO_EXEC_TIMEOUT_S or 1800)")
    p.add_argument("--flashboot", action="store_true", help="FlashBoot on (default off)")
    p.add_argument("--registry", default=None, help="container-registry credential id for a private image")
    p.add_argument("--registry-name", default=None, metavar="GLOB", help="pick the newest credential whose name matches, e.g. 'fv-ghcr-*'")
    p.add_argument("--min-cuda", default=DEFAULT_MIN_CUDA, help="minCudaVersion floor (default %(default)s)")
    p.add_argument("--env", action="append", default=None, metavar="K=V", help="extra container env, repeatable")
    p.add_argument("--i-know", action="store_true", help="allow --workers-max > 1")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="endpoint.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("up", help="create the endpoint (checks the pool and the name first)")
    up_flags(p)
    p.set_defaults(fn=cmd_up)
    p = sub.add_parser("plan", help="print the create body, send nothing")
    up_flags(p)
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("gpus", help="list the GPU catalog: id, pool, memory, serverless $/h")
    p.add_argument("--pool", default=None)
    p.set_defaults(fn=cmd_gpus)
    p = sub.add_parser("list", help="list this account's endpoints")
    p.set_defaults(fn=cmd_list)
    for name, fn, h in (("status", cmd_status, "GET the endpoint"), ("health", cmd_health, "the job API's health")):
        p = sub.add_parser(name, help=h)
        p.add_argument("--id", default=None, help="endpoint id (default STUDIO_ENDPOINT_ID or the ledger)")
        p.set_defaults(fn=fn)
    p = sub.add_parser("down", help="delete the endpoint")
    p.add_argument("--id", default=None)
    p.add_argument("--yes", action="store_true", help="confirm the delete")
    p.set_defaults(fn=cmd_down)
    p = sub.add_parser("backstop", help="delete the endpoint after N minutes, detached")
    p.add_argument("minutes", nargs="?", default=None)
    p.add_argument("--minutes", dest="minutes_opt", default="120")
    p.add_argument("--id", default=None)
    p.set_defaults(fn=cmd_backstop)
    p = sub.add_parser("run", help="submit one job and poll it")
    p.add_argument("input", help="the job input as JSON")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("cancel", help="cancel a job")
    p.add_argument("job_id")
    p.set_defaults(fn=cmd_cancel)
    p = sub.add_parser("smoke", help="op info + one 480p draft per engine, timed")
    p.set_defaults(fn=cmd_smoke)
    p = sub.add_parser("template", help="no-op on REST v2 (kept for old command lines)")
    p.set_defaults(fn=cmd_template)
    return ap


def main(argv: list[str]) -> None:
    a = parser().parse_args(argv[1:])
    a.fn(a)


if __name__ == "__main__":
    main(sys.argv)
