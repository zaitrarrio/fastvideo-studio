"""scripts/runpod/endpoint.py against RunPod's REST v2 shapes, without a
network: the create body for defaults and overrides, the pool check against
a fake `GET /v2/catalog/gpus`, the money guard on --workers-max, the
registry pick, the ledger round trip, and `up` / `plan` with `urllib`
replaced so nothing is sent."""

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "runpod" / "endpoint.py"


@pytest.fixture
def ep(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("studio_endpoint", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod   # dataclasses resolve `from __future__ import annotations` through sys.modules
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "STATE", tmp_path / ".runpod-endpoint.json")
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    for k in ("STUDIO_IMAGE", "STUDIO_CONTAINER_DISK_GB", "RUNPOD_VOLUME_ID", "RUNPOD_DC", "STUDIO_IDLE_TIMEOUT_S",
              "STUDIO_EXEC_TIMEOUT_S", "STUDIO_R2_BUCKET", "STUDIO_ESRGAN", "STUDIO_TAEH3", "STUDIO_LOAD_TIMEOUT_S",
              "STUDIO_ENDPOINT_ID", "STUDIO_ENDPOINT_NAME"):
        monkeypatch.delenv(k, raising=False)
    # Any urlopen that a test does not replace is a bug: nothing may be sent.
    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda *a, **k: pytest.fail("urlopen called"))
    return mod


IMAGE = "ghcr.io/zaitrarrio/fastvideo-studio-worker:sha-abc"
ENV = {"STUDIO_IMAGE": IMAGE}

CATALOG = {"gpus": [
    {"id": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "pool": "BLACKWELL_96", "memory": 96, "price": {"serverless": 3.69}},
    {"id": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "pool": "BLACKWELL_96", "memory": 96, "price": {"serverless": 3.69}},
    {"id": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition", "pool": "BLACKWELL_96", "memory": 96, "price": {"serverless": 3.69}},
    {"id": "NVIDIA GeForce RTX 5090", "pool": "ADA_32_PRO", "memory": 32, "price": {"serverless": 1.89}},
    {"id": "AMD Instinct MI300X OAM", "pool": None, "memory": 192, "price": {"secure": 2.39}},
]}

REGISTRIES = {"registries": [
    {"id": "cmrewljxt0013uo3qabbv220y", "name": "strobe-ghcr"},
    {"id": "cmufz4twn000hy3yclmxoab2n", "name": "fv-ghcr-1790281084"},
    {"id": "cmugdefkz0063zzggofzn5p3f", "name": "fv-ghcr-1790305046"},
    {"id": "cmuessbqa008fszbua8mpwx5j", "name": "fv-ghcr-1790209956"},
]}


def args(ep, *argv):
    return ep.parser().parse_args(list(argv))


# ---------------------------------------------------------------- the body

def test_default_body(ep):
    spec = ep.UpSpec.from_args(args(ep, "plan"), ENV)
    body = ep.endpoint_body(spec, ENV)
    assert body["name"] == "studio"
    assert body["type"] == "QUEUE"
    assert body["scaling"] == {"type": "QUEUE_DELAY", "queueDelay": 4}
    assert body["image"] == IMAGE
    assert body["disk"] == 300
    assert body["gpu"] == {"pools": ["BLACKWELL_96"], "count": 1,
                           "excludedTypes": ["NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"],
                           "allowedCudaVersions": [], "minCudaVersion": "13.0"}
    assert body["workers"] == {"min": 0, "max": 1, "idleTimeout": 5}
    assert body["timeout"] == 1_800_000
    assert body["flashboot"] == "OFF"
    assert body["mounts"] == {"network": [{"volumeId": "bz0ofkfx45"}]}
    assert body["dataCenterIds"] == ["EU-RO-1"]
    assert "registry" not in body
    env = body["env"]
    assert env["FV_R2_ENDPOINT"] == "{{ RUNPOD_SECRET_fv_r2_endpoint }}"
    assert env["FV_R2_ACCESS_KEY_ID"] == "{{ RUNPOD_SECRET_fv_r2_access_key_id }}"
    assert env["FV_R2_SECRET_ACCESS_KEY"] == "{{ RUNPOD_SECRET_fv_r2_secret_access_key }}"
    assert env["FV_R2_BUCKET"] == "{{ RUNPOD_SECRET_fv_r2_bucket }}"
    assert env["STUDIO_IMAGE"] == IMAGE
    assert "STUDIO_R2_BUCKET" not in env
    # No v1 field survives.
    for v1 in ("templateId", "gpuTypeIds", "networkVolumeId", "workersMin", "workersMax", "executionTimeoutMs", "scalerType"):
        assert v1 not in body
    json.dumps(body)  # serialisable


def test_override_body(ep):
    a = args(ep, "plan", "--name", "studio-2", "--image", "ghcr.io/x/y:z", "--pool", "ADA_32_PRO", "--exclude", "A", "--exclude", "B",
             "--gpu-count", "2", "--disk", "120", "--volume", "vol1", "--dc", "US-KS-2", "--idle", "9", "--timeout-s", "60",
             "--flashboot", "--registry", "reg1", "--min-cuda", "12.8", "--env", "STUDIO_FOO=1", "--env", "K=a=b")
    body = ep.endpoint_body(ep.UpSpec.from_args(a, {**ENV, "STUDIO_R2_BUCKET": "studio-out", "STUDIO_ESRGAN": "/w/esr.safetensors"}),
                            {"STUDIO_R2_BUCKET": "studio-out", "STUDIO_ESRGAN": "/w/esr.safetensors"})
    assert body["name"] == "studio-2"
    assert body["image"] == "ghcr.io/x/y:z"
    assert body["env"]["STUDIO_IMAGE"] == "ghcr.io/x/y:z"
    assert body["gpu"] == {"pools": ["ADA_32_PRO"], "count": 2, "excludedTypes": ["A", "B"], "allowedCudaVersions": [], "minCudaVersion": "12.8"}
    assert body["disk"] == 120
    assert body["mounts"] == {"network": [{"volumeId": "vol1"}]}
    assert body["dataCenterIds"] == ["US-KS-2"]
    assert body["workers"] == {"min": 0, "max": 1, "idleTimeout": 9}
    assert body["timeout"] == 60_000
    assert body["flashboot"] == "FLASHBOOT"
    assert body["registry"] == "reg1"
    assert body["env"]["STUDIO_R2_BUCKET"] == "studio-out"
    assert body["env"]["STUDIO_ESRGAN"] == "/w/esr.safetensors"
    assert body["env"]["STUDIO_FOO"] == "1"
    assert body["env"]["K"] == "a=b"


def test_env_defaults_feed_the_body(ep):
    env = {**ENV, "STUDIO_CONTAINER_DISK_GB": "350", "RUNPOD_VOLUME_ID": "v9", "RUNPOD_DC": "EU-CZ-1",
           "STUDIO_IDLE_TIMEOUT_S": "7", "STUDIO_EXEC_TIMEOUT_S": "900"}
    body = ep.endpoint_body(ep.UpSpec.from_args(args(ep, "plan"), env), env)
    assert body["disk"] == 350
    assert body["mounts"]["network"] == [{"volumeId": "v9"}]
    assert body["dataCenterIds"] == ["EU-CZ-1"]
    assert body["workers"]["idleTimeout"] == 7
    assert body["timeout"] == 900_000


def test_no_exclusions(ep):
    body = ep.endpoint_body(ep.UpSpec.from_args(args(ep, "plan", "--exclude", ""), ENV), ENV)
    assert body["gpu"]["excludedTypes"] == []


def test_missing_image_refused(ep):
    with pytest.raises(SystemExit, match="STUDIO_IMAGE"):
        ep.UpSpec.from_args(args(ep, "plan"), {})


# ---------------------------------------------------------------- guards

def test_workers_max_guard(ep):
    with pytest.raises(SystemExit, match="--i-know"):
        ep.UpSpec.from_args(args(ep, "plan", "--workers-max", "2"), ENV)
    spec = ep.UpSpec.from_args(args(ep, "plan", "--workers-max", "2", "--i-know"), ENV)
    assert ep.endpoint_body(spec, ENV)["workers"]["max"] == 2
    assert ep.endpoint_body(ep.UpSpec.from_args(args(ep, "plan", "--workers-max", "0"), ENV), ENV)["workers"]["max"] == 0


# ---------------------------------------------------------------- the pool check

def test_pool_gpus(ep):
    hit = ep.pool_gpus(CATALOG, "BLACKWELL_96")
    assert sorted(g["id"] for g in hit) == sorted(g["id"] for g in CATALOG["gpus"] if g["pool"] == "BLACKWELL_96")
    assert [g["id"] for g in ep.pool_gpus(CATALOG["gpus"], "ADA_32_PRO")] == ["NVIDIA GeForce RTX 5090"]


def test_unknown_pool_names_the_pools(ep):
    with pytest.raises(SystemExit) as e:
        ep.pool_gpus(CATALOG, "BLACKWELL_42")
    assert "BLACKWELL_42" in str(e.value) and "ADA_32_PRO" in str(e.value) and "BLACKWELL_96" in str(e.value)


def test_describe_pool(ep):
    text = ep.describe_pool(ep.pool_gpus(CATALOG, "BLACKWELL_96"), ep.DEFAULT_EXCLUDE + ["NVIDIA H100 PCIe"])
    assert "Server Edition" in text and "$3.69/h" in text
    assert text.count("(excluded)") == 1
    assert "not in this pool: NVIDIA H100 PCIe" in text


# ---------------------------------------------------------------- the registry pick

def test_pick_registry_newest(ep):
    regs = REGISTRIES["registries"]
    assert ep.pick_registry(regs, "fv-ghcr-*") == "cmugdefkz0063zzggofzn5p3f"
    assert ep.pick_registry(regs, "fv-ghcr") == "cmugdefkz0063zzggofzn5p3f"   # a bare prefix
    assert ep.pick_registry(regs, "strobe-ghcr") == "cmrewljxt0013uo3qabbv220y"
    with pytest.raises(SystemExit, match="no container-registry credential"):
        ep.pick_registry(regs, "nope-*")


# ---------------------------------------------------------------- the ledger

def test_ledger_round_trip(ep):
    assert ep.state() == {}
    ep.save({"id": "abc123", "name": "studio", "createdAt": "2026-11-18T00:00:00Z", "image": IMAGE})
    assert ep.STATE.is_file()
    assert ep.state() == {"id": "abc123", "name": "studio", "createdAt": "2026-11-18T00:00:00Z", "image": IMAGE}
    assert ep.endpoint_id() == "abc123"
    assert ep.endpoint_id("xyz") == "xyz"


def test_endpoint_id_without_ledger(ep, monkeypatch):
    with pytest.raises(SystemExit, match="no endpoint"):
        ep.endpoint_id()
    monkeypatch.setenv("STUDIO_ENDPOINT_ID", "fromenv")
    assert ep.endpoint_id() == "fromenv"


# ---------------------------------------------------------------- up / plan over a fake urllib

class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def fake_urlopen(routes: dict, sent: list):
    """`routes` maps `METHOD URL` to a body (dict) or a callable(body) → dict."""
    def urlopen(req, timeout=None):
        body = json.loads(req.data) if req.data else None
        assert req.get_header("Authorization") == "Bearer test-key"
        k = f"{req.get_method()} {req.full_url}"
        sent.append((k, body))
        if k not in routes:
            pytest.fail(f"unexpected request {k}")
        r = routes[k]
        out = r(body) if callable(r) else r
        return FakeResponse(json.dumps(out).encode())
    return urlopen


def test_up_sends_one_create_and_writes_the_ledger(ep, monkeypatch, capsys):
    monkeypatch.setenv("STUDIO_IMAGE", IMAGE)
    created = {"id": "ep123", "name": "studio", "image": IMAGE, "createdAt": "2026-11-18T10:00:00Z",
               "requestUrls": {"run": "https://api.runpod.ai/v2/ep123/run", "health": "https://api.runpod.ai/v2/ep123/health"}}
    sent = []
    routes = {
        f"GET {ep.REST}/catalog/gpus": CATALOG,
        f"GET {ep.REST}/serverless": {"endpoints": [{"id": "4x4j1n73cy733m", "name": "strobe"}]},
        f"GET {ep.REST}/registries": REGISTRIES,
        f"POST {ep.REST}/serverless": lambda body: {**created, **{k: body[k] for k in ("name", "image")}},
    }
    monkeypatch.setattr(ep.urllib.request, "urlopen", fake_urlopen(routes, sent))
    ep.main(["endpoint.py", "up", "--registry-name", "fv-ghcr-*"])
    posts = [(k, b) for k, b in sent if k.startswith("POST")]
    assert len(posts) == 1
    body = posts[0][1]
    assert body["registry"] == "cmugdefkz0063zzggofzn5p3f"
    assert body["gpu"]["pools"] == ["BLACKWELL_96"]
    assert body["workers"] == {"min": 0, "max": 1, "idleTimeout": 5}
    assert body["flashboot"] == "OFF"
    assert sent[0][0].startswith("GET")  # the catalog check comes first
    out = capsys.readouterr()
    assert out.out.strip().splitlines()[0] == "ep123"
    assert '"queueDelay": 4' in out.err          # the body was printed before sending
    assert "$3.69/h" in out.err                  # the pool's prices were printed
    led = ep.state()
    assert led["id"] == "ep123" and led["name"] == "studio" and led["image"] == IMAGE
    assert led["createdAt"] == "2026-11-18T10:00:00Z"
    assert led["requestUrls"]["run"].endswith("/ep123/run")


def test_up_refuses_a_taken_name(ep, monkeypatch):
    monkeypatch.setenv("STUDIO_IMAGE", IMAGE)
    sent = []
    routes = {
        f"GET {ep.REST}/catalog/gpus": CATALOG,
        f"GET {ep.REST}/serverless": {"endpoints": [{"id": "old1", "name": "studio"}]},
    }
    monkeypatch.setattr(ep.urllib.request, "urlopen", fake_urlopen(routes, sent))
    with pytest.raises(SystemExit, match="old1"):
        ep.main(["endpoint.py", "up"])
    assert not any(k.startswith("POST") for k, _ in sent)
    assert not ep.STATE.exists()


def test_up_refuses_an_unknown_pool_before_creating(ep, monkeypatch):
    monkeypatch.setenv("STUDIO_IMAGE", IMAGE)
    sent = []
    monkeypatch.setattr(ep.urllib.request, "urlopen", fake_urlopen({f"GET {ep.REST}/catalog/gpus": CATALOG}, sent))
    with pytest.raises(SystemExit, match="not in GET"):
        ep.main(["endpoint.py", "up", "--pool", "HOPPER_9000"])
    assert [k for k, _ in sent] == [f"GET {ep.REST}/catalog/gpus"]


def test_plan_prints_the_body_and_sends_nothing(ep, monkeypatch, capsys):
    monkeypatch.setenv("STUDIO_IMAGE", IMAGE)
    ep.main(["endpoint.py", "plan", "--registry-name", "fv-ghcr-*"])
    body = json.loads(capsys.readouterr().out)
    assert body["name"] == "studio" and body["image"] == IMAGE
    assert body["registry"].startswith("<newest registry matching")


def test_down_needs_yes_then_patches_and_deletes(ep, monkeypatch, capsys):
    ep.save({"id": "ep123", "name": "studio"})
    with pytest.raises(SystemExit, match="--yes"):
        ep.main(["endpoint.py", "down"])
    sent = []
    routes = {f"PATCH {ep.REST}/serverless/ep123": {"id": "ep123"}, f"DELETE {ep.REST}/serverless/ep123": {}}
    monkeypatch.setattr(ep.urllib.request, "urlopen", fake_urlopen(routes, sent))
    ep.main(["endpoint.py", "down", "--yes"])
    assert [k for k, _ in sent] == [f"PATCH {ep.REST}/serverless/ep123", f"DELETE {ep.REST}/serverless/ep123"]
    assert sent[0][1] == {"workers": {"min": 0, "max": 0}}
    assert not ep.STATE.exists()
    assert "down" in capsys.readouterr().out


def test_status_and_health_take_id_or_ledger(ep, monkeypatch, capsys):
    sent = []
    routes = {f"GET {ep.REST}/serverless/abc": {"id": "abc"}, f"GET {ep.QUEUE}/led/health": {"workers": {}}}
    monkeypatch.setattr(ep.urllib.request, "urlopen", fake_urlopen(routes, sent))
    ep.main(["endpoint.py", "status", "--id", "abc"])
    ep.save({"id": "led"})
    ep.main(["endpoint.py", "health"])
    assert [k for k, _ in sent] == [f"GET {ep.REST}/serverless/abc", f"GET {ep.QUEUE}/led/health"]
