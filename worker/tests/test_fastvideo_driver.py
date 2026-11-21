import re
"""The fastvideo driver against a fake fv-serve HTTP server in-process."""


import pytest

from fvstudio import flags
from fvstudio.engines import Ctx, fastvideo
from fvstudio.errors import JobError
from fvstudio.progress import Reporter
from fvstudio.r2 import LocalStore
from fvstudio.sidecar import SidecarManager

from fake_fv_serve import FakeFvServe


@pytest.fixture
def fake(tmp_path):
    f = FakeFvServe(tmp_path / "files").start()
    yield f
    f.stop()


def ctx_for(tmp_path, catalog, fake, store=None, bucket=None, sent=None):
    rep = Reporter(sent.append if sent is not None else None, "fastvideo", "x", min_interval_s=0)
    return Ctx(settings=flags.EngineSettings(), knobs=catalog.knobs, sidecars=SidecarManager(), ports={"fastvideo": fake.port},
               env={"fastvideo": {}}, job_dir=tmp_path / "job", store=store, bucket=bucket, log=lambda s: None, report=rep, fv_fake=True)


def test_t2v_job_runs_to_an_mp4(tmp_path, catalog, fake):
    sent = []
    ctx = ctx_for(tmp_path, catalog, fake, sent=sent)
    job = catalog.validate({"engine": "fastvideo", "model": "fastwan22-ti2v-5b", "prompt": "a fox", "width": 832, "height": 480,
                            "frames": 49, "seed": 11, "options": {"steps": 6}})
    res = fastvideo.run(ctx, job)
    assert res.video_mp4 is not None and res.video_mp4.stat().st_size > 100
    assert (res.width, res.height, res.frames, res.fps, res.seed) == (832, 480, 49, 24, 11)
    assert res.timings.denoise_s == 1.5 and res.timings.encode_s == 0.2 and res.timings.decode_s == 0.3
    assert fake.requests[-1] == {"model": "fastwan22-ti2v-5b", "prompt": "a fox", "seed": 11, "size": "832x480", "num_frames": 49, "steps": 6}
    stages = [r["stage"] for r in sent]
    assert "denoising" in stages and stages[-1] == "decoding"
    assert any(r.get("detail", "").endswith("%") for r in sent)
    assert "curl" in res.command and "fv/v1/jobs" in res.command


def test_media_become_urls(tmp_path, catalog, fake):
    store = LocalStore(tmp_path / "r2")
    up = tmp_path / "up.png"
    up.write_bytes(b"\x89PNG fake")
    store.put_file("studio", "uploads/u1.png", up)
    ctx = ctx_for(tmp_path, catalog, fake, store, "studio")
    job = catalog.validate({"engine": "fastvideo", "model": "fasth3-4step-vsa", "task": "fl2v", "prompt": "p", "width": 832, "height": 480,
                            "inputs": {"first_image": {"r2": "uploads/u1.png"}, "last_image": {"b64": "iVBORw0KGgo=", "mime": "image/png"}}})
    fastvideo.run(ctx, job)
    body = fake.requests[-1]
    assert body["image_url"].startswith("file://") and "uploads/u1.png" in body["image_url"]
    assert body["last_image_url"] == "data:image/png;base64,iVBORw0KGgo="


def test_failed_job_reports_engine(tmp_path, catalog):
    f = FakeFvServe(tmp_path / "files", fail_model="sol-h3").start()
    try:
        ctx = ctx_for(tmp_path, catalog, f)
        job = catalog.validate({"engine": "fastvideo", "model": "sol-h3", "prompt": "p"})
        with pytest.raises(JobError) as e:
            fastvideo.run(ctx, job)
        assert e.value.where == "engine" and "refuses" in str(e.value)
    finally:
        f.stop()


def test_refused_body_reports_engine(tmp_path, catalog, fake, monkeypatch):
    ctx = ctx_for(tmp_path, catalog, fake)
    job = catalog.validate({"engine": "fastvideo", "model": "sol-h3", "prompt": "p"})
    monkeypatch.setattr(flags, "fastvideo_body", lambda *a, **k: {"model": "sol-h3", "prompt": "p", "bogus": 1})
    with pytest.raises(JobError) as e:
        fastvideo.run(ctx, job)
    assert e.value.where == "engine" and "bogus" in str(e.value)


def test_group_config_keeps_the_process_group_with_the_requested_model_resident(catalog):
    from pathlib import Path
    base = Path(__file__).resolve().parents[1] / "configs" / "fastvideo.toml"
    members = fastvideo.group_members(catalog.models, "fastwan22-ti2v-5b")
    assert members == ["fastwan22-ti2v-5b", "wan22-ti2v-5b", "fastwan22-ti2v-5b-taehv"]
    out = fastvideo.group_config(base.read_text(), members, "fastwan22-ti2v-5b")
    blocks = out.split("[[models]]")[1:]
    assert len(blocks) == 3
    for b in blocks:
        v = re.search(r'variant = "([^"]+)"', b).group(1)
        assert v in members and (("resident = true" in b) == (v == "fastwan22-ti2v-5b"))
    assert "swap = true" in out and 'bind = "127.0.0.1:8000"' in out
    assert 'variant = "fasth3-4step-vsa"' not in out  # H3 and LTX / Wan cannot share a process
    assert 'variant = "fastwan21-1.3b"' not in out  # FastWan 2.1's VSA flag is process-wide
    h3 = fastvideo.group_members(catalog.models, "sol-h3")
    assert h3[0] == "sol-h3" and len(h3) == sum(1 for m in catalog.models if m.engine == "fastvideo" and m.family == "fv-h3")
    alone = fastvideo.group_config(base.read_text(), fastvideo.group_members([], "ltx25-ref2v"), "ltx25-ref2v")
    assert len(alone.split("[[models]]")) == 2
    import pytest
    from fvstudio.errors import JobError
    with pytest.raises(JobError):
        fastvideo.group_config(base.read_text(), ["no-such-model"], "no-such-model")