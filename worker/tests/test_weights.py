"""The weights resolver: volume links, the R2 store's manifest layout (through
the local store), a file inside a published tree, and the `stage` copy rule."""

import json
import os

import pytest

from fvstudio.errors import JobError
from fvstudio.r2 import LocalStore
from fvstudio.weights import COMPLETE_MARKER, Roots, TreeEntry, WeightsResolver


def roots(tmp_path) -> Roots:
    return Roots(volume=tmp_path / "vol", volume_weights=tmp_path / "vol" / "weights", volume_extras=tmp_path / "vol" / "h3-rs",
                 stage_dir=tmp_path / "stage", link_root=tmp_path / "w")


def publish(store: LocalStore, dest: str, rev: str, files: dict[str, bytes], created: str = "2026-10-11T00:00:00Z"):
    """fastvideo-rs's layout: weights/<dest>/<rev>/<path> + manifests/<dest>/<rev>.json."""
    tmp = store.root / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    recs = []
    for path, data in files.items():
        p = tmp / path.replace("/", "_")
        p.write_bytes(data)
        store.put_file("fv-weights", f"weights/{dest}/{rev}/{path}", p)
        recs.append({"path": path, "size": len(data), "sha256": "x"})
    m = tmp / "m.json"
    m.write_text(json.dumps({"dest": dest, "rev": rev, "created_at": created, "files": recs}))
    store.put_file("fv-weights", f"manifests/{dest}/{rev}.json", m)


def test_tree_entry_parsing():
    assert TreeEntry.parse("weights/h3-base") == TreeEntry("volume", "weights/h3-base", "h3-base")
    assert TreeEntry.parse("h3-rs/tables") == TreeEntry("volume", "h3-rs/tables", "h3-rs/tables")
    assert TreeEntry.parse("r2:fv-weights/weights/fastwan21-1.3b=fastwan") == TreeEntry("r2", "fv-weights/weights/fastwan21-1.3b", "fastwan")
    assert TreeEntry.parse("https:weights.fastwan.dev/v1/x.safetensors=fastwan-public/x.safetensors") == TreeEntry("https", "weights.fastwan.dev/v1/x.safetensors", "fastwan-public/x.safetensors")
    with pytest.raises(ValueError):
        TreeEntry.parse("nope/x")


def test_volume_trees_are_linked_not_copied(tmp_path):
    r = roots(tmp_path)
    (r.volume_weights / "h3-base" / "transformer").mkdir(parents=True)
    (r.volume_extras / "tables").mkdir(parents=True)
    (r.volume_weights / "auxiliary" / "tae").mkdir(parents=True)
    (r.volume_weights / "auxiliary" / "tae" / "taeh3.safetensors").write_bytes(b"t")
    w = WeightsResolver(r)
    assert w.ensure_tree("weights/h3-base") is False
    assert w.ensure_tree("h3-rs/tables") is False
    assert w.ensure_tree("weights/auxiliary/tae/taeh3.safetensors") is False
    assert os.path.realpath(r.link_root / "h3-base") == os.path.realpath(r.volume_weights / "h3-base")
    assert (r.link_root / "h3-rs" / "tables").is_symlink()
    assert (r.link_root / "auxiliary" / "tae" / "taeh3.safetensors").read_bytes() == b"t"
    assert not (r.link_root / "auxiliary").is_symlink()  # a real parent dir, no write through a volume link
    st = w.status("weights/h3-base")
    assert st.volume and st.linked and not st.staged
    with pytest.raises(JobError, match="not on the volume"):
        w.ensure_tree("h3-rs/nope")


def test_missing_volume_tree_comes_from_the_r2_store_by_manifest(tmp_path):
    r = roots(tmp_path)
    r.volume_weights.mkdir(parents=True)
    store = LocalStore(tmp_path / "r2")
    publish(store, "fastwan22-ti2v-5b", "3e18704", {"transformer/a.safetensors": b"A" * 10, "config.json": b"{}"})
    publish(store, "fastwan22-ti2v-5b", "0000000", {"old.bin": b"o"}, created="2026-01-01T00:00:00Z")  # an older revision loses
    w = WeightsResolver(r, store, log=lambda s: None)
    assert w.ensure_tree("weights/fastwan22-ti2v-5b") is True
    dst = r.stage_dir / "fastwan22-ti2v-5b"
    assert (dst / "transformer" / "a.safetensors").read_bytes() == b"A" * 10 and (dst / "config.json").is_file()
    assert (dst / ".fv-revision").read_text().strip() == "3e18704" and (dst / COMPLETE_MARKER).is_file()
    assert os.path.realpath(r.link_root / "fastwan22-ti2v-5b") == os.path.realpath(dst)
    assert w.ensure_tree("weights/fastwan22-ti2v-5b") is False  # idempotent
    st = w.status("weights/fastwan22-ti2v-5b")
    assert st.staged and st.linked and not st.volume


def test_file_inside_a_published_tree_fetches_only_that_file(tmp_path):
    r = roots(tmp_path)
    r.volume_weights.mkdir(parents=True)
    store = LocalStore(tmp_path / "r2")
    publish(store, "auxiliary/tae", "eu-20261011", {"taew2_1.safetensors": b"1" * 5, "taew2_2.safetensors": b"2" * 5, "taeh3.safetensors": b"3"})
    w = WeightsResolver(r, store, log=lambda s: None)
    assert w.ensure_tree("weights/auxiliary/tae/taew2_1.safetensors") is True
    tree = r.stage_dir / "auxiliary" / "tae"
    assert (tree / "taew2_1.safetensors").read_bytes() == b"1" * 5 and not (tree / "taeh3.safetensors").exists()
    assert not (tree / COMPLETE_MARKER).exists() and "taew2_1.safetensors" in (tree / ".studio-files").read_text()
    assert (r.link_root / "auxiliary" / "tae" / "taew2_1.safetensors").read_bytes() == b"1" * 5
    with pytest.raises(JobError, match="no manifest"):
        w.ensure_tree("weights/nothing-here")


def test_r2_prefix_entry_resolves_the_revision(tmp_path):
    r = roots(tmp_path)
    store = LocalStore(tmp_path / "r2")
    publish(store, "fastwan21-1.3b", "25e7ed7", {"transformer/x.safetensors": b"x" * 3, "tokenizer/tokenizer.json": b"{}"})
    w = WeightsResolver(r, store, log=lambda s: None)
    assert w.ensure_tree("r2:fv-weights/weights/fastwan21-1.3b=fastwan") is True
    assert (r.link_root / "fastwan" / "transformer" / "x.safetensors").read_bytes() == b"xxx"
    assert (r.stage_dir / "fastwan" / ".fv-revision").read_text().strip() == "25e7ed7"


def test_size_mismatch_is_refused(tmp_path):
    r = roots(tmp_path)
    store = LocalStore(tmp_path / "r2")
    publish(store, "t", "r1", {"a.bin": b"abc"})
    # corrupt the manifest's size
    m = json.loads(store.get_bytes("fv-weights", "manifests/t/r1.json"))
    m["files"][0]["size"] = 99
    (store.root / "fv-weights" / "manifests" / "t" / "r1.json").write_text(json.dumps(m))
    w = WeightsResolver(r, store, log=lambda s: None)
    with pytest.raises(JobError, match="manifest says 99"):
        w.ensure_tree("weights/t")


def test_stage_copies_subdirs_and_links_the_rest(tmp_path):
    r = roots(tmp_path)
    src = r.volume_weights / "h3-8step"
    for sub, files in {"transformer": ["a.safetensors", "b.safetensors"], "text_encoder": ["m.safetensors"], "vae": ["v.safetensors"]}.items():
        (src / sub).mkdir(parents=True)
        for f in files:
            (src / sub / f).write_bytes(os.urandom(64))
    (src / "fastvideo_inference.json").write_text("{}")
    w = WeightsResolver(r, log=lambda s: None)
    w.ensure_tree("weights/h3-8step")
    assert w.stage("h3-8step:text_encoder,vae") is True
    dst = r.stage_dir / "h3-8step"
    assert (dst / "text_encoder" / "m.safetensors").is_file() and not (dst / "text_encoder").is_symlink()
    assert (dst / "vae" / "v.safetensors").read_bytes() == (src / "vae" / "v.safetensors").read_bytes()
    assert (dst / "transformer").is_symlink() and os.path.realpath(dst / "transformer") == os.path.realpath(src / "transformer")
    assert (dst / "fastvideo_inference.json").is_symlink()
    assert os.path.realpath(r.link_root / "h3-8step") == os.path.realpath(dst)
    assert (dst / COMPLETE_MARKER).is_file()
    assert w.stage("h3-8step:text_encoder,vae") is False  # nothing to copy the second time
    # a whole tree
    assert w.stage("h3-8step") is True
    assert not (dst / "transformer").is_symlink() and (dst / "transformer" / "b.safetensors").is_file()
    # the volume was never written
    assert sorted(p.name for p in src.iterdir()) == ["fastvideo_inference.json", "text_encoder", "transformer", "vae"]


def test_stage_skips_a_missing_sub_and_refuses_a_missing_tree(tmp_path):
    r = roots(tmp_path)
    (r.volume_weights / "h3-base" / "vae").mkdir(parents=True)
    (r.volume_weights / "h3-base" / "vae" / "v").write_bytes(b"v")
    notes = []
    w = WeightsResolver(r, log=notes.append)
    w.ensure_tree("weights/h3-base")
    assert w.stage("h3-base:nope,vae") is True
    assert any("no nope" in n for n in notes)
    with pytest.raises(JobError, match="not on the volume"):
        w.stage("absent")


def test_ensure_model_reports_stage_time(tmp_path):
    r = roots(tmp_path)
    (r.volume_weights / "fastwan").mkdir(parents=True)
    w = WeightsResolver(r, log=lambda s: None)
    out = w.ensure_model(("weights/fastwan",), (), None)
    assert out["fetched"] == [] and out["staged"] == [] and out["stage_s"] >= 0
