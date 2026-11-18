"""`h3 serve`'s per-request media fields (h3-rs serve.rs: task, first_image /
last_image {b64, mime}, refs [{kind, b64, mime?, audio_b64?}], continue)."""

import base64

from fvstudio.engines import h3


def test_serve_media_fields_shape(tmp_path):
    png = tmp_path / "a.png"; png.write_bytes(b"\x89PNG-first")
    jpg = tmp_path / "b.jpg"; jpg.write_bytes(b"\xff\xd8-last")
    y4m = tmp_path / "r.y4m"; y4m.write_bytes(b"YUV4MPEG2 ref")
    wav = tmp_path / "r.wav"; wav.write_bytes(b"RIFF ref")
    cont = tmp_path / "c.y4m"; cont.write_bytes(b"YUV4MPEG2 cont")
    local = {"first_image": png, "last_image": jpg,
             "references": [("image", png, None), ("video", y4m, wav), ("audio", wav, None)],
             "continue": (cont, None)}
    b = h3.serve_media_fields("ref2v", local)
    assert b["task"] == "ref2va"
    assert b["first_image"] == {"b64": base64.b64encode(png.read_bytes()).decode(), "mime": "image/png"}
    assert b["last_image"]["mime"] == "image/jpeg"
    kinds = [r["kind"] for r in b["refs"]]
    assert kinds == ["image", "video", "audio"]
    assert b["refs"][1]["mime"] == "video/x-yuv4mpeg" and "audio_b64" in b["refs"][1]
    assert b["refs"][2]["mime"] == "audio/wav" and "audio_b64" not in b["refs"][2]
    assert b["continue"] == {"b64": base64.b64encode(cont.read_bytes()).decode()}
    assert h3.serve_fits(local)
    assert h3.serve_media_fields("fl2v", {"first_image": png}) == {"task": "fl2va", "first_image": b["first_image"]}
    assert h3.serve_media_fields("t2v", {}) == {"task": "t2va"}


def test_serve_media_over_the_body_cap_goes_to_the_cli(tmp_path):
    big = tmp_path / "big.y4m"
    with big.open("wb") as f:
        f.truncate(50 * 1024 * 1024)  # 50 MiB raw → 66.7 MiB of base64, over serve's 64 MiB
    assert not h3.serve_fits({"references": [("video", big, None)]})
    small = tmp_path / "small.y4m"
    small.write_bytes(b"x" * 1024)
    assert h3.serve_fits({"references": [("video", small, None)]})


def test_serve_supports_reads_info_tasks():
    assert h3.serve_supports({"tasks": ["t2va", "fl2va"]}, "fl2v")
    assert not h3.serve_supports({"tasks": ["t2va", "fl2va"]}, "ref2v")
    assert h3.serve_supports({"ready": True}, "t2v")  # a serve without `tasks` (older h3-rs) still takes plain t2v
    assert not h3.serve_supports(None, "fl2v")
