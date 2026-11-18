"""The engines' frame stream and y4m handling."""

import shutil

import pytest

from fvstudio import frames as fr


def test_frame_stream_roundtrip():
    f0, f1 = bytes([1, 2, 3] * 4), bytes([9, 8, 7] * 4)
    body = fr.encode_frame_stream(2, 2, 5, [f0, f1])
    assert body[:16] == (2).to_bytes(4, "little") + (2).to_bytes(4, "little") + (5).to_bytes(4, "little") + (2).to_bytes(4, "little")
    c = fr.parse_frame_stream(body)
    assert (c.width, c.height, c.first, c.count) == (2, 2, 5, 2)
    assert c.rgb == f0 + f1


def test_frame_stream_rejects_truncation():
    body = fr.encode_frame_stream(2, 2, 0, [bytes(12)])
    with pytest.raises(ValueError):
        fr.parse_frame_stream(body[:-1])
    with pytest.raises(ValueError):
        fr.parse_frame_stream(b"\x00" * 8)


def test_rgb_clip_accumulates_in_order(tmp_path):
    clip = fr.RgbClip(tmp_path / "f.rgb")
    clip.append(fr.parse_frame_stream(fr.encode_frame_stream(2, 2, 0, [bytes(12), bytes(12)])))
    clip.append(fr.parse_frame_stream(fr.encode_frame_stream(2, 2, 2, [bytes(12)])))
    with pytest.raises(ValueError, match="gap"):
        clip.append(fr.parse_frame_stream(fr.encode_frame_stream(2, 2, 5, [bytes(12)])))
    clip.close()
    assert clip.frames == 3 and (tmp_path / "f.rgb").stat().st_size == 36


def test_read_y4m_info(tmp_path):
    p = tmp_path / "a.y4m"
    frame = b"FRAME\n" + bytes(4 * 2 * 3 // 2)
    p.write_bytes(b"YUV4MPEG2 W4 H2 F24:1 Ip A1:1 C420jpeg\n" + frame * 3)
    info = fr.read_y4m_info(p)
    assert (info.width, info.height, info.fps, info.frames, info.colorspace) == (4, 2, 24.0, 3, "420jpeg")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_rgb_to_y4m_through_ffmpeg(tmp_path):
    rgb = tmp_path / "f.rgb"
    w, h, n = 32, 16, 5
    rgb.write_bytes(bytes([200, 30, 30] * (w * h)) * n)
    out = tmp_path / "f.y4m"
    fr.rgb_to_y4m(rgb, w, h, 24, out)
    info = fr.read_y4m_info(out)
    assert (info.width, info.height, info.frames) == (w, h, n)
    assert info.fps == 24.0
