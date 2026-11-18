"""The flag builder against command lines from the engines' run logs
(h3-rs docs/runs, the `+ timeout … /w/h3-bin …` lines): the same flag set,
whatever the order. Paths in the logs are the pod's (`/w`, `/workspace`);
the catalog's are `/workspace/w`, so paths are compared by their tree-relative
tail."""

import shlex
from pathlib import Path

import pytest

from fvstudio import flags
from fvstudio.errors import JobError

S = flags.EngineSettings(h3_bin="h3", fastwan_bin="fastwan", esrgan="/w/realesr-animevideov3.safetensors",
                         taeh3="/w/taeh3.safetensors")
IGNORE = {"--backend", "--seed", "--prompt", "--out", "--width", "--height", "--frames", "--task"}
# A flag at its clap default says nothing: the logs leave it out, an explicit knob spells it.
CLAP_DEFAULTS = {("--weights", "native"), ("--decoder-weights", "native"), ("--dense-first-steps", "0"),
                 ("--adapter-strength", "1"), ("--attention", "auto"), ("--steps", "3")}

# docs/runs/2026-11-14-runpod-6000s-probe-bug, 2026-11-15-runpod-6000s-longlive-plug,
# 2026-11-12-runpod-pro6000-v2-omni-trim, 2026-11-12-runpod-pro6000-variants.
LOG_LINES = {
    "v2-best": "/w/h3-bin generate --backend cuda --seed 1 --width 1344 --height 768 --frames 124 --prompt P --out /root/out/v2-best.y4m --variant fasth3-v2 --model-dir /w/h3-8step --transformer /w/h3-8step-mxfp8/transformer.safetensors --weights mxfp8 --decoder-weights mxfp8 --vsa-qk8 --pv8 --tables /root/v2.tables",
    "v2-budget": "/w/h3-bin generate --backend cuda --seed 1 --width 1344 --height 768 --frames 124 --prompt P --out /root/out/v2-budget.y4m --variant fasth3-v2 --model-dir /w/h3-8step --transformer /w/h3-8step-mxfp8/transformer.safetensors --weights mxfp8 --decoder-weights mxfp8 --vsa-qk8 --pv8 --tables /root/v2.tables --vsa-sparsity 0.9 --dense-first-steps 2",
    "v2-stream": "/w/h3-bin generate --backend cuda --seed 1 --width 1344 --height 768 --frames 124 --prompt P --out /root/out/v2-best-stream.y4m --variant fasth3-v2 --model-dir /w/h3-8step --transformer /w/h3-8step-mxfp8/transformer.safetensors --weights mxfp8 --decoder-weights mxfp8 --vsa-qk8 --pv8 --tables /root/v2.tables --stream-blocks",
    "plug-fast": "/w/h3-bin generate --backend cuda --seed 7 --width 1344 --height 768 --frames 124 --prompt P --out /root/out/plug-fast.y4m --variant h3-plug4 --model-dir /w/h3-base --variant-dir /workspace/h3-rs/plug4 --tables /workspace/h3-rs/plug4/plug4.tables --weights mxfp8 --decoder-weights mxfp8 --pv8",
    "plug-fl2va": "/w/h3-bin generate --backend cuda --seed 7 --width 1344 --height 768 --frames 124 --prompt P --out /root/out/plug-fl2va.y4m --variant h3-plug4 --model-dir /w/h3-base --variant-dir /workspace/h3-rs/plug4 --tables /workspace/h3-rs/plug4/plug4.tables --weights mxfp8 --decoder-weights mxfp8 --pv8 --task fl2va --first-image /workspace/h3-rs/assets/first.png --last-image /workspace/h3-rs/assets/last.png",
    "plug-ref2va": "/w/h3-bin generate --backend cuda --seed 7 --width 1344 --height 768 --frames 124 --prompt P --out /root/out/plug-ref2va.y4m --variant h3-plug4 --model-dir /w/h3-base --variant-dir /root/plug4 --transformer /w/h3-ref2va/transformer_ref --tables /root/plug4-ref.tables --weights mxfp8 --decoder-weights mxfp8 --pv8 --task ref2va --ref image:/workspace/h3-rs/assets/first.png",
    "base49-fast": "/w/h3-bin generate --backend cuda --seed 7 --width 1344 --height 768 --frames 124 --prompt P --out /root/out/base49-fast.y4m --variant h3 --model-dir /w/h3-base --weights mxfp8 --decoder-weights mxfp8 --pv8",
    "omni": "/w/h3-bin generate --backend cuda --variant fasth3-omni --model-dir /w/h3-base --variant-dir /w/fasth3-omni-8step-v1 --tables /workspace/h3-rs/tables/omni.tables --seed 7 --width 1344 --height 768 --frames 124 --prompt P --task ref2va --ref image:/workspace/h3-rs/assets/first.png --out /root/out/omni.y4m",
    "trim": "/w/h3-bin generate --backend cuda --variant fasth3-trim --model-dir /w/h3-base --transformer /w/fasth3-trim-8step-nvfp4/diffusion_models/fastvideo_fasth3_trim_8step_nvfp4.safetensors --width 1344 --height 768 --frames 124 --seed 7 --prompt P --out /root/out/trim.y4m",
    "pika-fl2va": "/w/h3-bin generate --backend cuda --variant pika-flex --model-dir /w/h3-base --variant-dir /w/pika-h3-flex --tables /workspace/h3-rs/tables/pika.tables --seed 7 --width 1344 --height 768 --frames 124 --prompt P --task fl2va --first-image /workspace/h3-rs/assets/first.png --last-image /workspace/h3-rs/assets/last.png --out /root/out/pika-fl2va.y4m",
    "regen2k": "/w/h3-bin regen2k --backend cuda --regen-dir /w/open-h3-regenerate-2k --model-dir /w/h3-base --draft /workspace/h3-rs/assets/fox.y4m --adaln-table /workspace/h3-rs/assets/adaln_t_table.bin --prompt P --out /root/out/regen.y4m",
}


def flag_map(argv: list[str]) -> dict:
    sub = argv[1]
    m = flags.parse_flags(argv[2:], flags.H3_VALUE_FLAGS, flags.H3_BOOL_FLAGS, flags.H3_ALIAS)
    out = {}
    for k, v in m.items():
        if k in IGNORE:
            continue
        if isinstance(v, str) and v.startswith("/"):
            v = tail(v)
        if isinstance(v, str) and ":" in v and k in ("--ref", "--continue"):
            kind, _, rest = v.partition(":")
            v = f"{kind}:{tail(rest)}"
        if (k, str(v)) in CLAP_DEFAULTS:
            continue
        out[k] = v
    return sub, out


def tail(path: str) -> str:
    """The file / directory name: the logs ran from pod copies (`/root/v2.tables`,
    `/w/h3-8step`), the catalog from `/workspace/w/h3-rs/tables/v2.tables`."""
    return path.rstrip("/").rsplit("/", 1)[-1]


def built(catalog, inp, media=None, fn="generate"):
    j = catalog.validate(inp)
    if fn == "regen":
        return flags.h3_regen2k_argv(j, S, catalog.knobs, Path("/x/fox.y4m"), Path("out-2k.y4m"), 7)
    return flags.h3_generate_argv(j, S, catalog.knobs, media or {}, Path("out.y4m"), 7)


REF = {"references": [("image", "/x/assets/first.png", None)]}
FL = {"first_image": "/x/assets/first.png", "last_image": "/x/assets/last.png"}
CASES = [
    ("v2-best", {"engine": "h3", "model": "fasth3-v2", "prompt": "P"}, None, "generate"),
    ("v2-budget", {"engine": "h3", "model": "fasth3-v2", "prompt": "P", "options": {"vsa_sparsity": 0.9, "dense_first_steps": 2}}, None, "generate"),
    ("v2-stream", {"engine": "h3", "model": "fasth3-v2", "prompt": "P", "options": {"stream_blocks": True}}, None, "generate"),
    ("plug-fast", {"engine": "h3", "model": "h3-plug4", "prompt": "P"}, None, "generate"),
    ("plug-fl2va", {"engine": "h3", "model": "h3-plug4", "task": "fl2v", "prompt": "P", "inputs": {"first_image": {"url": "https://e/a.png"}, "last_image": {"url": "https://e/b.png"}}}, FL, "generate"),
    ("plug-ref2va", {"engine": "h3", "model": "h3-plug4", "task": "ref2v", "prompt": "P", "inputs": {"references": [{"url": "https://e/a.png"}]}}, REF, "generate"),
    ("base49-fast", {"engine": "h3", "model": "h3", "prompt": "P", "options": {"steps": 49, "tables": False, "attention": "dense"}}, None, "generate"),
    ("omni", {"engine": "h3", "model": "fasth3-omni", "task": "ref2v", "prompt": "P", "inputs": {"references": [{"url": "https://e/a.png"}]}, "options": {"weights": "native", "decoder_weights": "native", "vsa_qk8": False, "pv8": False}}, REF, "generate"),
    ("trim", {"engine": "h3", "model": "fasth3-trim", "prompt": "P", "options": {"decoder_weights": "native", "vsa_qk8": False, "pv8": False}}, None, "generate"),
    ("pika-fl2va", {"engine": "h3", "model": "pika-flex", "task": "fl2v", "prompt": "P", "inputs": {"first_image": {"url": "https://e/a.png"}, "last_image": {"url": "https://e/b.png"}}, "options": {"weights": "native", "decoder_weights": "native", "pv8": False}}, FL, "generate"),
    ("regen2k", {"engine": "h3", "model": "regen-2k", "task": "regen2k", "prompt": "P", "width": 2560, "height": 1440, "inputs": {"draft": {"url": "https://e/d.mp4"}}, "options": {"weights": "native", "pv8": False}}, None, "regen"),
]


@pytest.mark.parametrize("name,inp,media,fn", CASES, ids=[c[0] for c in CASES])
def test_h3_lines_match_the_run_logs(catalog, name, inp, media, fn):
    want_sub, want = flag_map(shlex.split(LOG_LINES[name]))
    got_sub, got = flag_map(built(catalog, inp, media, fn))
    if name == "base49-fast":
        # 49 steps and dense attention are the base's own preset: the log left
        # them out, the model's knobs spell them.
        got.pop("--steps", None)
        got.pop("--attention", None)
    assert got_sub == want_sub
    assert got == want


def test_h3_serve_line_and_key(catalog):
    j = catalog.validate({"engine": "h3", "model": "fasth3-v2", "prompt": "P", "upscale": 2})
    argv = flags.h3_serve_argv(j, S, catalog.knobs, 7861)
    assert argv[:6] == ["h3", "serve", "--backend", "cuda", "--addr", "127.0.0.1:7861"]
    assert "--max-frames" in argv and argv[argv.index("--max-frames") + 1] == "362"
    assert argv[argv.index("--upscale") + 1] == S.esrgan
    assert "--task" not in argv
    assert flags.h3_serve_key(j, S, catalog.knobs) == flags.h3_serve_key(catalog.validate({"engine": "h3", "model": "fasth3-v2", "prompt": "Q", "upscale": 2}), S, catalog.knobs)
    assert flags.h3_serve_key(j, S, catalog.knobs) != flags.h3_serve_key(catalog.validate({"engine": "h3", "model": "fasth3-v1", "prompt": "P"}), S, catalog.knobs)


def test_h3_synthetic_serve_line(catalog):
    s = flags.EngineSettings(h3_bin="/b/h3", backend="cpu", synthetic=True)
    j = catalog.validate({"engine": "h3", "model": "fasth3-v2", "prompt": "P", "width": 480, "height": 272})
    assert flags.h3_serve_argv(j, s, catalog.knobs, 7871)[:7] == ["/b/h3", "serve", "--backend", "cpu", "--addr", "127.0.0.1:7871", "--synthetic"]


def test_h3_tables_follow_steps(catalog):
    j = catalog.validate({"engine": "h3", "model": "h3", "prompt": "P"})  # knob_defaults: steps 25, dense
    argv = flags.h3_generate_argv(j, S, catalog.knobs, {}, Path("o.y4m"), 1)
    assert "--tables" in argv and argv[argv.index("--steps") + 1] == "25"
    j = catalog.validate({"engine": "h3", "model": "h3", "prompt": "P", "options": {"steps": 30}})
    assert "--tables" not in flags.h3_generate_argv(j, S, catalog.knobs, {}, Path("o.y4m"), 1)


def test_h3_turbo_ref_uses_transformer_ref(catalog):
    j = catalog.validate({"engine": "h3", "model": "h3-turbo4-ref", "task": "ref2v", "prompt": "P", "inputs": {"references": [{"url": "https://e/a.png"}]}})
    argv = flags.h3_generate_argv(j, S, catalog.knobs, REF, Path("o.y4m"), 1)
    assert argv[argv.index("--transformer") + 1] == S.h3_ref_transformer
    assert "--tables" not in argv
    assert argv[argv.index("--variant-dir") + 1].endswith("h3-ref2va/Minimax-h3-Turbo")


def test_h3_preview_and_upscale_need_their_weights(catalog):
    bare = flags.EngineSettings()
    j = catalog.validate({"engine": "h3", "model": "fasth3-v2", "prompt": "P", "options": {"preview": True}})
    with pytest.raises(JobError, match="TAEH3"):
        flags.h3_generate_argv(j, bare, catalog.knobs, {}, Path("o.y4m"), 1)
    argv = flags.h3_generate_argv(j, S, catalog.knobs, {}, Path("o.y4m"), 1)
    assert "--preview" in argv and argv[argv.index("--tae") + 1] == S.taeh3
    j = catalog.validate({"engine": "h3", "model": "fasth3-v2", "prompt": "P", "upscale": 3})
    with pytest.raises(JobError, match="Real-ESRGAN"):
        flags.h3_generate_argv(j, bare, catalog.knobs, {}, Path("o.y4m"), 1)
    argv = flags.h3_generate_argv(j, S, catalog.knobs, {}, Path("o.y4m"), 1)
    assert argv[argv.index("--upscale-factor") + 1] == "3"


def test_h3_continue_and_video_refs(catalog):
    j = catalog.validate({"engine": "h3", "model": "h3-plug4", "task": "ref2v", "prompt": "P", "inputs": {"references": [{"url": "https://e/a.mp4"}]}})
    argv = flags.h3_generate_argv(j, S, catalog.knobs, {"references": [("video", "/x/r.y4m", "/x/r.wav")]}, Path("o.y4m"), 1)
    assert argv[argv.index("--ref") + 1] == "video:/x/r.y4m,audio=/x/r.wav"
    argv = flags.h3_generate_argv(j, S, catalog.knobs, {"continue": ("/x/c.y4m", None)}, Path("o.y4m"), 1)
    assert argv[argv.index("--continue") + 1] == "/x/c.y4m"


def test_fastwan_serve_and_generate_lines(catalog):
    j = catalog.validate({"engine": "fastwan", "model": "fastwan-1.3b", "prompt": "P", "options": {"weights": "int8", "attention": "vsa", "vsa_sparsity": 0.9, "tae_chunk": 4}, "upscale": 4})
    argv = flags.fastwan_serve_argv(j, S, catalog.knobs, 7862)
    m = flags.parse_flags(argv[2:], flags.FW_VALUE_FLAGS, flags.FW_BOOL_FLAGS)
    assert m["--model-dir"] == "/workspace/w/fastwan" and m["--tae"].endswith("auxiliary/tae/taew2_1.safetensors")
    assert m["--weights"] == "int8" and m["--attention"] == "vsa" and m["--vsa-sparsity"] == "0.9" and m["--tae-chunk"] == "4"
    assert m["--upscaler"] == S.esrgan and m["--max-frames"] == "81" and m["--max-pixels"] == str(832 * 480)
    j = catalog.validate({"engine": "fastwan", "model": "self-forcing-1.3b", "task": "long", "prompt": "P", "frames": 321, "options": {"cache_frames": 12, "sink_frames": 3}, "upscale": 2})
    argv = flags.fastwan_generate_argv(j, S, catalog.knobs, Path("out.y4m"), 7)
    m = flags.parse_flags(argv[2:], flags.FW_VALUE_FLAGS, flags.FW_BOOL_FLAGS)
    assert m["--causal"] is True and m["--attention"] == "dense" and m["--cache-frames"] == "12" and m["--sink-frames"] == "3"
    assert m["--transformer"].endswith("self-forcing-1.3b-f16.safetensors") and m["--text-encoder"].endswith("fastwan/text_encoder")
    assert m["--tokenizer"].endswith("tokenizer/tokenizer.json") and m["--frames"] == "321"
    assert m["--upscale"] == S.esrgan and m["--upscale-x"] == "2"
    j = catalog.validate({"engine": "fastwan", "model": "history-forcing-1.3b", "task": "long", "prompt": "P"})
    m = flags.parse_flags(flags.fastwan_generate_argv(j, S, catalog.knobs, Path("o.y4m"), 7)[2:], flags.FW_VALUE_FLAGS, flags.FW_BOOL_FLAGS)
    assert m["--cache-frames"] == "21"  # the model's knob_defaults


def test_fastwan_synthetic_serve_line(catalog):
    s = flags.EngineSettings(fastwan_bin="/b/fastwan", backend="cpu", synthetic=True)
    j = catalog.validate({"engine": "fastwan", "model": "fastwan-1.3b", "prompt": "P", "width": 480, "height": 272, "frames": 17, "upscale": 2})
    argv = flags.fastwan_serve_argv(j, s, catalog.knobs, 7872)
    assert argv[:7] == ["/b/fastwan", "serve", "--backend", "cpu", "--addr", "127.0.0.1:7872", "--synthetic"]
    assert "--upscaler" in argv


def test_fastvideo_bodies(catalog):
    j = catalog.validate({"engine": "fastvideo", "model": "fasth3-4step-vsa", "task": "fl2v", "prompt": "P", "seed": 3, "frames": 124,
                          "inputs": {"first_image": {"url": "https://e/a.png"}, "last_image": {"url": "https://e/b.png"}}})
    b = flags.fastvideo_body(j, catalog.knobs, {"first_image": "https://e/a.png", "last_image": "data:image/png;base64,AA=="}, 3)
    assert b == {"model": "fasth3-4step-vsa", "prompt": "P", "seed": 3, "size": "1344x768", "num_frames": 124,
                 "image_url": "https://e/a.png", "last_image_url": "data:image/png;base64,AA=="}
    j = catalog.validate({"engine": "fastvideo", "model": "ltx25-distill-sol", "task": "a2v", "prompt": "", "inputs": {"audio": {"url": "https://e/a.wav"}}, "options": {"fps": "50"}})
    b = flags.fastvideo_body(j, catalog.knobs, {"audio": "https://e/a.wav"}, None)
    assert b["audio_url"] == "https://e/a.wav" and b["fps"] == 50 and "seed" not in b
    j = catalog.validate({"engine": "fastvideo", "model": "ltx25-distill-sol", "task": "retake", "prompt": "P", "frames": 49, "inputs": {"video": {"url": "https://e/v.mp4"}}})
    b = flags.fastvideo_body(j, catalog.knobs, {"video": "https://e/v.mp4"}, 1)
    assert b["video_url"] == "https://e/v.mp4" and b["start_s"] == 0.0 and b["end_s"] == pytest.approx(49 / 24, abs=1e-3)
    assert "size" not in b and "num_frames" not in b and "fps" not in b
    j = catalog.validate({"engine": "fastvideo", "model": "h3-ref2v-turbo", "task": "ref2v", "prompt": "P", "inputs": {"references": [{"url": "https://e/a.png"}]}, "options": {"reference_strength": 0.5, "prompt_expansion": "quality"}})
    b = flags.fastvideo_body(j, catalog.knobs, {"references": ["https://e/a.png"]}, None)
    assert b["reference_urls"] == ["https://e/a.png"] and b["reference_strength"] == 0.5 and b["prompt_expansion"] == "quality"
    assert set(b) <= flags.FV_FIELDS


def test_fastvideo_flow_shift_is_not_a_knob(catalog):
    # fv-serve's NativeBody has no flow_shift field (deny_unknown_fields), so the catalog must not offer it.
    assert "fv.flow_shift" not in catalog.knobs
    with pytest.raises(JobError, match="flow_shift"):
        catalog.validate({"engine": "fastvideo", "model": "wan22-ti2v-5b", "prompt": "P", "options": {"flow_shift": 5.0}})


def test_command_line_is_shell_safe():
    assert flags.command_line(["h3", "generate", "--prompt", "it's a 'fox'"]) == "h3 generate --prompt 'it'\"'\"'s a '\"'\"'fox'\"'\"''"
