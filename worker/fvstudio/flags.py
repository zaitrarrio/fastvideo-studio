"""Catalog recipe + job → the engines' exact command lines and fv-serve's
request body. Every flag here is one of the clap definitions in
`h3-rs/crates/h3-cli/src/main.rs` and `fastwan-rs/crates/fastwan-cli/src/main.rs`;
the fv-serve fields are `NativeBody` in
`fastvideo-rs/crates/fastvideo-serve/src/native.rs` (`deny_unknown_fields`).
Pure functions, snapshot-tested against lines from the engines' run logs."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .catalog import Job, Knob
from .errors import JobError

H3_TASK = {"t2v": "t2va", "fl2v": "fl2va", "ref2v": "ref2va", "extend": "ref2va"}
# Flags whose value is the next token (the rest take none).
H3_VALUE_FLAGS = {"--variant", "--task", "--model-dir", "--variant-dir", "--transformer", "--adapter",
                  "--adapter-strength", "--tae", "--tables", "--steps", "--attention", "--vsa-sparsity",
                  "--dense-first-steps", "--weights", "--decoder-weights", "--ffn-chunk", "--backend",
                  "--addr", "--max-pixels", "--max-frames", "--tile", "--upscale", "--upscale-factor",
                  "--width", "--height", "--frames", "--seed", "--prompt", "--out", "--first-image",
                  "--last-image", "--ref", "--continue", "--aspect", "--tier", "--regen-dir", "--draft",
                  "--adaln-table", "--start-sigma", "--embeds", "--gallery", "--allow-origin", "--runs"}
H3_BOOL_FLAGS = {"--vsa-qk8", "--qk8", "--pv8", "--stream-blocks", "--preview", "--synthetic", "--ops-only"}
H3_ALIAS = {"--qk8": "--vsa-qk8"}
FW_VALUE_FLAGS = {"--attention", "--vsa-sparsity", "--weights", "--backend", "--addr", "--model-dir", "--tae",
                  "--transformer", "--gallery", "--max-pixels", "--max-frames", "--tae-chunk", "--allow-origin",
                  "--upscaler", "--embeds", "--prompt", "--text-encoder", "--tokenizer", "--runs", "--seed", "--out",
                  "--ffn-chunk", "--upscale", "--upscale-x", "--cache-frames", "--sink-frames", "--width",
                  "--height", "--frames"}
FW_BOOL_FLAGS = {"--causal", "--synthetic"}

# The order flags are emitted in (readable, stable for the snapshot tests).
H3_ORDER = ["--variant", "--task", "--model-dir", "--variant-dir", "--transformer", "--adapter", "--adapter-strength",
            "--tables", "--tae", "--steps", "--attention", "--vsa-sparsity", "--dense-first-steps", "--vsa-qk8",
            "--pv8", "--weights", "--decoder-weights", "--stream-blocks", "--preview"]
FW_ORDER = ["--model-dir", "--transformer", "--tae", "--text-encoder", "--tokenizer", "--weights", "--attention",
            "--vsa-sparsity", "--tae-chunk", "--causal", "--cache-frames", "--sink-frames"]

# fv-serve's `NativeBody` fields (native.rs): a knob whose `field` is not here cannot be sent.
FV_FIELDS = {"model", "prompt", "negative_prompt", "seed", "size", "width", "height", "aspect_ratio", "short_edge",
             "seconds", "num_frames", "fps", "steps", "guidance", "image_url", "last_image_url", "reference_urls",
             "reference_strength", "reference_lora_strength", "audio_url", "video_url", "start_s", "end_s",
             "retake_mode", "extend_s", "extend_at", "context_s", "prompt_expansion"}
CONTINUE_FRAMES = 39  # h3 `--continue`: 5 / 22 / 39 frames at the canvas


@dataclass(frozen=True)
class EngineSettings:
    """Worker-side paths the command lines need."""

    h3_bin: str = "h3"
    fastwan_bin: str = "fastwan"
    backend: str = "cuda"
    synthetic: bool = False
    esrgan: str | None = None  # realesr-animevideov3.safetensors
    taeh3: str | None = None  # taeh3.safetensors for h3 --preview
    h3_ref_transformer: str = "/workspace/w/h3-ref2va/transformer_ref"


# ---- flag maps -------------------------------------------------------------------


def parse_flags(tokens: list[str], value_flags: set[str], bool_flags: set[str], alias: dict[str, str] | None = None) -> dict[str, Any]:
    """`["--weights", "mxfp8", "--pv8"]` → `{"--weights": "mxfp8", "--pv8": True}`
    (repeated value flags collect into a list)."""
    out: dict[str, Any] = {}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if not t.startswith("--"):
            raise ValueError(f"unexpected token {t!r} in flags")
        f = (alias or {}).get(t, t)
        if "=" in t:
            f, v = t.split("=", 1)
            f = (alias or {}).get(f, f)
            _put(out, f, v)
            i += 1
        elif f in bool_flags:
            out[f] = True
            i += 1
        elif f in value_flags:
            if i + 1 >= len(tokens):
                raise ValueError(f"flag {t} needs a value")
            _put(out, f, tokens[i + 1])
            i += 2
        else:
            raise ValueError(f"unknown flag {t}")
    return out


def _put(out: dict, f: str, v: str) -> None:
    if f in out:
        out[f] = (out[f] if isinstance(out[f], list) else [out[f]]) + [v]
    else:
        out[f] = v


def emit(flags: dict[str, Any], order: list[str]) -> list[str]:
    argv: list[str] = []
    seen = set()
    for f in order + [f for f in flags if f not in order]:
        if f in seen or f not in flags:
            continue
        seen.add(f)
        v = flags[f]
        if v is None or v is False:
            continue
        if v is True:
            argv.append(f)
        elif isinstance(v, list):
            for x in v:
                argv += [f, str(x)]
        else:
            argv += [f, _fmt(v)]
    return argv


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        s = f"{v:.6g}"
        return s
    return str(v)


def knob_overlay(job: Job, knobs: dict[str, Knob], cli_only: set[str] | None = None) -> dict[str, Any]:
    """Flags from the knobs the job set explicitly plus the model's
    `knob_defaults`; the recipe's own flags cover the rest."""
    out: dict[str, Any] = {}
    for kid in job.model.knobs:
        knob = knobs[kid]
        if not knob.cli:
            continue
        if cli_only is not None and knob.cli not in cli_only:
            continue
        if kid not in job.explicit and kid not in job.model.knob_defaults:
            continue
        # A conditional knob (`when`) is emitted only when its condition holds.
        if any(job.opt(dep) not in (want, "auto") for dep, want in knob.when.items()):
            continue
        v = job.options.get(kid)
        if knob.type == "bool":
            out[knob.cli] = bool(v)
        elif knob.type == "select":
            out[knob.cli] = None if (knob.cli == "--attention" and v == "auto") else v
        else:
            out[knob.cli] = v
    return out


# ---- h3 ------------------------------------------------------------------------


def h3_task(job: Job) -> str:
    try:
        return H3_TASK[job.task]
    except KeyError:
        raise JobError("validate", f"h3 has no task {job.task}") from None


def h3_model_flags(job: Job, s: EngineSettings, knobs: dict[str, Knob], for_serve: bool) -> dict[str, Any]:
    r = job.model.recipe
    flags: dict[str, Any] = {"--variant": r.get("variant", job.model.id)}
    for key, flag in (("model_dir", "--model-dir"), ("variant_dir", "--variant-dir"), ("transformer", "--transformer"),
                      ("adapter", "--adapter")):
        if r.get(key):
            flags[flag] = r[key]
    ref = job.task == "ref2v"
    # Ref2VA on a `transformer` variant (base, plug4, pika) and the Turbo Ref2V
    # LoRAs run over h3-ref2va/transformer_ref, which is under neither
    # --model-dir (h3-base) nor --variant-dir: name it (the 2026-11-15 and
    # 2026-11-17 runs do). Omni's tree carries its own transformer_ref.
    other_dit = ref and not r.get("transformer") and "weights/h3-ref2va" in job.model.trees
    if other_dit:
        flags["--transformer"] = s.h3_ref_transformer
    steps = job.opt("steps") if "h3.steps" in job.model.knobs else None
    steps_explicit = job.is_explicit("steps") or "h3.steps" in job.model.knob_defaults
    if steps_explicit and steps is not None:
        flags["--steps"] = int(steps)
    # AdaLN tables record their DiT: on the other DiT only a `tables_ref` fits.
    tables = r.get("tables_ref") if other_dit else r.get("tables")
    if tables and job.opt("tables", True) and "h3.tables" in job.model.knobs:
        ts = r.get("tables_steps")
        if ts is None or not steps_explicit or int(steps) == int(ts):
            flags["--tables"] = tables
    if job.opt("preview", False):
        if not s.taeh3:
            raise JobError("validate", "preview decoding needs the TAEH3 weights (STUDIO_TAEH3)")
        flags["--tae"] = s.taeh3
        flags["--preview"] = True
    if job.is_explicit("adapter_strength"):
        flags["--adapter-strength"] = float(job.opt("adapter_strength"))
    # serve: an engine started normally serves t2va and fl2va from the same
    # DiT; Ref2VA needs transformer_ref, which only `--task ref2va` loads.
    if not for_serve or h3_task(job) == "ref2va":
        flags["--task"] = h3_task(job)
    return flags


def h3_attn_flags(job: Job, knobs: dict[str, Knob]) -> dict[str, Any]:
    recipe = parse_flags(list(job.model.recipe.get("flags", [])), H3_VALUE_FLAGS, H3_BOOL_FLAGS, H3_ALIAS)
    overlay = knob_overlay(job, knobs)
    overlay.pop("--preview", None)  # handled with --tae in h3_model_flags
    flags = dict(recipe)
    for f, v in overlay.items():
        if v is None or v is False:
            flags.pop(f, None)
        else:
            flags[f] = v
    if flags.get("--attention") == "auto":
        flags.pop("--attention")
    if flags.get("--attention") == "dense":
        flags.pop("--vsa-sparsity", None)
        flags.pop("--dense-first-steps", None)
    return flags


def h3_serve_argv(job: Job, s: EngineSettings, knobs: dict[str, Knob], port: int) -> list[str]:
    argv = [s.h3_bin, "serve", "--backend", s.backend, "--addr", f"127.0.0.1:{port}"]
    if s.synthetic:
        argv.append("--synthetic")
    else:
        flags = {**h3_model_flags(job, s, knobs, for_serve=True), **h3_attn_flags(job, knobs)}
        argv += emit(flags, H3_ORDER)
    argv += ["--max-pixels", str(h3_max_pixels(job)), "--max-frames", str(job.model.frames.max)]
    if job.upscale and job.engine == "h3":
        if not s.esrgan and not s.synthetic:
            raise JobError("validate", "upscaling needs the Real-ESRGAN weights (STUDIO_ESRGAN)")
        argv += ["--upscale", s.esrgan or "synthetic"]
    return argv


def h3_serve_key(job: Job, s: EngineSettings, knobs: dict[str, Knob]) -> str:
    """What a resident `h3 serve` was started for: everything but the port."""
    argv = h3_serve_argv(job, s, knobs, 0)
    return " ".join(argv)


def h3_max_pixels(job: Job) -> int:
    up32 = lambda v: (v + 31) // 32 * 32  # noqa: E731  (finish::canvas_edge)
    return max(up32(w) * up32(h) for w, h in job.model.sizes)


def h3_generate_argv(job: Job, s: EngineSettings, knobs: dict[str, Knob], media: dict[str, Any], out: Path, seed: int) -> list[str]:
    """`media`: `first_image` / `last_image` (Path), `references` (list of
    `(kind, path, audio_path | None)`), `continue` (`(y4m, wav | None)`)."""
    flags = {**h3_model_flags(job, s, knobs, for_serve=False), **h3_attn_flags(job, knobs)}
    argv = [s.h3_bin, "generate", "--backend", s.backend] + emit(flags, H3_ORDER)
    argv += ["--width", str(job.width), "--height", str(job.height), "--frames", str(job.frames), "--seed", str(seed)]
    if media.get("first_image"):
        argv += ["--first-image", str(media["first_image"])]
    if media.get("last_image"):
        argv += ["--last-image", str(media["last_image"])]
    for kind, path, audio in media.get("references", []):
        spec = f"{kind}:{path}"
        if kind == "video" and audio:
            spec += f",audio={audio}"
        argv += ["--ref", spec]
    if media.get("continue"):
        y4m, wav = media["continue"]
        argv += ["--continue", f"{y4m},audio={wav}" if wav else str(y4m)]
    if job.upscale:
        argv += ["--upscale", _esrgan(s), "--upscale-factor", str(job.upscale)]
    argv += ["--prompt", job.prompt, "--out", str(out)]
    return argv


def h3_regen2k_argv(job: Job, s: EngineSettings, knobs: dict[str, Knob], draft: Path, out: Path, seed: int) -> list[str]:
    r = job.model.recipe
    for key in ("regen_dir", "model_dir", "adaln_table"):
        if not r.get(key):
            raise JobError("validate", f"regen2k recipe lacks {key}")
    argv = [s.h3_bin, "regen2k", "--backend", s.backend, "--regen-dir", r["regen_dir"], "--model-dir", r["model_dir"],
            "--draft", str(draft), "--adaln-table", r["adaln_table"]]
    argv += emit(h3_attn_flags(job, knobs), H3_ORDER)
    if job.model.steps:
        argv += ["--steps", str(job.model.steps)]
    argv += ["--seed", str(seed), "--prompt", job.prompt, "--out", str(out)]
    return argv


# ---- fastwan ------------------------------------------------------------------


def fastwan_flags(job: Job, knobs: dict[str, Knob], cli_only: set[str] | None = None) -> dict[str, Any]:
    recipe = parse_flags(list(job.model.recipe.get("flags", [])), FW_VALUE_FLAGS, FW_BOOL_FLAGS)
    overlay = knob_overlay(job, knobs, cli_only)
    flags = dict(recipe)
    for f, v in overlay.items():
        if v is None or v is False:
            flags.pop(f, None)
        else:
            flags[f] = v
    if flags.get("--attention") == "auto":
        flags.pop("--attention")
    if flags.get("--attention") == "dense":
        flags.pop("--vsa-sparsity", None)
    return flags


FW_SERVE_FLAGS = {"--weights", "--attention", "--vsa-sparsity", "--tae-chunk"}


def fastwan_serve_argv(job: Job, s: EngineSettings, knobs: dict[str, Knob], port: int) -> list[str]:
    r = job.model.recipe
    argv = [s.fastwan_bin, "serve", "--backend", s.backend, "--addr", f"127.0.0.1:{port}"]
    if s.synthetic:
        argv.append("--synthetic")
    else:
        if not r.get("model_dir"):
            raise JobError("validate", f"{job.model.id}: recipe has no model_dir for fastwan serve")
        flags = {"--model-dir": r["model_dir"]}
        if r.get("transformer"):
            flags["--transformer"] = r["transformer"]
        if r.get("tae"):
            flags["--tae"] = r["tae"]
        flags.update(fastwan_flags(job, knobs, FW_SERVE_FLAGS))
        argv += emit(flags, FW_ORDER)
    argv += ["--max-pixels", str(max(w * h for w, h in job.model.sizes)), "--max-frames", str(job.model.frames.max)]
    if job.upscale:
        if not s.esrgan and not s.synthetic:
            raise JobError("validate", "upscaling needs the Real-ESRGAN weights (STUDIO_ESRGAN)")
        argv += ["--upscaler", s.esrgan or "synthetic"]
    return argv


def fastwan_serve_key(job: Job, s: EngineSettings, knobs: dict[str, Knob]) -> str:
    return " ".join(fastwan_serve_argv(job, s, knobs, 0))


def fastwan_generate_argv(job: Job, s: EngineSettings, knobs: dict[str, Knob], out: Path, seed: int) -> list[str]:
    r = dict(job.model.recipe)
    # A serve recipe names the FastWan download (transformer/, text_encoder/,
    # tokenizer/): `generate` wants each part spelled out.
    if r.get("model_dir"):
        r.setdefault("transformer", f"{r['model_dir']}/transformer")
        r.setdefault("text_encoder", f"{r['model_dir']}/text_encoder")
        r.setdefault("tokenizer", f"{r['model_dir']}/tokenizer/tokenizer.json")
        r.setdefault("tae", f"{r['model_dir']}/taew2_1.safetensors")
    for key in ("transformer", "tae", "text_encoder", "tokenizer"):
        if not r.get(key):
            raise JobError("validate", f"{job.model.id}: recipe lacks {key} for fastwan generate")
    flags: dict[str, Any] = {"--transformer": r["transformer"], "--tae": r["tae"], "--text-encoder": r["text_encoder"],
                             "--tokenizer": r["tokenizer"]}
    flags.update(fastwan_flags(job, knobs))
    argv = [s.fastwan_bin, "generate", "--backend", s.backend] + emit(flags, FW_ORDER)
    argv += ["--width", str(job.width), "--height", str(job.height), "--frames", str(job.frames), "--seed", str(seed)]
    if job.upscale:
        argv += ["--upscale", _esrgan(s), "--upscale-x", str(job.upscale)]
    argv += ["--prompt", job.prompt, "--out", str(out)]
    return argv


# ---- fastvideo -------------------------------------------------------------------


def fastvideo_body(job: Job, knobs: dict[str, Knob], media_urls: dict[str, Any], seed: int | None) -> dict[str, Any]:
    """`POST /fv/v1/jobs` body. `media_urls`: `first_image` / `last_image` /
    `audio` / `video` (str) and `references` (list[str]), as https or data URIs."""
    body: dict[str, Any] = {"model": job.model.id, "prompt": job.prompt}
    if job.negative_prompt:
        body["negative_prompt"] = job.negative_prompt
    if seed is not None:
        body["seed"] = seed
    edit = job.task in ("extend", "retake")
    if not edit:
        body["size"] = f"{job.width}x{job.height}"
        body["num_frames"] = job.frames
    for kid in job.model.knobs:
        knob = knobs[kid]
        if not knob.fv_field:
            continue
        v = job.options.get(kid)
        if v is None:
            continue
        if kid not in job.explicit and kid not in job.model.knob_defaults and knob.default is None:
            continue
        if knob.fv_field not in FV_FIELDS:
            if kid in job.explicit:
                raise JobError("validate", f"option {knob.short}: fv-serve's jobs API has no `{knob.fv_field}` field")
            continue
        if knob.fv_field == "prompt_expansion" and v == "disabled" and kid not in job.explicit:
            continue
        if knob.fv_field == "fps":
            if edit:
                continue
            v = int(v)
        body[knob.fv_field] = v
    if job.task in ("i2v", "fl2v") and media_urls.get("first_image"):
        body["image_url"] = media_urls["first_image"]
    if job.task == "fl2v" and media_urls.get("last_image"):
        body["last_image_url"] = media_urls["last_image"]
    if job.task == "ref2v":
        body["reference_urls"] = list(media_urls.get("references", []))
    if job.task == "a2v":
        body["audio_url"] = media_urls["audio"]
        if media_urls.get("first_image"):
            body["image_url"] = media_urls["first_image"]
    if edit:
        body["video_url"] = media_urls["video"]
        seconds = job.frames / float(job.fps_for())
        if job.task == "extend":
            body["extend_s"] = round(seconds, 3)
        else:
            body["start_s"] = 0.0
            body["end_s"] = round(seconds, 3)
    unknown = set(body) - FV_FIELDS
    assert not unknown, unknown
    return body


def _esrgan(s: EngineSettings) -> str:
    """The Real-ESRGAN weights, or (synthetic engines) the file's name for the command line."""
    if s.esrgan:
        return s.esrgan
    if s.synthetic:
        return "realesr-animevideov3.safetensors"
    raise JobError("validate", "upscaling needs the Real-ESRGAN weights (STUDIO_ESRGAN)")


def command_line(argv: list[str]) -> str:
    """The line to reproduce a run on a rented box: the binary by name."""
    return shlex.join([argv[0].rsplit("/", 1)[-1], *argv[1:]])
