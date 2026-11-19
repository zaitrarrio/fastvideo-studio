"""Prompt expansion for every engine through a llama-server sidecar
(Qwen3-VL-4B-Instruct, Q8_0 + F16 vision projector): one chat completion
per job that rewrites the prompt in the spirit of the model family's prompt
guide, cleaned and held under the family's text-encoder budget. A port of
fastvideo-rs's batch rewrites (`crates/fastvideo-upsampler/src/rewrite.rs`,
`docs/serve/prompt-upsampling.md` "Batch rewrites") and of its launcher's
llama-server flags (`deploy/runpod/fv-upsampler.sh`).

The worker wires it in: `plan_sidecar` → `Sidecar(ENGINE, …, llama_probe)`
through `SidecarManager.ensure(…, vram_gib=VRAM_GIB)`, then
`Upsampler(base).expand(...)` before the engine runs. A rewrite that fails
for any reason returns the original prompt and records why
(`Upsampler.last_error`): the job never fails because the LLM did.

Pure apart from `Upsampler.expand` (one HTTP call) and `llama_probe`
(two). Stdlib only.
"""

from __future__ import annotations

import base64
import re
import socket
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Callable

from . import httpc

# ---- the model tree (fastvideo-rs scripts/gpu/weights-manifest.tsv line 98,
# docs/serve/upsampler-local.md §1) --------------------------------------------
TREE_NAME = "qwen3-vl-4b-instruct-gguf"
TREE = f"r2:fv-weights/weights/{TREE_NAME}={TREE_NAME}"
MODEL_FILE = "Qwen3VL-4B-Instruct-Q8_0.gguf"
MMPROJ_FILE = "mmproj-Qwen3VL-4B-Instruct-F16.gguf"
MODEL_BYTES = 4_280_406_144  # upsampler-local.md §1
MMPROJ_BYTES = 836_180_256
FETCH_GIB = round((MODEL_BYTES + MMPROJ_BYTES) / 2**30, 2)  # the first-use fetch: 4.77 GiB (5.12 GB)

# ---- the sidecar (fv-upsampler.sh's defaults, except the context) -----------
ENGINE = "upsampler"  # the SidecarManager key; never an engine the catalog lists
ALIAS = "qwen3-vl-4b-instruct"  # FV_UPSAMPLER_MODEL: llama-server's --alias, the `model` of every request
LLAMA_BIN = "/opt/fastvideo-rs/bin/llama-server"  # in the worker image, from the fastvideo-rs serve image
DEFAULT_PORT = 18080  # FV_UPSAMPLER_PORT
CTX = 16384  # FV_UPSAMPLER_CTX is 65536 for the director's sessions; one Studio rewrite needs far less
PARALLEL = 1  # FV_UPSAMPLER_PARALLEL 2: the Studio rewrites one prompt at a time
KV_TYPE = "q8_0"  # FV_UPSAMPLER_KV_TYPE
CACHE_REUSE = 256  # FV_UPSAMPLER_CACHE_REUSE
CACHE_RAM_MIB = 4096  # FV_UPSAMPLER_CACHE_RAM_MIB

# ---- memory -------------------------------------------------------------------
# Qwen3-VL-4B's language tower: 36 layers, 8 KV heads of 128 (Qwen3-4B's
# config). KV per token = 2 (K, V) × 36 × 8 × 128 values; q8_0 stores 32
# values in 34 bytes (int8 + an f16 scale), f16 2 bytes a value. At 65536
# tokens this gives 4.78 GiB, the "~4.8 GB" fv-upsampler.sh states.
KV_LAYERS, KV_HEADS, KV_HEAD_DIM = 36, 8, 128
KV_BYTES_PER_VALUE = {"q8_0": 34 / 32, "f16": 2.0, "bf16": 2.0, "q4_0": 18 / 32}
COMPUTE_GIB = 0.5  # llama.cpp's compute buffers + the projector's image activations


def kv_bytes(ctx: int = CTX, kv: str = KV_TYPE) -> int:
    """The KV cache in bytes for `ctx` tokens at the `kv` cache type."""
    per_value = KV_BYTES_PER_VALUE.get(kv, 2.0)
    return int(ctx * 2 * KV_LAYERS * KV_HEADS * KV_HEAD_DIM * per_value)


def vram_gib(ctx: int = CTX, kv: str = KV_TYPE, vision: bool = True) -> float:
    """What `SidecarManager.ensure` reserves: the Q8_0 weights (3.99 GiB) +
    the F16 projector (0.78 GiB) + the KV cache (1.20 GiB at 16384 tokens
    q8_0) + compute buffers (0.5 GiB) = 6.5 GiB at the defaults."""
    total = MODEL_BYTES + (MMPROJ_BYTES if vision else 0) + kv_bytes(ctx, kv)
    return round(total / 2**30 + COMPUTE_GIB, 1)


VRAM_GIB = vram_gib()  # 6.5

# ---- the knob -------------------------------------------------------------------
KNOB = "expand.prompt"  # catalog.json: a shared knob every model lists (the worker handles it, no cli / field)
MODES = ("off", "balanced", "quality")


def mode_of(job: Any) -> str:
    """The job's expansion mode: `off`, `balanced` or `quality` (a `Job`'s
    effective options carry the knob under its full id; the page sends it
    as `options.prompt`, which the catalog maps back)."""
    opts = getattr(job, "options", None) or {}
    v = str(opts.get(KNOB) or "off").lower()
    return v if v in MODES else "off"


# ---- families -------------------------------------------------------------------
FAMILIES = ("h3", "ltx", "wan", "other")
# Catalog `family` ids (and engine names) → the prompt guide.
_FAMILY_OF = {"h3": "h3", "fv-h3": "h3", "fastwan": "wan", "wan": "wan", "ltx": "ltx", "ltx2": "ltx", "fastvideo": "other"}


def family_of(name: str | None) -> str:
    """The prompt family of a catalog family / engine id: `h3` (h3, fv-h3),
    `wan` (fastwan, wan), `ltx`, else `other`."""
    return _FAMILY_OF.get((name or "").lower(), "other")


# Prompt budgets in characters, fastvideo-rs's defaults without the family's
# tokenizer (`crates/fastvideo-upsampler/src/lib.rs` 81-97, prompt-upsampling.md
# "Configuration" `[upsampler.max_chars]`): Wan's umT5-XXL takes 512 tokens
# → 1800; LTX's Gemma 3 1024 tokens → 3600; H3's Qwen3-VL text tower 1024
# tokens (refused beyond) → 3200. The Studio has no tokenizer, so characters
# are the limit; the catalog caps the input at 2000 anyway.
MAX_CHARS = {"wan": 1800, "ltx": 3600, "h3": 3200, "other": 1800}
# The answer's token cap: H3's 1024-token layout; Wan's 512.
MAX_TOKENS = {"wan": 512, "ltx": 1024, "h3": 1024, "other": 512}
TEMPERATURE = 0.6  # rewrite.rs `request`

# Target length in words per (mode, family): rewrite.rs `Effort::words`.
WORDS = {
    ("balanced", "wan"): (60, 100),
    ("quality", "wan"): (100, 160),
    ("balanced", "h3"): (60, 110),
    ("quality", "h3"): (120, 220),
    ("balanced", "ltx"): (80, 130),
    ("quality", "ltx"): (140, 200),
    ("balanced", "other"): (80, 130),
    ("quality", "other"): (140, 200),
}

# ---- the system prompts: fastvideo-rs crates/fastvideo-upsampler/src/rewrite.rs
# (COMMON_RULES lines 82-86, family_guide lines 88-103, H3_RULES lines
# 135-139, the image line and the H3 shots in system_prompt lines 105-133),
# verbatim. ----------------------------------------------------------------------
COMMON_RULES = """Rules:
- Keep the user's subjects, intent, quoted text and any style or camera instruction.
- Add concrete visual detail only. One continuous shot, present tense.
- One English paragraph. No title, list, markdown or comments.
- Reply with the prompt only."""

H3_RULES = """Rules:
- Keep the user's subjects, intent, quoted text and any style or camera instruction.
- Add concrete visual detail only. Present tense.
- English, the three labels exactly as shown, each section on its own line. No markdown or comments.
- Reply with the prompt only."""

WAN_GUIDE = """Rewrite the user's request as a video prompt.
Order: subject and action, appearance, setting, style, shot and camera motion, lighting.
Stress motion: what moves, where, how fast. This model makes no sound: no audio, no dialogue."""

LTX_GUIDE = """Rewrite the user's request as a video prompt, the way a cinematographer describes a shot.
Start directly with the main action. Then movements, appearance, background, camera angle and movement, lighting and color.
End with the ambience and sound effects. Spoken words go in double quotes after the speaker."""

H3_GUIDE = """Rewrite the user's request as a video prompt for MiniMax H3 in its structured format, three labelled sections:
integrated_multimodal_description: the setting, each subject with a concrete look, the action through the shot in present tense, every spoken line inline as Name says <d>[English] the line</d>, then camera language (shot size, angle, push in, pan, tracking), lighting and style.
overall_soundscape: ambience and sound effects.
non_diegetic_music: the score, or None."""

GENERIC_GUIDE = """Rewrite the user's request as a detailed video prompt.
Describe the subject, action, setting, camera, lighting and mood."""

FAMILY_GUIDE = {"wan": WAN_GUIDE, "ltx": LTX_GUIDE, "h3": H3_GUIDE, "other": GENERIC_GUIDE}

# rewrite.rs line 115: the text-only image line (the LLM does not see the image).
IMAGE_UNSEEN = "\nThe first frame is an image you cannot see: do not describe its look; describe the motion and camera from it."
# Ours: the image is attached as a vision part (the projector is loaded), the
# way fv-serve's director attaches an opening image.
IMAGE_SEEN = "\nThe first frame is the attached image: keep its subjects and look as they are; describe the motion and camera from it."
H3_SHOTS = {
    "balanced": "Keep it short: one paragraph of description.",
    "quality": "Be rich and specific. When the request has distinct shots, write the description as [Shot 1], [Shot 2] … blocks, each a few sentences; otherwise one paragraph.",
}

# H3's section labels (Context-IR).
H3_DESCRIPTION = "integrated_multimodal_description:"
H3_SOUNDSCAPE = "overall_soundscape:"
H3_MUSIC = "non_diegetic_music:"


def system_prompt(family: str, mode: str, image: bool = False, vision: bool = False, negative: str | None = None) -> str:
    """The system prompt for one rewrite (rewrite.rs `system_prompt`):
    the family's guide, the image line when the clip starts from an image
    (`vision`: it is attached), the H3 shot rule, the rules and the
    length bound. A negative prompt is not rewritten; when given, the LLM
    is told what the user excludes so the rewrite does not add it."""
    family = family if family in FAMILIES else "other"
    mode = mode if mode in ("balanced", "quality") else "balanced"
    lo, hi = WORDS[(mode, family)]
    max_chars = MAX_CHARS[family]
    image_line = (IMAGE_SEEN if vision else IMAGE_UNSEEN) if image else ""
    neg = ""
    if negative and negative.strip():
        neg = f"\n- The user excludes: {' '.join(negative.split())}. Do not add any of it."
    if family == "h3":
        return f"{H3_GUIDE}{image_line}\n{H3_SHOTS[mode]}\n\n{H3_RULES}\n- {lo} to {hi} words, at most {max_chars} characters.{neg}"
    return f"{FAMILY_GUIDE[family]}{image_line}\n\n{COMMON_RULES}\n- {lo} to {hi} words, at most {max_chars} characters.{neg}"


# ---- cleaning (rewrite.rs `strip_wrappers`, `clean`, `cut`, `H3Sections`) -------

def strip_think(s: str) -> str:
    """A leading `<think>…</think>` block removed (chat.rs `strip_think`)."""
    t = s.lstrip()
    if t.startswith("<think>"):
        rest = t[len("<think>"):]
        i = rest.find("</think>")
        return "" if i < 0 else rest[i + len("</think>"):].lstrip()
    if "</think>" in t and "<think>" not in t:
        return t.split("</think>", 1)[1].lstrip()
    return t


_LABELS = ("rewritten prompt:", "enhanced prompt:", "expanded prompt:", "prompt:")
_QUOTE_PAIRS = (('"', '"'), ("“", "”"), ("'", "'"))


def strip_wrappers(answer: str) -> str:
    """Code fences, a leading `Prompt:` label and wrapping quotes removed
    (line breaks kept)."""
    s = strip_think(answer).strip()
    if s.startswith("```"):
        s = s.lstrip("`")
        first, nl, rest = s.partition("\n")
        if nl and " " not in first.strip():  # a language tag on the fence line
            s = rest
        s = s.rstrip().rstrip("`")
    s = s.strip()
    low = s.lower()
    for label in _LABELS:
        if low.startswith(label):
            s = s[len(label):].lstrip()
            break
    s = s.strip()
    if len(s) >= 2:
        for a, b in _QUOTE_PAIRS:
            if s[0] == a and s[-1] == b:
                inner = s[1:-1]
                # Only a wrapper: no other quote of that kind inside (dialogue keeps its quotes).
                if a not in inner and b not in inner:
                    s = inner
                break
    return s


def cut(s: str, max_chars: int) -> str:
    """`s` within `max_chars`: whole when it fits, else up to the last
    sentence end that fits (when that keeps at least half), else the last
    whole word."""
    if len(s) <= max_chars:
        return s
    head = s[:max_chars]
    ends = [m.end() for m in re.finditer(r"[.!?]", head)]
    if ends and ends[-1] >= len(head) // 2:
        return head[:ends[-1]].rstrip()
    i = head.rfind(" ")
    return head[:i].rstrip() if i > 0 else head


def clean(answer: str, max_chars: int) -> str | None:
    """A one-paragraph prompt: wrappers off, whitespace collapsed, cut to
    `max_chars`. `None` when nothing is left."""
    joined = " ".join(strip_wrappers(answer).split())
    if not joined:
        return None
    return cut(joined, max_chars)


def _squash(s: str) -> str:
    lines = [" ".join(l.split()) for l in s.splitlines()]
    return "\n".join(l for l in lines if l)


@dataclass
class H3Sections:
    """The three sections of an H3 prompt (`[Shot N]` blocks stay in the
    description, each on its own line)."""

    description: str
    soundscape: str | None = None
    music: str | None = None

    @staticmethod
    def parse(s: str) -> "H3Sections":
        """Splits an answer at the labels; text without labels is all description."""
        pos = {lab: s.find(lab) for lab in (H3_DESCRIPTION, H3_SOUNDSCAPE, H3_MUSIC)}
        found = sorted(i for i in pos.values() if i >= 0)

        def next_after(start: int) -> int:
            later = [i for i in found if i > start]
            return later[0] if later else len(s)

        d = pos[H3_DESCRIPTION]
        if d >= 0:
            desc = s[d + len(H3_DESCRIPTION):next_after(d)]
        else:
            desc = s[:found[0]] if found else s

        def section(label: str) -> str | None:
            i = pos[label]
            if i < 0:
                return None
            t = _squash(s[i + len(label):next_after(i)]).replace("\n", " ")
            return t or None

        return H3Sections(_squash(desc), section(H3_SOUNDSCAPE), section(H3_MUSIC))

    def render(self) -> str:
        v = [f"{H3_DESCRIPTION} {self.description}"]
        if self.soundscape:
            v.append(f"{H3_SOUNDSCAPE} {self.soundscape}")
        if self.music:
            v.append(f"{H3_MUSIC} {self.music}")
        return "\n".join(v)

    def fit(self, max_chars: int) -> str:
        """Within `max_chars` by priority: the music becomes `None.`, then
        the description is cut at a sentence (else word) boundary, never
        inside `<d>…</d>`; the soundscape is kept."""
        fits = lambda p: len(p) <= max_chars  # noqa: E731
        p = self.render()
        if fits(p):
            return p
        if self.music and self.music.rstrip(".").lower() != "none":
            self.music = "None."
            p = self.render()
            if fits(p):
                return p
        desc = self.description
        self.description = cut_to_fit(desc, lambda d: fits(H3Sections(d, self.soundscape, self.music).render()))
        p = self.render()
        return p if fits(p) else cut_to_fit(p, fits)


def cut_to_fit(s: str, fits: Callable[[str], bool]) -> str:
    """The longest prefix of `s` that `fits`, ending at a sentence end
    (else a word), never inside a `<d>…</d>` dialogue pair (tokens.rs
    `cut_to_fit`)."""
    if fits(s):
        return s
    inside = [False] * (len(s) + 1)
    i = 0
    while True:
        o = s.find("<d>", i)
        if o < 0:
            break
        e = s.find("</d>", o)
        end = len(s) if e < 0 else e + 4
        for x in range(o + 1, end):
            inside[x] = True
        i = end
        if i >= len(s):
            break
    words: list[int] = []
    sentences: list[int] = []
    for b, c in enumerate(s):
        if inside[b]:
            continue
        if c in (" ", "\n"):
            words.append(b)
            prev = s[:b].rstrip()
            if prev.endswith((".", "!", "?", "</d>")):
                sentences.append(len(prev))
    ok = lambda b: fits(s[:b].rstrip())  # noqa: E731

    def best(cands: list[int]) -> int | None:
        lo, hi = 0, len(cands)
        while lo < hi:  # shorter prefixes never exceed a longer one's length
            mid = (lo + hi) // 2
            if ok(cands[mid]):
                lo = mid + 1
            else:
                hi = mid
        return cands[lo - 1] if lo > 0 else None

    sent = best(sentences)
    word = best(words)
    if sent is not None and (word is None or sent * 2 >= word):
        return s[:sent].rstrip()
    if word is not None:
        return s[:word].rstrip()
    return ""


def clean_h3(answer: str, max_chars: int) -> str | None:
    """The H3 cleaning: the Context-IR sections kept on their own lines (an
    answer without labels becomes the description), within `max_chars`."""
    sec = H3Sections.parse(strip_wrappers(answer))
    if not sec.description.strip():
        return None
    return sec.fit(max_chars)


def clean_for(family: str, answer: str, max_chars: int | None = None) -> str | None:
    """`clean` within the family's budget; H3 keeps its sections."""
    family = family if family in FAMILIES else "other"
    limit = MAX_CHARS[family] if max_chars is None else max_chars
    if family == "h3":
        return clean_h3(answer, limit)
    return clean(answer, limit)


# ---- the sidecar: argv, env, probe ------------------------------------------------

def llama_argv(bin: str, model_path: str, mmproj_path: str | None, port: int, ctx: int = CTX,
               parallel: int = PARALLEL, kv: str = KV_TYPE, alias: str = ALIAS,
               cache_reuse: int = CACHE_REUSE, cache_ram_mib: int = CACHE_RAM_MIB) -> list[str]:
    """fv-upsampler.sh `launch`'s command line: model (+ projector), loopback
    host, port, alias, every layer on the GPU, the context, the slots, a
    KV per slot, flash attention with a quantized KV, cached prefix reuse,
    no auto-fit, no web UI."""
    argv = [bin, "-m", model_path]
    if mmproj_path:
        argv += ["--mmproj", mmproj_path]
    argv += [
        "--host", "127.0.0.1", "--port", str(port), "--alias", alias,
        "-ngl", "99", "-c", str(ctx), "-np", str(parallel), "--no-kv-unified",
        "-fa", "on", "-ctk", kv, "-ctv", kv,
        "--cache-reuse", str(cache_reuse), "--cache-ram", str(cache_ram_mib),
        "--fit", "off", "--no-webui",
    ]
    return argv


def plan_sidecar(bin: str, tree_dir: str, port: int = DEFAULT_PORT, env: dict[str, str] | None = None,
                 ctx: int = CTX, vision: bool = True) -> tuple[list[str], dict[str, str]]:
    """`(argv, env)` for a `Sidecar(ENGINE, key, argv, env, llama_probe(base))`
    serving the GGUF tree at `tree_dir` (`<link_root>/qwen3-vl-4b-instruct-gguf`
    after `WeightsResolver.ensure_tree(TREE)`). Pure: the file names are the
    tree's (`MODEL_FILE`, `MMPROJ_FILE`); `env` is passed through (the
    caller's `env_for("fastvideo")`: llama-server is the fastvideo-rs
    image's binary, built against the image's CUDA runtime, not the cu12
    wheels the h3 / fastwan binaries get)."""
    model = f"{tree_dir.rstrip('/')}/{MODEL_FILE}"
    mmproj = f"{tree_dir.rstrip('/')}/{MMPROJ_FILE}" if vision else None
    return llama_argv(bin, model, mmproj, port, ctx=ctx), dict(env or {})


def sidecar_key(tree_dir: str, port: int = DEFAULT_PORT, ctx: int = CTX, vision: bool = True) -> str:
    """What the sidecar was started for (`SidecarManager.ensure`'s `key`)."""
    return " ".join(plan_sidecar(LLAMA_BIN, tree_dir, port, ctx=ctx, vision=vision)[0][1:])


def base_url(port: int = DEFAULT_PORT) -> str:
    return f"http://127.0.0.1:{port}"


def llama_probe(base: str, model: str = ALIAS) -> Callable[[], tuple[bool, str | None]]:
    """A `Sidecar` readiness probe like the launcher's: `/health` 200, then
    one 1-token completion (the CUDA kernels run, not just the load).
    Never fatal: llama-server answers 503 on `/health` while loading."""
    base = base.rstrip("/")

    def probe() -> tuple[bool, str | None]:
        try:
            status, _ = httpc.get_status(f"{base}/health", timeout=2)
        except Exception:  # noqa: BLE001
            return False, None
        if status != 200:
            return False, None
        try:
            r = httpc.post_json(f"{base}/v1/chat/completions",
                                {"model": model, "messages": [{"role": "user", "content": "ok"}], "max_tokens": 1},
                                timeout=30)
        except Exception:  # noqa: BLE001
            return False, None
        return isinstance(r, dict) and bool(r.get("choices")), None

    return probe


# ---- the client ------------------------------------------------------------------

MAX_IMAGE_DATA_URL = 8 * 1024 * 1024  # llama-server takes image parts up to 10 MB; the inline media limit is 6 MiB


@dataclass
class Upsampler:
    """One llama-server's OpenAI-compatible chat API on `base`
    (`http://127.0.0.1:18080`). `vision`: the projector is loaded, so an
    image goes as an `image_url` content part (fv-serve's `OpenAiChat::body`)."""

    base: str = base_url()
    model: str = ALIAS
    vision: bool = True
    log: Callable[[str], None] = field(default=lambda s: None)
    last_error: str | None = None
    calls: int = 0
    failures: int = 0

    @property
    def url(self) -> str:
        return f"{self.base.rstrip('/')}/v1/chat/completions"

    def request_body(self, prompt: str, family: str, mode: str, negative: str | None = None,
                     image_b64: str | None = None, image_mime: str = "image/jpeg") -> dict:
        """The chat request (pure): system prompt per family / mode, the
        prompt as the user text, the image as a `data:` part when the
        projector is loaded (else the system prompt says an unseen image
        sets the first frame), temperature 0.6, the family's token cap."""
        family = family_of(family) if family not in FAMILIES else family
        has_image = bool(image_b64)
        attach = has_image and self.vision and len(image_b64 or "") <= MAX_IMAGE_DATA_URL
        system = system_prompt(family, mode, image=has_image, vision=attach, negative=negative)
        user: Any = prompt
        if attach:
            user = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{image_b64}"}},
            ]
        return {
            "model": self.model,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS[family],
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }

    def expand(self, prompt: str, family: str, mode: str, negative: str | None = None,
               image_b64: str | None = None, image_mime: str = "image/jpeg", timeout_s: float = 60) -> str:
        """The rewritten prompt, or `prompt` itself when the mode is off, the
        prompt is empty, or the call fails (HTTP error, timeout, connection
        refused, no content, nothing left after cleaning): `last_error` then
        says why, for the job log."""
        self.last_error = None
        if mode not in ("balanced", "quality") or not prompt.strip():
            return prompt
        fam = family if family in FAMILIES else family_of(family)
        body = self.request_body(prompt, fam, mode, negative, image_b64, image_mime)
        self.calls += 1
        try:
            r = httpc.post_json(self.url, body, timeout=timeout_s)
            answer = _content_of(r)
            if answer is None:
                raise RuntimeError("no content in the completion")
            out = clean_for(fam, answer)
            if not out:
                raise RuntimeError("nothing left after cleaning the completion")
            return out
        except httpc.HttpError as e:
            self._fail(f"HTTP {e.status}: {e.body[:200]}")
        except (socket.timeout, TimeoutError):
            self._fail(f"timeout after {timeout_s:.0f} s")
        except urllib.error.URLError as e:
            self._fail(f"{e.reason}")
        except Exception as e:  # noqa: BLE001
            self._fail(f"{type(e).__name__}: {e}")
        return prompt

    def _fail(self, why: str) -> None:
        self.failures += 1
        self.last_error = why
        self.log(f"[{ENGINE}] prompt expansion failed ({why}); generating with the original prompt")


def _content_of(r: Any) -> str | None:
    """The first choice's message text of a chat completion, or None."""
    try:
        choices = r["choices"]
        msg = choices[0]["message"]
        content = msg.get("content")
    except (KeyError, IndexError, TypeError, AttributeError):
        return None
    if isinstance(content, list):  # content parts
        content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    if not isinstance(content, str) or not content.strip():
        return None
    return content


def image_data_b64(path: str) -> str:
    """A file's bytes as base64 (for `expand(image_b64=…)`)."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")
