"""`catalog/catalog.json`: load it and validate a job against it.

A knob the catalog does not list is refused, never defaulted (design §2).
Everything here is pure: no I/O apart from reading the catalog file.
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import JobError

OPS = ("generate", "info", "warm")
UPSCALE_VALUES = (0, 2, 3, 4)
PROMPT_MAX_CHARS = 2000
# Tasks whose prompt fv-serve lets be empty (the source media carries the content).
PROMPT_OPTIONAL_TASKS = ("extend", "retake", "a2v")
MEDIA_INPUT_KEYS = ("first_image", "last_image", "references", "audio", "video", "draft")
MAX_REFERENCES = 12
# Knob-id prefixes per engine: `options` keys may be short (`weights`) or full (`h3.weights`).
KNOB_PREFIX = {"h3": "h3", "fastwan": "fw", "fastvideo": "fv"}
# Knobs any engine's model may list (`expand.prompt`: the worker handles them,
# no `cli` / `field`). The page sends them by short name like the others.
SHARED_KNOB_PREFIXES = ("expand",)


def default_catalog_path() -> Path:
    env = os.environ.get("STUDIO_CATALOG")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for p in (here.parents[2] / "catalog" / "catalog.json", Path("/opt/studio/catalog/catalog.json")):
        if p.is_file():
            return p
    return here.parents[2] / "catalog" / "catalog.json"


@dataclass(frozen=True)
class FrameGrid:
    """`"17n+5"`: frames = step·n + offset, within [min, max]."""

    step: int
    offset: int
    min: int
    max: int
    default: int
    choices: tuple[int, ...]

    @staticmethod
    def parse(spec: dict) -> "FrameGrid":
        m = re.fullmatch(r"\s*(\d+)\s*n\s*\+\s*(\d+)\s*", spec["grid"])
        if not m:
            raise ValueError(f"bad frame grid {spec['grid']!r}")
        return FrameGrid(
            step=int(m.group(1)),
            offset=int(m.group(2)),
            min=int(spec["min"]),
            max=int(spec["max"]),
            default=int(spec["default"]),
            choices=tuple(int(c) for c in spec.get("choices", [])),
        )

    def check(self, frames: int) -> str | None:
        """None when `frames` is on the grid and in range, else why not."""
        if frames < self.min or frames > self.max:
            return f"frames must be between {self.min} and {self.max}"
        if (frames - self.offset) % self.step != 0:
            return f"frames must be on the {self.step}n+{self.offset} grid (e.g. {', '.join(map(str, self.choices[:4]))})"
        return None


@dataclass(frozen=True)
class Knob:
    id: str
    label: str
    type: str  # select | bool | int | float
    default: Any
    options: tuple[str, ...] = ()
    min: float | None = None
    max: float | None = None
    group: str = ""
    cli: str | None = None
    fv_field: str | None = None
    when: dict[str, Any] = field(default_factory=dict)

    @property
    def short(self) -> str:
        return self.id.split(".", 1)[1]

    @staticmethod
    def parse(kid: str, spec: dict) -> "Knob":
        return Knob(
            id=kid,
            label=spec.get("label", kid),
            type=spec["type"],
            default=spec.get("default"),
            options=tuple(str(o[0]) for o in spec.get("options", [])),
            min=spec.get("min"),
            max=spec.get("max"),
            group=spec.get("group", ""),
            cli=spec.get("cli"),
            fv_field=spec.get("field"),
            when=dict(spec.get("when", {})),
        )

    def coerce(self, value: Any) -> Any:
        """The value in the knob's type, or raise ValueError with a reason."""
        if self.type == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.lower() in ("true", "false"):
                return value.lower() == "true"
            raise ValueError("expects true or false")
        if self.type == "select":
            s = str(value).lower() if isinstance(value, bool) else str(value)
            if s not in self.options:
                raise ValueError(f"must be one of {', '.join(self.options)}")
            return s
        if self.type == "int":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
                raise ValueError("expects an integer")
            v = int(value)
        elif self.type == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("expects a number")
            v = float(value)
        else:
            raise ValueError(f"unknown knob type {self.type}")
        if self.min is not None and v < self.min:
            raise ValueError(f"must be at least {self.min}")
        if self.max is not None and v > self.max:
            raise ValueError(f"must be at most {self.max}")
        return v


@dataclass(frozen=True)
class Task:
    id: str
    label: str
    inputs: tuple[str, ...]
    optional: tuple[str, ...]

    @property
    def required(self) -> tuple[str, ...]:
        return tuple(i for i in self.inputs if i not in self.optional)


@dataclass(frozen=True)
class Model:
    id: str
    engine: str
    family: str
    label: str
    tasks: tuple[str, ...]
    sizes: tuple[tuple[int, int], ...]
    default_size: tuple[int, int]
    frames: FrameGrid
    knobs: tuple[str, ...]
    recipe: dict[str, Any]
    trees: tuple[str, ...]
    stage: tuple[str, ...]
    vram_gib: float
    min_sm: int
    fps: int
    audio: bool
    steps: int | None = None
    cli_only: bool = False
    hidden: bool = False
    negative_prompt: bool = False
    aliases: tuple[str, ...] = ()
    knob_defaults: dict[str, Any] = field(default_factory=dict)
    tier: str | None = None
    verify: str | None = None
    raw: dict = field(default_factory=dict, repr=False, compare=False)

    @staticmethod
    def parse(spec: dict) -> "Model":
        return Model(
            id=spec["id"],
            engine=spec["engine"],
            family=spec["family"],
            label=spec.get("label", spec["id"]),
            tasks=tuple(spec["tasks"]),
            sizes=tuple((int(w), int(h)) for w, h in spec["sizes"]),
            default_size=(int(spec["default_size"][0]), int(spec["default_size"][1])),
            frames=FrameGrid.parse(spec["frames"]),
            knobs=tuple(spec.get("knobs", [])),
            recipe=dict(spec.get("recipe", {})),
            trees=tuple(spec.get("trees", [])),
            stage=tuple(spec.get("stage", [])),
            vram_gib=float(spec.get("vram_gib", 0)),
            min_sm=int(spec.get("min_sm", 0)),
            fps=int(spec.get("fps", 24)),
            audio=bool(spec.get("audio", False)),
            steps=spec.get("steps"),
            cli_only=bool(spec.get("cli_only", False)),
            hidden=bool(spec.get("hidden", False)),
            negative_prompt=bool(spec.get("negative_prompt", False)),
            aliases=tuple(spec.get("aliases", [])),
            knob_defaults=dict(spec.get("knob_defaults", {})),
            tier=spec.get("tier"),
            verify=spec.get("verify"),
            raw=spec,
        )


@dataclass(frozen=True)
class Media:
    """One `<media>`: exactly one of url / b64 / r2."""

    kind: str  # url | b64 | r2
    value: str
    mime: str | None = None

    @property
    def inline_bytes(self) -> int:
        if self.kind != "b64":
            return 0
        s = self.value.rstrip("=")
        return (len(s) * 3) // 4

    def decode_b64(self) -> bytes:
        return base64.b64decode(self.value, validate=False)


@dataclass
class Job:
    """A validated `op: generate` input."""

    engine: str
    model: Model
    task: str
    prompt: str
    width: int
    height: int
    frames: int
    seed: int | None
    options: dict[str, Any]  # effective values, full knob ids, defaults applied
    explicit: dict[str, Any]  # the knobs the request set, full knob ids
    inputs: dict[str, Any]  # media per input key (list for references)
    upscale: int
    negative_prompt: str | None = None
    client: Any = None

    def opt(self, short: str, default: Any = None) -> Any:
        """An effective option by its short name (`weights`)."""
        return self.options.get(f"{KNOB_PREFIX[self.engine]}.{short}", default)

    def is_explicit(self, short: str) -> bool:
        return f"{KNOB_PREFIX[self.engine]}.{short}" in self.explicit

    def fps_for(self) -> int:
        """The clip's frame rate: the `fps` knob when the model has one, else the model's."""
        v = self.options.get(f"{KNOB_PREFIX[self.engine]}.fps")
        return int(v) if v else self.model.fps


class Catalog:
    def __init__(self, data: dict):
        self.data = data
        self.version = data.get("version")
        self.weights_root: str = data.get("weights_root", "/workspace/w")
        self.engines: dict[str, dict] = data["engines"]
        self.tasks: dict[str, Task] = {
            k: Task(k, v.get("label", k), tuple(v.get("inputs", [])), tuple(v.get("optional", [])))
            for k, v in data["tasks"].items()
        }
        self.knobs: dict[str, Knob] = {k: Knob.parse(k, v) for k, v in data["knobs"].items()}
        self.models: list[Model] = [Model.parse(m) for m in data["models"]]
        self.trees: dict[str, str] = dict(data.get("trees", {}))
        self.upload_limits: dict[str, int] = dict(data.get("upload_limits", {}))
        self._check()

    @staticmethod
    def load(path: Path | str | None = None) -> "Catalog":
        p = Path(path) if path else default_catalog_path()
        with open(p, encoding="utf-8") as f:
            return Catalog(json.load(f))

    def _check(self) -> None:
        """Internal consistency: every model's knobs and tasks exist."""
        for m in self.models:
            for k in m.knobs:
                if k not in self.knobs:
                    raise ValueError(f"catalog: model {m.id} lists unknown knob {k}")
                if not k.startswith(KNOB_PREFIX[m.engine] + ".") and k.split(".", 1)[0] not in SHARED_KNOB_PREFIXES:
                    raise ValueError(f"catalog: model {m.id} lists knob {k} of another engine")
            for t in m.tasks:
                if t not in self.tasks:
                    raise ValueError(f"catalog: model {m.id} lists unknown task {t}")
            if m.engine not in self.engines:
                raise ValueError(f"catalog: model {m.id} has unknown engine {m.engine}")

    @property
    def inline_limit(self) -> int:
        return int(self.upload_limits.get("inline_b64_bytes", 6 * 1024 * 1024))

    def model(self, engine: str, model_id: str) -> Model | None:
        for m in self.models:
            if m.engine == engine and (m.id == model_id or model_id in m.aliases):
                return m
        return None

    def models_of(self, engine: str) -> list[Model]:
        return [m for m in self.models if m.engine == engine]

    def knob_defaults(self, model: Model) -> dict[str, Any]:
        """Effective defaults: the knob's, then the model's `knob_defaults`."""
        out = {k: self.knobs[k].default for k in model.knobs}
        for k, v in model.knob_defaults.items():
            if k in out:
                out[k] = v
        return out

    # ---- validation ----------------------------------------------------

    def validate(self, inp: Any) -> Job:
        """A `Job` from an `op: generate` input, or `JobError(where="validate")`."""
        bad = lambda msg: JobError("validate", msg)  # noqa: E731
        if not isinstance(inp, dict):
            raise bad("input must be a JSON object")
        op = inp.get("op", "generate")
        if op != "generate":
            raise bad(f"op {op!r} is not generate")
        engine = inp.get("engine")
        if engine not in self.engines:
            raise bad(f"engine must be one of {', '.join(self.engines)}")
        model_id = inp.get("model")
        if not isinstance(model_id, str):
            raise bad("model is required")
        model = self.model(engine, model_id)
        if model is None:
            raise bad(f"unknown {engine} model {model_id!r}")
        task_id = inp.get("task", model.tasks[0])
        if task_id not in model.tasks:
            raise bad(f"model {model.id} does not do task {task_id!r} (it does {', '.join(model.tasks)})")
        task = self.tasks[task_id]

        prompt = inp.get("prompt", "")
        if not isinstance(prompt, str):
            raise bad("prompt must be a string")
        prompt = prompt.strip()
        if len(prompt) > PROMPT_MAX_CHARS:
            raise bad(f"prompt is over {PROMPT_MAX_CHARS} characters")
        if not prompt and task_id not in PROMPT_OPTIONAL_TASKS:
            raise bad("prompt is required")
        negative = inp.get("negative_prompt")
        if negative is not None:
            if not model.negative_prompt:
                raise bad(f"model {model.id} takes no negative_prompt")
            if not isinstance(negative, str) or len(negative) > PROMPT_MAX_CHARS:
                raise bad("negative_prompt must be a string of at most 2000 characters")

        width = inp.get("width", model.default_size[0])
        height = inp.get("height", model.default_size[1])
        if not _is_int(width) or not _is_int(height):
            raise bad("width and height must be integers")
        if (int(width), int(height)) not in model.sizes:
            sizes = ", ".join(f"{w}x{h}" for w, h in model.sizes)
            raise bad(f"{width}x{height} is not one of {model.id}'s sizes ({sizes})")
        frames = inp.get("frames", model.frames.default)
        if not _is_int(frames):
            raise bad("frames must be an integer")
        why = model.frames.check(int(frames))
        if why:
            raise bad(why)
        seed = inp.get("seed")
        if seed is not None:
            if not _is_int(seed) or int(seed) < 0 or int(seed) >= 2**63:
                raise bad("seed must be a non-negative integer")
            seed = int(seed)

        inputs = self._validate_inputs(inp.get("inputs") or {}, task, model)
        explicit = self._validate_options(inp.get("options") or {}, model)
        options = self.knob_defaults(model)
        options.update(explicit)
        # `when`: a conditional knob is accepted only when its condition holds
        # (a value equal to its default is dropped instead of refused: the
        # page may send every control).
        for kid in list(explicit):
            knob = self.knobs[kid]
            for dep_short, want in knob.when.items():
                dep = f"{KNOB_PREFIX[model.engine]}.{dep_short}"
                have = options.get(dep)
                # `auto` leaves the choice to the variant's preset: the knob stays.
                if have != want and have != "auto":
                    if explicit[kid] == knob.default:
                        del explicit[kid]
                        options[kid] = knob.default
                    else:
                        raise bad(f"option {knob.short} applies only when {dep_short} is {want!r} (it is {have!r})")
        upscale = self._validate_upscale(inp.get("upscale"), options, model, explicit)
        return Job(
            engine=engine,
            model=model,
            task=task_id,
            prompt=prompt,
            width=int(width),
            height=int(height),
            frames=int(frames),
            seed=seed,
            options=options,
            explicit=explicit,
            inputs=inputs,
            upscale=upscale,
            negative_prompt=negative,
            client=inp.get("client"),
        )

    def _validate_inputs(self, raw: Any, task: Task, model: Model) -> dict[str, Any]:
        bad = lambda msg: JobError("validate", msg)  # noqa: E731
        if not isinstance(raw, dict):
            raise bad("inputs must be an object")
        for k in raw:
            if k not in MEDIA_INPUT_KEYS:
                raise bad(f"unknown input {k!r}")
            if k not in task.inputs:
                raise bad(f"task {task.id} takes no input {k!r}")
        for k in task.required:
            if k not in raw or raw[k] in (None, [], {}):
                raise bad(f"task {task.id} needs input {k!r}")
        out: dict[str, Any] = {}
        for k, v in raw.items():
            if v is None:
                continue
            if k == "references":
                if not isinstance(v, list) or not v:
                    raise bad("references must be a non-empty list of media")
                if len(v) > MAX_REFERENCES:
                    raise bad(f"at most {MAX_REFERENCES} references")
                out[k] = [self._validate_media(x, f"references[{i}]") for i, x in enumerate(v)]
            else:
                out[k] = self._validate_media(v, k)
        return out

    def _validate_media(self, v: Any, what: str) -> Media:
        bad = lambda msg: JobError("validate", msg)  # noqa: E731
        if not isinstance(v, dict):
            raise bad(f"{what} must be a media object")
        keys = [k for k in ("url", "b64", "r2") if v.get(k)]
        if len(keys) != 1:
            raise bad(f"{what} must have exactly one of url, b64, r2")
        kind = keys[0]
        value = v[kind]
        if not isinstance(value, str):
            raise bad(f"{what}.{kind} must be a string")
        if kind == "url":
            if not value.startswith("https://"):
                raise bad(f"{what}.url must be https://")
            return Media("url", value)
        if kind == "r2":
            if not value.startswith("uploads/") or ".." in value:
                raise bad(f"{what}.r2 must be an uploads/<key> object")
            return Media("r2", value)
        mime = v.get("mime")
        if not isinstance(mime, str) or "/" not in mime:
            raise bad(f"{what}.mime is required with b64")
        m = Media("b64", value, mime)
        if m.inline_bytes > self.inline_limit:
            raise bad(f"{what} inline data is over the {self.inline_limit // (1024 * 1024)} MiB limit; upload it instead")
        return m

    def _validate_options(self, raw: Any, model: Model) -> dict[str, Any]:
        bad = lambda msg: JobError("validate", msg)  # noqa: E731
        if not isinstance(raw, dict):
            raise bad("options must be an object")
        prefix = KNOB_PREFIX[model.engine]
        out: dict[str, Any] = {}
        for key, value in raw.items():
            kid = key if "." in key else f"{prefix}.{key}"
            if kid not in model.knobs:
                # A shared knob by its short name (`prompt` → `expand.prompt`).
                shared = [k for k in model.knobs if k.split(".", 1)[0] in SHARED_KNOB_PREFIXES and k.split(".", 1)[1] == key]
                if len(shared) != 1:
                    raise bad(f"model {model.id} has no option {key!r}")
                kid = shared[0]
            knob = self.knobs[kid]
            if value is None:
                continue
            try:
                out[kid] = knob.coerce(value)
            except ValueError as e:
                raise bad(f"option {knob.short}: {e}") from None
        return out

    def _validate_upscale(self, raw: Any, options: dict, model: Model, explicit: dict) -> int:
        """Top-level `upscale: 0|2|3|4` (jobs.md). The `*.upscale_factor` knob
        is the page's control for it: accepted in `options` and ignored."""
        bad = lambda msg: JobError("validate", msg)  # noqa: E731
        kid = f"{KNOB_PREFIX[model.engine]}.upscale_factor"
        explicit.pop(kid, None)
        if raw is None or raw is False:
            factor = 0
        else:
            if raw is True:
                raise bad("upscale must be 0, 2, 3 or 4 (not true)")
            if not _is_int(raw) or int(raw) not in UPSCALE_VALUES:
                raise bad("upscale must be 0, 2, 3 or 4")
            factor = int(raw)
        if factor:
            if kid not in model.knobs:
                raise bad(f"model {model.id} has no upscaler")
            if str(factor) not in self.knobs[kid].options:
                raise bad(f"model {model.id} upscales by {', '.join(o for o in self.knobs[kid].options if o != '0')} only")
        if kid in model.knobs:
            options[kid] = str(factor)
        return factor


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)
