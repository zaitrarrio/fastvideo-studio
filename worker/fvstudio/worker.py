"""The worker: configuration from the environment, the three ops
(`generate`, `info`, `warm`), the generate pipeline (validate → stage →
run → mux → upload) and the jobs.md output / error shapes."""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import __version__, gpu, mux
from .catalog import Catalog, Job
from .engines import Ctx, RunResult, fastvideo, fastwan, h3
from .errors import JobError
from . import media as md, upsampler as up
from .sidecar import Sidecar
from .flags import EngineSettings
from .progress import Reporter
from .r2 import LocalStore, ObjectStore, R2Config, R2Store
from .sidecar import SidecarManager
from .weights import Roots, WeightsResolver

Progress = Callable[[dict], None]
DRIVERS = {"h3": h3.run, "fastwan": fastwan.run, "fastvideo": fastvideo.run}


def env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    catalog_path: Path
    bin_dir: Path = Path("/opt/studio/bin")
    fv_serve_bin: str = "/opt/fastvideo-rs/bin/fv-serve"
    fv_config: str = "/opt/studio/configs/fastvideo.toml"
    fv_state_dir: str = "/workspace/fv-state"
    jobs_dir: Path = Path("/workspace/jobs")
    ports: dict[str, int] = field(default_factory=lambda: {"h3": 7861, "fastwan": 7862, "fastvideo": 8010, "upsampler": up.DEFAULT_PORT})  # not 8000: the serve mode listens there on a pod
    llama_bin: str = up.LLAMA_BIN
    cuda12_lib: str = ""  # a separate CUDA runtime for h3 / fastwan; empty since the one-runtime image
    backend: str = "cuda"
    synthetic: bool = False
    fv_fake: bool = False
    image: str = ""
    url_ttl_s: int = 24 * 3600
    esrgan: str | None = None
    taeh3: str | None = None
    keep_job_dirs: bool = False
    load_timeout_s: float = 1800.0

    @staticmethod
    def from_env() -> "Config":
        e = os.environ
        c = Config(catalog_path=Path(e["STUDIO_CATALOG"]) if e.get("STUDIO_CATALOG") else _default_catalog())
        c.bin_dir = Path(e.get("STUDIO_BIN_DIR", str(c.bin_dir)))
        c.fv_serve_bin = e.get("STUDIO_FV_SERVE", c.fv_serve_bin)
        c.fv_config = e.get("STUDIO_FV_CONFIG", c.fv_config)
        c.llama_bin = e.get("STUDIO_LLAMA_SERVER", c.llama_bin)
        c.fv_state_dir = e.get("STUDIO_FV_STATE_DIR", c.fv_state_dir)
        c.jobs_dir = Path(e.get("STUDIO_JOBS_DIR", str(c.jobs_dir)))
        for eng, var in (("h3", "STUDIO_H3_PORT"), ("fastwan", "STUDIO_FASTWAN_PORT"), ("fastvideo", "STUDIO_FV_PORT")):
            if e.get(var):
                c.ports[eng] = int(e[var])
        c.cuda12_lib = e.get("STUDIO_CUDA12_LIB", c.cuda12_lib)
        c.backend = e.get("STUDIO_BACKEND", c.backend)
        c.synthetic = env_flag("STUDIO_SYNTHETIC")
        c.fv_fake = env_flag("STUDIO_FV_FAKE")
        c.image = e.get("STUDIO_IMAGE", "")
        c.url_ttl_s = int(e.get("STUDIO_URL_TTL_S", c.url_ttl_s))
        c.esrgan = e.get("STUDIO_ESRGAN") or "/workspace/w/h3-rs/assets/realesr-animevideov3.safetensors"
        c.taeh3 = e.get("STUDIO_TAEH3") or "/workspace/w/auxiliary/tae/taeh3.safetensors"
        c.keep_job_dirs = env_flag("STUDIO_KEEP_JOB_DIRS")
        c.load_timeout_s = float(e.get("STUDIO_LOAD_TIMEOUT_S", c.load_timeout_s))
        return c


def _default_catalog() -> Path:
    from .catalog import default_catalog_path

    return default_catalog_path()


class Worker:
    def __init__(self, cfg: Config, catalog: Catalog, store: ObjectStore | None, bucket: str | None,
                 weights_store: ObjectStore | None = None, log: Callable[[str], None] | None = None):
        self.cfg = cfg
        self.catalog = catalog
        self.store = store
        self.bucket = bucket
        self.log = log or (lambda s: print(s, file=sys.stderr, flush=True))
        self.roots = Roots.from_catalog(catalog.trees)
        self.weights = WeightsResolver(self.roots, weights_store or store, log=self.log, dry_run=env_flag("STUDIO_STAGE_DRY_RUN"))
        self.sidecars = SidecarManager(log=self.log, load_timeout_s=cfg.load_timeout_s)
        self.settings = EngineSettings(
            h3_bin=str(cfg.bin_dir / "h3"), fastwan_bin=str(cfg.bin_dir / "fastwan"), backend=cfg.backend,
            synthetic=cfg.synthetic,
            esrgan=cfg.esrgan if cfg.esrgan and Path(cfg.esrgan).is_file() else None,
            taeh3=cfg.taeh3 if cfg.taeh3 and Path(cfg.taeh3).is_file() else None,
            h3_ref_transformer=str(self.roots.link_root / "h3-ref2va" / "transformer_ref"),
        )
        self.started = time.time()
        self.jobs_done = 0

    @staticmethod
    def from_env(log: Callable[[str], None] | None = None) -> "Worker":
        cfg = Config.from_env()
        catalog = Catalog.load(cfg.catalog_path)
        store: ObjectStore | None = None
        bucket: str | None = None
        local = os.environ.get("STUDIO_R2_LOCAL_DIR")
        if local:
            store = LocalStore(Path(local))
            bucket = os.environ.get("STUDIO_R2_BUCKET") or os.environ.get("FV_R2_BUCKET") or "studio"
        else:
            r2 = R2Config.from_env()
            if r2 is not None:
                store = R2Store(r2)
                bucket = r2.bucket
        return Worker(cfg, catalog, store, bucket, log=log)

    # ---- environment per engine ------------------------------------------------

    def env_for(self, engine: str) -> dict[str, str]:
        env = dict(os.environ)
        if engine in ("h3", "fastwan"):
            if self.cfg.cuda12_lib:  # STUDIO_CUDA12_LIB: a separate runtime ahead of the image's
                env["LD_LIBRARY_PATH"] = self.cfg.cuda12_lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        else:
            env["FV_WEIGHTS"] = str(self.roots.link_root)
            env["FV_STATE_DIR"] = self.cfg.fv_state_dir
            env.setdefault("RUST_LOG", "info")
        return env

    # ---- ops -------------------------------------------------------------------------

    def handle(self, job: dict, progress: Progress | None = None) -> dict:
        inp = job.get("input") if isinstance(job, dict) else None
        if not isinstance(inp, dict):
            return JobError("validate", "input must be an object").to_output()
        op = inp.get("op", "generate")
        try:
            if op == "info":
                return self.info()
            if op == "warm":
                return self.warm(inp, progress)
            if op == "generate":
                return self.generate(str(job.get("id") or f"local-{int(time.time())}"), inp, progress)
            if op == "r2":
                return self.r2check()
            if op == "check":
                return self.check(inp)
            raise JobError("validate", f"unknown op {op!r}")
        except JobError as e:
            self.log(f"job failed ({e.where}): {e.message}")
            return e.to_output()
        except Exception as e:  # noqa: BLE001
            self.log(f"job crashed: {type(e).__name__}: {e}")
            # Only the job's own engine's sidecar log: another engine's tail
            # misled the first smoke (fastwan's crash carried h3's log).
            sc = self.sidecars.get(str(inp.get("engine") or ""))
            tail = sc.tail(50) if sc is not None else []
            return JobError("engine", f"{type(e).__name__}: {e}", tail[-50:]).to_output()

    def expand_prompt(self, job: Job, job_dir: Path, report: Reporter) -> str | None:
        """Prompt expansion (the `expand.prompt` knob) for every engine through
        the llama-server sidecar (Qwen3-VL-4B-Instruct, fetched from R2 on
        first use). Returns the rewritten prompt, or None when off or when
        anything failed: a failed rewrite never fails the job."""
        mode = up.mode_of(job)
        if mode == "off" or not job.prompt.strip():
            return None
        try:
            report("staging", "prompt expansion model", force=True)
            self.weights.ensure_tree(up.TREE)
            tree_dir = str(self.roots.link_root / up.TREE_NAME)
            port = self.cfg.ports.get("upsampler", up.DEFAULT_PORT)
            argv, env = up.plan_sidecar(self.cfg.llama_bin, tree_dir, port, env=self.env_for("fastvideo"))
            key = up.sidecar_key(tree_dir, port)
            base = up.base_url(port)
            self.sidecars.ensure(up.ENGINE, key, lambda: Sidecar(up.ENGINE, key, argv, env, up.llama_probe(base)),
                                 up.VRAM_GIB, lambda st, t: report("loading", f"prompt expansion {t:.0f} s"))
            image_b64 = image_mime = None
            first = job.inputs.get("first_image")
            if first is not None and job.task in ("i2v", "fl2v"):
                lm = md.fetch_media(first, job_dir / "in", "first_image", self.store, self.bucket)
                image_b64, image_mime = up.image_data_b64(str(lm.path)), lm.mime
            u = up.Upsampler(base, vision=True, log=self.log)
            report("encoding", "expanding the prompt", force=True)
            new = u.expand(job.prompt, job.model.family, mode, negative=job.negative_prompt,
                           image_b64=image_b64, image_mime=image_mime or "image/jpeg", timeout_s=90)
            if u.last_error is None and new.strip() and new != job.prompt:
                job.prompt = new
                return new
            return None
        except Exception as e:  # noqa: BLE001
            self.log(f"[upsampler] prompt expansion unavailable ({type(e).__name__}: {e}); generating with the original prompt")
            return None

    def check(self, inp: dict) -> dict:
        """`op: check`: the engines' own conformance checkers on this card
        (`h3 check --backend cuda`, `fastwan check --backend cuda`): every op
        against the CPU reference, the way the engines' CLAUDE.md trusts GPU
        code. Stops the sidecars first (the checkers want the GPU)."""
        engines = inp.get("engines") or ["h3", "fastwan"]
        self.sidecars.stop_all()
        out: dict = {"op": "check", "backend": self.cfg.backend, "results": {}}
        for name in engines:
            if name not in ("h3", "fastwan"):
                out["results"][name] = {"error": "no checker for this engine"}
                continue
            argv = [str(self.cfg.bin_dir / name), "check", "--backend", self.cfg.backend]
            t0 = time.time()
            try:
                r = subprocess.run(argv, capture_output=True, text=True, timeout=float(inp.get("timeout_s", 1800)), env=self.env_for(name))
                lines = (r.stdout + r.stderr).strip().splitlines()
                out["results"][name] = {"exit": r.returncode, "seconds": round(time.time() - t0, 1),
                                        "tail": lines[-40:], "command": " ".join(argv)}
            except Exception as e:  # noqa: BLE001
                out["results"][name] = {"error": f"{type(e).__name__}: {e}", "command": " ".join(argv)}
        g = gpu.query()
        out["gpu"] = g.name if g else None
        out["driver"] = g.driver if g else None
        return out

    def r2check(self) -> dict:
        """`op: r2`: can this worker write the Studio bucket and read the
        weights store? Names and outcomes only, never a secret."""
        e = os.environ
        key = e.get("STUDIO_R2_ACCESS_KEY_ID") or e.get("FV_R2_ACCESS_KEY_ID") or ""
        out: dict = {
            "op": "r2",
            "bucket": self.bucket,
            "endpoint": e.get("STUDIO_R2_ENDPOINT") or e.get("FV_R2_ENDPOINT") or None,
            "access_key_id_len": len(key),
            "access_key_id_unsubstituted": key.startswith("{{"),
            "weights_bucket": self.weights.weights_bucket,
        }
        if self.store is None or not self.bucket:
            out["write"] = "no object store configured"
            return out
        probe = self.cfg.jobs_dir / "r2-probe.txt"
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text(f"fastvideo-studio r2 probe {time.time():.0f}\n")
        try:
            self.store.put_file(self.bucket, "probe/r2check.txt", probe, "text/plain")
            out["write"] = "ok"
            out["presigned_url_host"] = self.store.presign_get(self.bucket, "probe/r2check.txt", 60).split("/")[2]
        except Exception as ex:  # noqa: BLE001
            out["write"] = f"{type(ex).__name__}: {ex}"[:300]
        try:
            keys = [k for k, _ in self.store.list_keys(self.weights.weights_bucket, "manifests/")]
            out["read_weights"] = f"ok ({len(keys)} manifests)"
        except Exception as ex:  # noqa: BLE001
            out["read_weights"] = f"{type(ex).__name__}: {ex}"[:300]
        return out

    def info(self) -> dict:
        g = gpu.query()
        trees: dict[str, dict] = {}
        for m in self.catalog.models:
            for t in m.trees:
                if t not in trees:
                    st = self.weights.status(t)
                    trees[t] = {"volume": st.volume, "staged": st.staged, "linked": st.linked}
        return {
            "op": "info",
            "worker_version": __version__,
            "gpu": g.name if g else None,
            "driver": g.driver if g else None,
            "vram_total_gib": round(g.memory_total_mib / 1024, 1) if g else None,
            "vram_free_gib": round(g.free_gib, 1) if g else None,
            "compute_cap": g.compute_cap if g else None,
            "image": self.cfg.image or os.environ.get("RUNPOD_IMAGE", ""),
            "catalog_version": self.catalog.version,
            "versions": self.versions(),
            "sidecars": self.sidecars.resident(),
            "upsampler": {"model": up.ALIAS, "tree": up.TREE_NAME, "port": self.cfg.ports.get("upsampler", up.DEFAULT_PORT),
                          "vram_gib": up.VRAM_GIB, "resident": up.ENGINE in self.sidecars.sidecars,
                          "binary": os.path.exists(self.cfg.llama_bin)},
            "trees": trees,
            "staged": sorted(p.name for p in self.roots.stage_dir.iterdir()) if self.roots.stage_dir.is_dir() else [],
            "volume": {"path": str(self.roots.volume), "mounted": self.roots.volume.is_dir(),
                       "weights": sorted(p.name for p in self.roots.volume_weights.iterdir())[:200] if self.roots.volume_weights.is_dir() else []},
            "uptime_s": round(time.time() - self.started, 1),
            "jobs_done": self.jobs_done,
            "pod": os.environ.get("RUNPOD_POD_ID"),
            "endpoint": os.environ.get("RUNPOD_ENDPOINT_ID"),
        }

    def versions(self) -> dict[str, str | None]:
        """`h3` / `fastwan` have no --version: the image records their commits
        in /opt/studio/{h3,fastwan}.sha; fv-serve prints its build identity."""
        out: dict[str, str | None] = {}
        for name in ("h3", "fastwan"):
            sha = self.cfg.bin_dir.parent / f"{name}.sha"
            if sha.is_file():
                out[name] = f"{name} @ {sha.read_text().strip()[:12]}"
                continue
            path = self.cfg.bin_dir / name
            try:
                r = subprocess.run([str(path), "--help"], capture_output=True, text=True, timeout=30, env=self.env_for(name))
                first = (r.stdout or r.stderr).strip().splitlines()[0] if (r.stdout or r.stderr).strip() else ""
                out[name] = f"{name} ({first[:60]})" if r.returncode == 0 else f"{name}: exit {r.returncode}"
            except Exception as e:  # noqa: BLE001
                out[name] = f"{name}: unavailable ({type(e).__name__})"
        try:
            r = subprocess.run([self.cfg.fv_serve_bin, "--version"], capture_output=True, text=True, timeout=30, env=self.env_for("fastvideo"))
            txt = (r.stdout or r.stderr).strip()
            out["fv-serve"] = txt.splitlines()[0] if txt else f"exit {r.returncode}"
        except Exception as e:  # noqa: BLE001
            out["fv-serve"] = f"unavailable ({type(e).__name__})"
        return out

    def warm(self, inp: dict, progress: Progress | None) -> dict:
        engine = inp.get("engine")
        model = self.catalog.model(engine, str(inp.get("model", ""))) if engine in self.catalog.engines else None
        if model is None:
            raise JobError("validate", "warm needs a catalog engine and model")
        report = Reporter(progress, engine, model.id)
        t0 = time.time()
        report("staging", force=True)
        staged = self.weights.ensure_model(model.trees, model.stage, lambda d: report("staging", d))
        probe = {"engine": engine, "model": model.id, "prompt": "warm", "width": model.default_size[0],
                 "height": model.default_size[1], "frames": model.frames.default}
        job = self.catalog.validate(probe)
        ctx = self._ctx(job, self.cfg.jobs_dir / "warm", report)
        load_s = 0.0
        if engine == "h3" and not model.cli_only:
            from . import flags as fl
            from .sidecar import Sidecar, http_info_probe

            port = self.cfg.ports["h3"]
            key = fl.h3_serve_key(job, self.settings, self.catalog.knobs)
            _, load_s = self.sidecars.ensure("h3", key, lambda: Sidecar("h3", key, fl.h3_serve_argv(job, self.settings, self.catalog.knobs, port), ctx.env["h3"], http_info_probe(f"http://127.0.0.1:{port}/api/info")), model.vram_gib, lambda st, t: report("loading", f"{t:.0f} s"))
        elif engine == "fastwan" and not model.cli_only:
            from . import flags as fl
            from .sidecar import Sidecar, http_info_probe

            port = self.cfg.ports["fastwan"]
            key = fl.fastwan_serve_key(job, self.settings, self.catalog.knobs)
            _, load_s = self.sidecars.ensure("fastwan", key, lambda: Sidecar("fastwan", key, fl.fastwan_serve_argv(job, self.settings, self.catalog.knobs, port), ctx.env["fastwan"], http_info_probe(f"http://127.0.0.1:{port}/api/info")), model.vram_gib, lambda st, t: report("loading", f"{t:.0f} s"))
        elif engine == "fastvideo" and not self.cfg.fv_fake:
            from .sidecar import Sidecar, ping_probe

            port = self.cfg.ports["fastvideo"]
            _, load_s = self.sidecars.ensure("fastvideo", fastvideo.KEY, lambda: Sidecar("fastvideo", fastvideo.KEY, fastvideo.serve_argv(ctx), ctx.env["fastvideo"], ping_probe(f"http://127.0.0.1:{port}/ping")), model.vram_gib, lambda st, t: report("loading", f"{t:.0f} s"))
        report("done", force=True)
        return {"op": "warm", "engine": engine, "model": model.id, "stage_s": staged["stage_s"], "fetched": staged["fetched"],
                "staged": staged["staged"], "load_s": round(load_s, 1), "total_s": round(time.time() - t0, 1),
                "sidecars": self.sidecars.resident()}

    # ---- generate -------------------------------------------------------------------

    def generate(self, job_id: str, inp: dict, progress: Progress | None) -> dict:
        t_start = time.time()
        job = self.catalog.validate(inp)
        report = Reporter(progress, job.engine, job.model.id)
        report("queued", force=True)
        job_dir = self.cfg.jobs_dir / _safe(job_id)
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)
        job_dir.mkdir(parents=True, exist_ok=True)
        try:
            report("staging", force=True)
            staged = self.weights.ensure_model(job.model.trees, job.model.stage, lambda d: report("staging", d))
            original_prompt = job.prompt
            expanded = self.expand_prompt(job, job_dir, report)
            ctx = self._ctx(job, job_dir, report)
            res = DRIVERS[job.engine](ctx, job)
            res.timings.stage_s = staged["stage_s"]
            out = self._finish(job_id, job, res, job_dir, report, t_start, original_prompt=original_prompt, expanded_prompt=expanded)
            self.jobs_done += 1
            return out
        finally:
            if not self.cfg.keep_job_dirs:
                shutil.rmtree(job_dir, ignore_errors=True)

    def _ctx(self, job: Job, job_dir: Path, report: Reporter) -> Ctx:
        return Ctx(
            settings=self.settings, knobs=self.catalog.knobs, sidecars=self.sidecars, ports=self.cfg.ports,
            env={e: self.env_for(e) for e in ("h3", "fastwan", "fastvideo")}, job_dir=job_dir,
            store=self.store, bucket=self.bucket, log=self.log, report=report,
            fv_serve_bin=self.cfg.fv_serve_bin, fv_config=self.cfg.fv_config, fv_state_dir=self.cfg.fv_state_dir,
            fv_weights=str(self.roots.link_root), fv_fake=self.cfg.fv_fake, models=self.catalog.models,
        )

    def _finish(self, job_id: str, job: Job, res: RunResult, job_dir: Path, report: Reporter, t_start: float, *,
                original_prompt: str | None = None, expanded_prompt: str | None = None) -> dict:
        # ---- mux
        report("muxing", force=True)
        mp4 = job_dir / "video.mp4"
        audio_wav = res.audio_wav
        try:
            if res.video_mp4 is not None:
                shutil.move(str(res.video_mp4), mp4)
                if job.engine == "h3" and job.model.audio:
                    audio_wav = mux.extract_wav(mp4, job_dir / "audio.wav")
            else:
                assert res.video_y4m is not None
                r = mux.mux(res.video_y4m, audio_wav, mp4)
                res.timings.mux_s = r.seconds
            thumb = mux.thumbnail(mp4, job_dir / "thumb.jpg", res.frames, res.fps)
        except RuntimeError as e:
            raise JobError("mux", str(e)) from None
        # ---- upload
        report("uploading", force=True)
        urls: dict[str, str | None] = {"video_url": None, "audio_url": None, "thumbnail_url": None}
        expires = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=self.cfg.url_ttl_s)
        total_s = time.time() - t_start
        res.timings.total_s = total_s
        g = gpu.query()
        output = {
            "engine": job.engine, "model": job.model.id, "task": job.task, "seed": res.seed,
            "width": res.width, "height": res.height, "frames": res.frames, "fps": res.fps,
            "audio": bool(audio_wav and audio_wav.is_file()) if job.engine == "h3" else bool(res.extra.get("audio") or (audio_wav is not None)),
            "upscale": {"factor": res.upscaled[0], "width": res.upscaled[1], "height": res.upscaled[2]} if res.upscaled else None,
            "timings": res.timings.to_json(),
            "command": res.command,
            "worker": {"gpu": g.name if g else None, "driver": g.driver if g else None,
                       "image": self.cfg.image or os.environ.get("RUNPOD_IMAGE", "")},
            "expires_at": expires.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "expanded_prompt": expanded_prompt,
            "client": job.client,
        }
        if self.store is None or not self.bucket:
            raise JobError("upload", "no object store configured (FV_R2_* / STUDIO_R2_LOCAL_DIR)")
        prefix = f"jobs/{_safe(job_id)}/"
        try:
            self.store.put_file(self.bucket, prefix + "video.mp4", mp4, "video/mp4")
            urls["video_url"] = self.store.presign_get(self.bucket, prefix + "video.mp4", self.cfg.url_ttl_s)
            if job.engine == "h3" and audio_wav and audio_wav.is_file():
                self.store.put_file(self.bucket, prefix + "audio.wav", audio_wav, "audio/wav")
                urls["audio_url"] = self.store.presign_get(self.bucket, prefix + "audio.wav", self.cfg.url_ttl_s)
            self.store.put_file(self.bucket, prefix + "thumb.jpg", thumb, "image/jpeg")
            urls["thumbnail_url"] = self.store.presign_get(self.bucket, prefix + "thumb.jpg", self.cfg.url_ttl_s)
            output.update(urls)
            meta = job_dir / "job.json"
            meta.write_text(json.dumps({"id": job_id, "input": _redact_input(job, original_prompt), "output": output}, indent=1))
            self.store.put_file(self.bucket, prefix + "job.json", meta, "application/json")
        except JobError:
            raise
        except Exception as e:  # noqa: BLE001
            raise JobError("upload", f"{type(e).__name__}: {e}") from None
        report("done", force=True)
        return output


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:120] or "job"


def _redact_input(job: Job, original_prompt: str | None = None) -> dict:
    """The validated request without inline media bytes (the prompt as sent,
    before any expansion)."""
    inputs = {}
    for k, v in job.inputs.items():
        if isinstance(v, list):
            inputs[k] = [{"kind": m.kind, "value": m.value if m.kind != "b64" else f"<{m.inline_bytes} bytes>"} for m in v]
        else:
            inputs[k] = {"kind": v.kind, "value": v.value if v.kind != "b64" else f"<{v.inline_bytes} bytes>"}
    return {"engine": job.engine, "model": job.model.id, "task": job.task, "prompt": original_prompt if original_prompt is not None else job.prompt,
            "negative_prompt": job.negative_prompt, "width": job.width, "height": job.height, "frames": job.frames,
            "seed": job.seed, "options": job.explicit, "upscale": job.upscale, "inputs": inputs, "client": job.client}
