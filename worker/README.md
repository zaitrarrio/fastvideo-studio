# fastvideo-studio worker

The RunPod serverless worker of `docs/design.md`: one container, three
engines (`h3 serve` / `fastwan serve` / `fv-serve` as resident sidecars, the
engines' CLIs per job for the tasks serve lacks), driven by `handler.py`.
The contract is `docs/jobs.md`; the catalog (`catalog/catalog.json`)
decides what a job may ask for.

**Stdlib only.** The worker has no third-party dependency: `fvstudio/rp.py`
speaks RunPod's serverless protocol over `urllib` (what runpod-python's
`rp_job` / `rp_http` / `rp_ping` do, and what fastvideo-rs's
`fastvideo-deploy` and h3-rs's `h3 worker` do in Rust) and `fvstudio/r2.py`
signs R2's S3 API itself (SigV4 with `hmac` / `hashlib`, byte-identical to
botocore's signer in `tests/test_r2.py`). No venv, no `pip install` in the
image; `requirements.txt` lists test tools only.

**Not yet run on a GPU.** Everything here was exercised on a CPU box against
the engines' `--synthetic` serve modes, a fake fv-serve and a local R2
(`tests/test_e2e.py`). The first real run is
`python3 scripts/runpod/endpoint.py smoke`; nothing below claims a speed.

## How a job runs

1. **validate** (`fvstudio/catalog.py`): engine, model (or alias), task,
   prompt (≤ 2000 chars), size (one of the model's), frames (on its grid:
   `17n+5`, `4n+1`, `12n+9`, `8n+1`), seed, media refs (`url` https only,
   `b64` ≤ 6 MiB, `r2` under `uploads/`), knobs by type / range / `when`
   (`auto` attention keeps the VSA knobs: the preset decides), top-level
   `upscale` against the model's `*.upscale_factor` options (the knob itself
   is the page's control and is ignored in `options`). A knob the model does
   not list is refused → `{"error", "where": "validate"}`.
2. **stage** (`fvstudio/weights.py`): every catalog tree present under
   `/workspace/w` (volume trees linked; `weights/<tree>` missing from the
   volume fetched from the R2 store `fv-weights` by its manifest, size
   checked; `r2:` / `https:` entries fetched once to `/workspace/stage`), then
   the model's `stage` entries copied onto the container disk with h3-rs's
   `stage TREE [SUB…]` rule (8 parallel `cp --update=none`, sizes verified,
   the rest of a partially staged tree symlinked back) and the link
   repointed. The volume is never written.
3. **run** (`fvstudio/engines/`): the engine's driver.
   - h3: `h3 serve` keyed by its argv (another model or recipe restarts it);
     `t2v` through serve; `fl2v` / `ref2v` / `extend` through serve when
     `/api/info.tasks` lists them (the parallel h3-rs change), else
     `h3 generate` per job; `regen2k` and `cli_only` models always the CLI.
     Ref2VA on a `transformer` variant (base, plug4, pika) and the Turbo
     Ref2V LoRAs get `--transformer /workspace/w/h3-ref2va/transformer_ref`
     and only a `tables_ref`.
   - fastwan: `fastwan serve` for clips, `fastwan generate --causal` for
     `long`.
   - fastvideo: one `fv-serve --config configs/fastvideo.toml` (swap = true,
     every variant `resident = false`, auth none, local artifacts, memory
     jobs, bound to 127.0.0.1:8010, the worker's fastvideo port) through `POST /fv/v1/jobs`; media as
     https URLs, presigned R2 GETs or data URIs; `extend` / `retake` derive
     `extend_s` / `start_s..end_s` from the requested frames.
   Before a sidecar starts, free VRAM (`nvidia-smi`) is checked against
   the model's `vram_gib` and the other engines' sidecars are stopped when
   it does not fit. h3 and fastwan are built with `--features cuda-13` and
   load the image's CUDA 13.4 cuBLAS / cuBLASLt / NVRTC like fv-serve;
   fv-serve keeps the image's CUDA 13.4.
4. **mux** (`fvstudio/mux.py`): the frame stream (16-byte LE u32 header +
   RGB8) → y4m through ffmpeg's rawvideo input, y4m + wav → H.264 / AAC
   yuv420p MP4 with `+faststart`; fv-serve's own MP4 is downloaded from its
   `/files` URL; a JPEG thumbnail from the middle frame.
5. **upload**: `jobs/<id>/{video.mp4, audio.wav (h3), thumb.jpg, job.json}`
   to the Studio bucket, presigned GETs (24 h) in the output.

Progress goes out as RunPod's `{"status": "IN_PROGRESS", "output": …}`
(`rp.RunpodClient.progress_update`, from a sender thread so the job never
waits on the platform) with the object `{stage, detail, step, total_steps,
frames, total_frames, elapsed_s, engine, model}` verbatim, the jobs.md
stage ids. A sidecar that dies restarts on the next
job; a job whose sidecar dies fails with `where: "engine"` and the sidecar's
last 50 log lines in `log_tail`.

## The RunPod loop (`fvstudio/rp.py`)

```text
take      GET  $RUNPOD_WEBHOOK_GET_JOB      ($ID → worker id)  &job_in_progress=0|1
            long poll (90 s); 204 / 400: no job; 429: wait 5 s; errors: 1 → 30 s back-off
            200: {"id", "input"} or a list of them
progress  POST $RUNPOD_WEBHOOK_POST_OUTPUT  ($RUNPOD_POD_ID → worker id, $ID → job id) &isStream=false
            {"status": "IN_PROGRESS", "output": <progress object>}       one attempt
done      POST same URL                     {"output": …} and / or {"error": …}
            3 attempts, back-off 1, 1, 2 s; no `status` field: the platform
            marks the job COMPLETED / FAILED by the presence of `error`
ping      GET  $RUNPOD_WEBHOOK_PING         ($RUNPOD_POD_ID → worker id) ?job_id=<running>&runpod_version=…
            every $RUNPOD_PING_INTERVAL ms (10000)
headers   Authorization: $RUNPOD_AI_API_KEY (raw); on POSTs X-Request-ID: <job id>,
          Content-Type: application/x-www-form-urlencoded (the body is JSON)
```

A handler result with an `error` key is FAILED with that error lifted out
beside `output` (as the SDK's `run_job` does), so `output.where` and
`log_tail` survive for the Studio and `endpoint.py`; an exception becomes
the SDK's JSON report (`error_type`, `error_message`, `error_traceback`,
`hostname`, `worker_id`, `runpod_version`). Concurrency is 1. SIGTERM /
SIGINT stop the loop after the current job's result is posted. Without
`RUNPOD_WEBHOOK_GET_JOB` the handler runs in local mode:
`python3 handler.py --test-input job.json` runs that one job
(`{"input": …}`) and prints the job-done body (exit 1 when it failed).
`tests/test_rp.py` drives the loop against a fake RunPod.

`op: info` returns `gpu, driver, image, catalog_version, versions (h3,
fastwan, fv-serve --version), sidecars, trees (volume / staged / linked per
catalog tree), staged, volume`. `op: warm` stages a model's trees and starts
its sidecar.

## Environment

| Variable | Meaning (default) |
|---|---|
| `FV_R2_ENDPOINT`, `FV_R2_ACCESS_KEY_ID`, `FV_R2_SECRET_ACCESS_KEY`, `FV_R2_BUCKET` | R2: the Studio bucket (outputs, `uploads/`) and the `fv-weights` store, from the `fv_r2_*` RunPod secrets (`STUDIO_R2_*` override) |
| `STUDIO_R2_LOCAL_DIR` | a directory instead of R2 (tests / local runs) |
| `STUDIO_CATALOG` | `/opt/studio/catalog/catalog.json` |
| `STUDIO_BIN_DIR` | `/opt/studio/bin` (`h3`, `fastwan`) |
| `STUDIO_FV_SERVE`, `STUDIO_FV_CONFIG`, `STUDIO_FV_STATE_DIR` | `/opt/fastvideo-rs/bin/fv-serve`, `/opt/studio/configs/fastvideo.toml`, `/workspace/fv-state` |
| `STUDIO_VOLUME`, `STUDIO_VOLUME_WEIGHTS`, `STUDIO_VOLUME_EXTRAS`, `STUDIO_STAGE_DIR`, `STUDIO_LINK_ROOT` | the catalog's `trees` roots (`/runpod-volume`, `…/weights`, `…/h3-rs`, `/workspace/stage`, `/workspace/w`) |
| `STUDIO_JOBS_DIR` | `/workspace/jobs` (a job's scratch, removed after upload; `STUDIO_KEEP_JOB_DIRS=1` keeps it) |
| `STUDIO_H3_PORT`, `STUDIO_FASTWAN_PORT`, `STUDIO_FV_PORT` | 7861, 7862, 8000 |
| `STUDIO_CUDA12_LIB` | empty (set to a directory to put a separate CUDA runtime ahead of the image's for h3 / fastwan) |
| `STUDIO_BACKEND`, `STUDIO_SYNTHETIC`, `STUDIO_FV_FAKE` | `cuda`; `1` starts the serves with `--synthetic`; `1` assumes an fv-serve already listens (tests) |
| `STUDIO_ESRGAN` | Real-ESRGAN weights for `upscale` (`/workspace/w/h3-rs/assets/realesr-animevideov3.safetensors`; upscale jobs are refused while the file is missing) |
| `STUDIO_TAEH3` | `taeh3.safetensors` for `preview` (`/workspace/w/auxiliary/tae/taeh3.safetensors`) |
| `STUDIO_IMAGE` | the image tag reported in `worker.image` |
| `STUDIO_URL_TTL_S`, `STUDIO_LOAD_TIMEOUT_S` | 86400, 1800 |
| `STUDIO_STAGE_DRY_RUN` | `1`: link instead of copying (tests) |

## Running locally against the synthetic engines

```sh
cargo build --release -p h3-cli        # in h3-rs    (default features: wgpu; no GPU needed for --synthetic)
cargo build --release -p fastwan-cli   # in fastwan-rs
cd worker && python3 -m pytest -q tests                  # 124 unit tests, no engines (pytest.ini skips e2e)
STUDIO_E2E=1 python3 -m pytest -q tests/test_e2e.py -s   # builds both, runs the handler end to end
```

The e2e test runs `h3 serve --synthetic --backend cpu` (a 480×272×124 clip
in ~32 s on 4 cores: 8 steps of ~2.5 s, 11 s of decode), `fastwan serve
--synthetic --backend cpu` (480×272×17 in ~15 s, 9.5 s of it the synthetic
×2 upscaler), a fake fv-serve
(`tests/fake_fv_serve.py`), a recording `progress_update` and a directory as R2,
and checks the MP4s with ffprobe (the mux assertions are skipped without
ffmpeg). A hand run without pytest:

```sh
STUDIO_BIN_DIR=$PWD/bin STUDIO_BACKEND=cpu STUDIO_SYNTHETIC=1 STUDIO_FV_FAKE=1 \
STUDIO_R2_LOCAL_DIR=/tmp/r2 STUDIO_STAGE_DRY_RUN=1 STUDIO_VOLUME=/tmp/vol \
python3 -c 'import handler; h = handler.make_handler(progress_update=lambda j, d: print(d)); \
print(h({"id": "x", "input": {"engine": "h3", "model": "fasth3-v2", "prompt": "a fox", "width": 480, "height": 272}}))'
```

## The image

`worker/Dockerfile` (built by `.github/workflows/worker-image.yml` into
`ghcr.io/zaitrarrio/fastvideo-studio-worker:{sha-<sha>,latest}`):

- stage 1 (`engines`), `rust:1.97-bullseye`: `h3` and `fastwan` from the
  pinned commits (`H3_REF`, `FASTWAN_REF`; branch
  `claude/wizardly-clarke-1a2jta` at the time of writing) with
  `--features cuda-13` (cudarc binds CUDA 13 and loads the libraries at run
  time: no CUDA toolkit in the build, no second 12.8 runtime, no wheels; h3
  without its default `wgpu` feature);
- stage 2 (`fv`), the fastvideo-rs serve image, a file source only:
  fv-serve, llama-server, its CUDA 13.4 cuBLAS / cuBLASLt / NVRTC / cudart,
  cuDNN 9 and its ffmpeg / ffprobe;
- stage 3, `ubuntu:22.04` + `python3` (the handler and `fvstudio` are
  stdlib-only: no venv, no pip, nothing installed) + the four codec
  libraries + the files above, the two binaries at `/opt/studio/bin`, the
  catalog, `configs/fastvideo.toml` and this package; entrypoint
  `python3 -I /opt/studio/handler.py` (not fv-boot: the handler links the
  volume and stages weights itself).

`FV_SERVE_IMAGE` is a build arg: until fastvideo-rs promotes a release,
`:cuda-stable` may not exist (`docs/serve/images.md`) and `:cuda-latest` is
the fallback. `REPO_READ_TOKEN` (workflow secret, Docker secret `repo_token`)
clones private repositories; it never lands in a layer.

## GPU / driver requirements

- An RTX PRO 6000 class card (the catalog's `vram_gib` is 40–48 GB for the
  h3 recipes, 80 GB for the fastvideo H3 tiers, 8 GB for FastWan; the
  endpoint script asks for `NVIDIA RTX PRO 6000 Blackwell Server Edition`
  in EU-RO-1 beside the weights volume `bz0ofkfx45`). The measured recipe
  numbers in `h3-rs/CLAUDE.md` are for the 600 W Server Edition; a
  Workstation / Max-Q card runs slower and is not comparable.
- A driver with CUDA 13 (≥ 580): fv-serve's floor (`minCudaVersion`
  13.0 on the endpoint). h3 and fastwan use the 12.8 runtime libraries,
  which a 13-capable driver runs too.
- `nvidia-smi` is injected by the container runtime (`utility`
  capability) and is what the VRAM check and `op: info` read.

## RunPod

`scripts/runpod/endpoint.py` (REST v2, `api.runpod.io/v2`, with
`RUNPOD_API_KEY`): `gpus` (the catalog: id, pool, memory, serverless $/h),
`plan` (prints the create body without calling), `up` (checks the pool
and the name, creates, writes `.runpod-endpoint.json`), `list`, `status`,
`health`, `down --yes`, `run '<json>'`, `cancel <id>`, `smoke` (op info +
one 480p draft each on h3, fastwan and fastvideo, timed), `backstop
<minutes>` (a detached timer that runs `down --yes`); `template` is a
no-op (v2 endpoints take the image directly). GPUs are chosen by pool
(`BLACKWELL_96` minus the Max-Q edition by default) and a private ghcr
image needs a registry credential (`--registry-name 'fv-ghcr-*'`). Money
guards as in fastvideo-rs: workers 0 / 1 (`--i-know` for more), idle
timeout 5 s, FlashBoot off, request timeout 30 min, the backstop. Secrets
are referenced as `{{ RUNPOD_SECRET_fv_r2_* }}` in the endpoint env. Size
`STUDIO_CONTAINER_DISK_GB` / `--disk` for the staged trees (the V2 recipe
stages ~103 GB; h3-base's `transformer` ~100 GB more for the base / plug /
turbo recipes). `cd worker && python3 -m pytest -q tests/test_endpoint_script.py`
checks the bodies and the guards against fake v2 responses.

## Prompt expansion

Every model lists the `expand.prompt` knob (the page sends it as
`options.prompt`: `off` by default, `balanced` or `quality`). When it is
on, the worker rewrites the job's prompt before the engine runs, the way
fastvideo-rs's batch rewrites do (`fvstudio/upsampler.py`, a port of
`crates/fastvideo-upsampler/src/rewrite.rs` and of
`deploy/runpod/fv-upsampler.sh`):

- **What runs.** llama.cpp's `llama-server` (the fastvideo-rs serve
  image's binary, `/opt/fastvideo-rs/bin/llama-server`) as a fourth
  sidecar (`upsampler`) on `127.0.0.1:18080`, serving Qwen3-VL-4B-Instruct
  in Q8_0 with its F16 vision projector: the same flags as fastvideo-rs's
  launcher (`-ngl 99`, flash attention, `q8_0` KV, cached prefix reuse,
  `--fit off`, no web UI) with a 16 384-token context and one slot (the
  launcher's 65 536 / 2 are for the director's long sessions; the Studio
  rewrites one prompt at a time). Readiness is `/health` 200 and one
  1-token completion, like the launcher.
- **Memory.** Weights 3.99 GiB + projector 0.78 GiB + KV 1.20 GiB (q8_0 at
  16 384 tokens: 2 × 36 layers × 8 heads × 128 × 1.0625 B per token) +
  ~0.5 GiB of compute buffers = **6.5 GiB** (`upsampler.VRAM_GIB`), which
  `SidecarManager.ensure` reserves beside the engine; at the launcher's
  65 536 tokens the KV alone is 4.8 GiB. The sidecar stays resident across
  jobs and engines until a model that does not fit evicts it.
- **First use.** The GGUF tree `qwen3-vl-4b-instruct-gguf` (4.28 GB model
  + 0.84 GB projector) is not on the volume: `WeightsResolver.ensure_tree`
  fetches it from the R2 weights store (`upsampler.TREE`) into the stage
  dir on the first expanded job, ~5 GB once per container.
- **The rewrite.** One `POST /v1/chat/completions` (temperature 0.6, the
  family's token cap): a system prompt per family, ported verbatim from
  fastvideo-rs: H3 (`h3`, `fv-h3`: the Context-IR layout with its three
  labelled sections, `[Shot N]` blocks in quality mode), Wan (`fastwan`,
  `wan`: silent model, motion first), LTX (`ltx`: a cinematographer's
  paragraph ending in the soundscape); the job's prompt as the user text.
  Balanced / quality set the target length (60-110 / 120-220 words for
  H3, 60-100 / 100-160 for Wan, 80-130 / 140-200 for LTX). The answer is
  cleaned (think blocks, code fences, a `Prompt:` label and wrapping
  quotes removed, whitespace collapsed; H3 keeps its section lines) and
  cut to the family's text-encoder budget in characters (Wan 1800, LTX
  3600, H3 3200, the same budgets fastvideo-rs uses without a tokenizer).
- **Images and the negative prompt.** A first-frame image goes to the LLM
  as an `image_url` data part (the projector is loaded) with a system line
  to keep its look and describe the motion from it; without the projector
  the text-only wording of fastvideo-rs is used (an unseen image sets the
  first frame). The negative prompt is never rewritten; the LLM is only
  told what it excludes.
- **Fallback.** Any failure (the fetch, the sidecar not coming up, an HTTP
  error, a timeout, an empty answer) leaves the original prompt and logs
  `prompt expansion failed (<why>); generating with the original prompt`
  (`Upsampler.last_error`); the job never fails because of the LLM. The
  output carries the rewrite as `expanded_prompt` (null when off or
  fallen back) and `info` lists the sidecar with the others.

## Catalog notes

The catalog is the contract and is not edited from here, apart from the
path corrections the engines' sources and run logs required (listed in the
handover report): the FastWan TAE lives in `auxiliary/tae`, `h3 --adapter`
takes a file, `h3 regen2k` needs `--adaln-table`, the Trim NVFP4 file's
name, and `h3-base` has `transformer`, not `transformer_prequant_mxfp8`.
`fv.flow_shift` has no field in fv-serve's native jobs API and is refused
when set. The R2 `fastwan21-1.3b` tree sits under a revision directory
(`weights/<dest>/<rev>/`), which the resolver finds through its manifest.
