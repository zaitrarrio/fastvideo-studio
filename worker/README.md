# fastvideo-studio worker

The RunPod serverless worker of `docs/design.md`: one container, three
engines (`h3 serve` / `fastwan serve` / `fv-serve` as resident sidecars, the
engines' CLIs per job for the tasks serve lacks), driven by `handler.py`
through the `runpod` SDK. The contract is `docs/jobs.md`; the catalog
(`catalog/catalog.json`) decides what a job may ask for.

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
     jobs, bound to 127.0.0.1:8000) through `POST /fv/v1/jobs`; media as
     https URLs, presigned R2 GETs or data URIs; `extend` / `retake` derive
     `extend_s` / `start_s..end_s` from the requested frames.
   Before a sidecar starts, free VRAM (`nvidia-smi`) is checked against
   the model's `vram_gib` and the other engines' sidecars are stopped when
   it does not fit. h3 and fastwan run with `LD_LIBRARY_PATH=/opt/cuda12/lib`
   (the 12.8 cuBLAS / cuBLASLt / NVRTC from the `nvidia-*-cu12` wheels);
   fv-serve keeps the image's CUDA 13.4.
4. **mux** (`fvstudio/mux.py`): the frame stream (16-byte LE u32 header +
   RGB8) → y4m through ffmpeg's rawvideo input, y4m + wav → H.264 / AAC
   yuv420p MP4 with `+faststart`; fv-serve's own MP4 is downloaded from its
   `/files` URL; a JPEG thumbnail from the middle frame.
5. **upload**: `jobs/<id>/{video.mp4, audio.wav (h3), thumb.jpg, job.json}`
   to the Studio bucket, presigned GETs (24 h) in the output.

Progress goes out with `runpod.serverless.progress_update` as
`{stage, detail, step, total_steps, frames, total_frames, elapsed_s, engine,
model}` with the jobs.md stage ids. A sidecar that dies restarts on the next
job; a job whose sidecar dies fails with `where: "engine"` and the sidecar's
last 50 log lines in `log_tail`.

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
| `STUDIO_CUDA12_LIB` | `/opt/cuda12/lib` |
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
cd worker && python3 -m pytest -q tests                  # 70 unit tests, no engines (pytest.ini skips e2e)
STUDIO_E2E=1 python3 -m pytest -q tests/test_e2e.py -s   # builds both, runs the handler end to end
```

The e2e test runs `h3 serve --synthetic --backend cpu` (a 480×272×124 clip
in ~32 s on 4 cores: 8 steps of ~2.5 s, 11 s of decode), `fastwan serve
--synthetic --backend cpu` (480×272×17 in ~15 s, 9.5 s of it the synthetic
×2 upscaler), a fake fv-serve
(`tests/fake_fv_serve.py`), a fake `runpod` module and a directory as R2,
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

- stage 1, `rust:1.97-bookworm`: `h3` and `fastwan` from the pinned commits
  (`H3_REF`, `FASTWAN_REF`; branch `claude/wizardly-clarke-1a2jta` at the
  time of writing) with `--features cuda` (cudarc, dynamic loading: no CUDA
  toolkit in the build; h3 without its default `wgpu` feature);
- stage 2, `ghcr.io/zaitrarrio/fastvideo-rs-serve:cuda-stable` (fv-serve,
  fv-boot, CUDA 13.4 libraries, a static ffmpeg with libx264 / aac / nvenc,
  Ubuntu 22.04, python3.10): `python3-pip` / `-venv`, a venv at
  `/opt/studio/venv` with `runpod`, `boto3`, `nvidia-cublas-cu12==12.8.*`,
  `nvidia-cuda-nvrtc-cu12==12.8.*` (their `.so`s linked into
  `/opt/cuda12/lib` as `libcublas.so.12`, `libcublasLt.so.12`,
  `libnvrtc.so.12`, `libnvrtc-builtins…`), the two binaries at
  `/opt/studio/bin`, the catalog, `configs/fastvideo.toml` and this package;
  entrypoint `python /opt/studio/handler.py` (not fv-boot: the handler links
  the volume and stages weights itself).

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

`scripts/runpod/endpoint.py` (REST v1 with `RUNPOD_API_KEY`): `template`,
`up`, `down`, `status`, `health`, `run '<json>'`, `cancel <id>`, `smoke`
(op info + one 480p draft each on h3, fastwan and fastvideo, timed),
`backstop <minutes>` (a detached timer that runs `down`), `plan` (prints
both bodies without calling). Money guards as in fastvideo-rs: workers 0 / 1,
idle timeout 5 s, FlashBoot off, execution timeout 30 min, the backstop.
Secrets are referenced as `{{ RUNPOD_SECRET_fv_r2_* }}` in the template env.
Size `STUDIO_CONTAINER_DISK_GB` for the staged trees (the V2 recipe stages
~103 GB; h3-base's `transformer` ~100 GB more for the base / plug / turbo
recipes).

## Catalog notes

The catalog is the contract and is not edited from here, apart from the
path corrections the engines' sources and run logs required (listed in the
handover report): the FastWan TAE lives in `auxiliary/tae`, `h3 --adapter`
takes a file, `h3 regen2k` needs `--adaln-table`, the Trim NVFP4 file's
name, and `h3-base` has `transformer`, not `transformer_prequant_mxfp8`.
`fv.flow_shift` has no field in fv-serve's native jobs API and is refused
when set. The R2 `fastwan21-1.3b` tree sits under a revision directory
(`weights/<dest>/<rev>/`), which the resolver finds through its manifest.
