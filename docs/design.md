# fastvideo-studio: design

One RunPod serverless endpoint that runs jobs for three engines, and one
web app ("Studio") that drives it. State and invariants; numbers are the
engines' own (`h3-rs/CLAUDE.md`, `fastwan-rs/CLAUDE.md`, fastvideo-rs
`docs/serve/e2e/serverless.md`).

## 1. Pieces

```
browser ── Studio page (static) ── Studio Worker (Cloudflare, holds the RunPod key)
                                         │  /run /status /cancel          │ R2 (uploads, outputs)
                                         ▼                                ▼
                               RunPod queue endpoint ── worker container ── /runpod-volume (EU weights)
                                                          handler.py
                                                   ┌────────┼─────────┐
                                                h3 serve  fastwan   fv-serve   (resident sidecars,
                                                (h3-rs)   serve     (fastvideo-rs)  one per engine)
                                                h3 generate / fastwan generate   (per-job CLI for
                                                                                  tasks serve lacks)
```

- **Worker** (`worker/`): one image, `ghcr.io/zaitrarrio/fastvideo-studio-worker`,
  built by CI from pinned commits of the three repos on top of
  `ghcr.io/zaitrarrio/fastvideo-rs-serve:cuda-stable` (fv-serve, fv-boot,
  CUDA 13.4 libs). The `h3` and `fastwan` binaries are built with
  `--features cuda` (cudarc, `cuda-12080`, dynamic loading) and get their own
  CUDA 12.8 cuBLAS / cuBLASLt / NVRTC from the `nvidia-*-cu12` wheels under
  `/opt/cuda12/lib`, selected per sidecar with `LD_LIBRARY_PATH`; fv-serve
  keeps the image's 13.4 libraries. Endpoint `minCudaVersion` is 13.0
  (fv-serve's floor); a 13-capable driver runs the 12.8 runtime too.
- **Queue, not load-balancer**: jobs run 10 s to 8 min; load-balancing
  endpoints cap a request at 5.5 min and drop on overload.
- **`handler.py`** owns the RunPod job loop (`fvstudio/rp.py`: the SDK's
  protocol over `urllib`, no SDK, concurrency 1)
  and dispatches on `input.engine`. Progress goes out with
  `progress_update`; the result is JSON with presigned R2 URLs (the `/run`
  output cap is 10 MB; an MP4 is not).
- **Sidecars**: `h3 serve` and `fastwan serve` are kept resident, keyed by
  `(engine, model, recipe)`; a job for another model restarts that engine's
  sidecar (h3: ~2 min from staged weights). `fv-serve` runs once with
  `engine.swap = true` and every variant `resident = false`, bound to
  127.0.0.1 with `auth.mode = none`, started directly (not through fv-boot:
  the handler does the volume link and weights itself). Before starting a
  sidecar the handler checks free VRAM (`nvidia-smi`) against the catalog's
  `vram_gib` and stops the other engines' sidecars when it does not fit.
- **Tasks the sidecars cannot take** go through the CLI per job, at the
  cost of a model load: `h3 generate` for FL2VA / Ref2VA / `--continue`
  (until `h3 serve` accepts image inputs: `h3-rs` branch
  `claude/wizardly-clarke-1a2jta`), `h3 regen2k`, `fastwan generate --causal`
  for long videos. The handler picks serve when `/api/info` advertises the
  task, else the CLI, so the fallback disappears as the engines grow.
- **Outputs**: y4m + wav (h3, fastwan) are muxed with ffmpeg to H.264 + AAC
  MP4; fv-serve's MP4 is downloaded from its local `/files` URL. The
  handler uploads `video.mp4`, `audio.wav` (h3), `thumb.jpg` and
  `job.json` to R2 under `jobs/<id>/` and returns presigned GETs (24 h).
- **Weights**: the EU network volume `bz0ofkfx45` (`fv-weights-h3-ltx-500`,
  EU-RO-1, mounted at `/runpod-volume`) holds the H3 and LTX trees and the
  h3-rs extras (`/runpod-volume/weights/<tree>`, `/runpod-volume/h3-rs/…`).
  It is a FUSE filesystem the mmap loaders page at 0.2 GB/s, so trees are
  **staged** to the container disk (`/workspace/stage/<tree>`, 8 parallel
  `cp`, size-verified, same rule as `h3-rs/scripts/runpod/make_boot.py`)
  before a sidecar starts; what is not on the volume (FastWan, Wan 2.2,
  Self-Forcing) is fetched from R2 `fv-weights` / `fastwan-public` to
  `/workspace/stage/` on first use. The volume is add-only and the worker
  never writes it. A root of symlinks `/workspace/w/<tree>` points at
  wherever each tree lives; recipes are written against `/workspace/w`.
- **Studio Worker** (`studio/`): a Cloudflare Worker with static assets. It
  holds `RUNPOD_API_KEY` / `RUNPOD_ENDPOINT_ID`, exposes
  `/api/catalog`, `POST /api/jobs`, `GET /api/jobs/:id`,
  `POST /api/jobs/:id/cancel`, `POST /api/uploads` (R2 binding), and
  `GET /api/health` (the endpoint's `/health`). Access is gated by a shared
  secret (`STUDIO_TOKEN`, entered once, kept in localStorage): the endpoint
  costs money per job.
- **Studio page** (`studio/web/`): vanilla HTML / CSS / ES modules in the
  Prompt Studio style (same tokens, fonts, panels, stages grid, session
  history). The form is generated from `catalog/catalog.json`.

## 2. Catalog is the contract

`catalog/catalog.json` is the single source of truth: engines, families,
tasks, models, each model's sizes / frame grid / tasks / knobs / recipe /
weight trees / VRAM. The worker validates a job against it and turns
`options` into CLI flags or fv-serve fields; the page builds its controls
from it. A knob the catalog does not list is refused, never defaulted.
`worker/tests` and `studio/tests` both load the same file.

## 3. Studio UI rules

- One screen. Left: the form. Right: the clip, the stages, the session.
- Pick **model** first (cards grouped by family, one recommended per
  family), then a **task** chip the model supports, then the prompt and the
  task's inputs (drop zones for images / audio / video / draft), then
  size, length, seed. That is the whole basic form.
- **Advanced** is one disclosure per model, listing only that model's
  knobs at the recipe defaults, with "Reset to recipe". A model's speed
  recipe is on by default (h3: stored MXFP8, MXFP8 decoder, INT8 Q·Kᵀ,
  FP8 P·V, tables), since that is what the measurements use.
- The equivalent command line (`h3 generate …`) is shown under the form
  so a run is reproducible on a rented box.
- Progress: stages from the worker's `progress_update`; the clip plays
  from the MP4 URL; the session keeps the last 24 jobs in localStorage.

## 4. What is not done here

- No GPU in this session: the handler is tested against the engines'
  `--synthetic` serve modes and a fake fv-serve on CPU, and against a fake
  RunPod loop. First real run: `python3 scripts/runpod/endpoint.py smoke`.
- The image is built by `.github/workflows/worker-image.yml`; building
  needs read access to the three repos.
- Pricing / quotas / multi-user auth: out of scope.
