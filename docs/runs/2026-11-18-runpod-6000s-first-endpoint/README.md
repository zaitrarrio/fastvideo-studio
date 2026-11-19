# 2026-11-18: first jobs on the studio endpoint (RunPod, RTX PRO 6000 Server Edition)

Endpoint `wusgagr2t5am6d` (`studio`, pool BLACKWELL_96 with the Max-Q edition
excluded, EU-RO-1, volume `bz0ofkfx45`, workers 0/1, idle 5 s, FlashBoot off,
container disk 300 GB), image `ghcr.io/zaitrarrio/fastvideo-studio-worker:sha-313743e`
(h3 @ e029781, fastwan @ 74c27bb, fv-serve 0.1.87 from `fastvideo-rs-serve:cuda-latest`
digest aa2a4657…). Worker: NVIDIA RTX PRO 6000 Blackwell Server Edition, driver
595.112.07, compute 12.0, 95.6 GB. Created with `scripts/runpod/endpoint.py up`
(REST v2: `rest.runpod.io` no longer resolves).

## `op: info` (job 8aca3baa…)

Queued at 20:28 behind a throttled worker ("the host's GPUs are in use by other
workloads"); ran once a card freed up. Volume mounted at `/runpod-volume` with
the H3 / LTX trees, `h3-rs/{assets,plug4,tables}` and `upscaler`; `h3-rs/cfg`
absent (the catalog's `verify` note stands); no FastWan / Wan 2.2 trees on the
volume (R2 on first use, as designed). No sidecar resident on a fresh worker.

## `h3-plug4` T2V 832×480×124, seed 7 (job 242cbd5a…)

| stage | time |
|---|---|
| queue → worker start (cold, image pull) | 157 s (`delayTime`) |
| staging h3-base + h3-ref2va + plug4 to the container disk | 52 s |
| `h3 serve` load (MXFP8 base, plug4 adapter branch, tables) | ~100 s |
| prompt encode (Qwen3-VL) | ~60 s |
| denoise, 4 steps | ~8 s (≈2 s per forward) |
| decode (MXFP8 decoder) | ~5 s |
| execution total | 234 s |

The denoise matches h3-rs's 480p draft number (8.6 s after load on the 600 W
Server Edition). The MP4 was muxed; the job then failed at `uploading`:
`Credential access key has length 43, should be 32`, i.e. the
`{{ RUNPOD_SECRET_studio_r2_access_key_id }}` reference was passed through
unsubstituted because the RunPod secrets `studio_r2_access_key_id` /
`studio_r2_secret_access_key` did not exist yet. Everything up to the upload
is verified on real hardware; the second job of an engine on a warm worker
would skip the staging and load.

Not measured yet: the warm second job, fastwan (R2 fetch of the tree), the
fastvideo-rs families, upscaling (`STUDIO_ESRGAN` still unset: the volume's
`upscaler/` file name was not visible to `op: info`).

## 2026-11-19 00:20 UTC: the smoke on image `sha-14dbbce` (R2 secrets in place)

`op: r2`: `write: ok` to `fastvideo-studio`, `read_weights: ok (58 manifests)` from `fv-weights`.

| job | result | stage | load | encode | denoise | decode | mux | total (execution) |
|---|---|---|---|---|---|---|---|---|
| `h3-plug4` T2V 832×480×124, seed 7 | COMPLETED, MP4 832×480 124 frames H.264 + AAC 32 kHz (2.1 MB) | 56.5 s | 109.7 s | 79.0 s | 7.8 s (4 steps) | 3.3 s | 0.7 s | 258.0 s |
| `fastwan-1.3b` T2V 832×480×81, seed 7 | COMPLETED, MP4 832×480 81 frames (1.1 MB) | 124.7 s (first R2 fetch of the FastWan tree) | 8.5 s | 1.4 s | 1.6 s (3 steps) | 0.3 s | 0.9 s | 138.2 s |
| `fasth3-4step-vsa-480p-taeh3` (fastvideo-rs) | FAILED: fv-serve refuses one process for `fasth3-4step-vsa` and `ltx25-distill-sol` ("need different process-wide settings: FASTVIDEO_BF16_ACT=1 vs {}; serve them from separate processes") | | | | | | | |

The fastwan clip's 1.6 s denoise matches fastwan-rs's 2.2 s clip on the 600 W
Server Edition (denoise 1.61 s). The h3 stage / load / encode are paid once
per worker; the worker idles out after 5 s, so a quiet endpoint pays them
again on the next job (FlashBoot is off).

## 2026-11-19 00:30 UTC: the fastvideo-rs job on image `sha-e1e9f66` (one fv-serve per model)

`fasth3-4step-vsa-480p-taeh3` T2V 832×480×124, seed 7 (worker: RTX PRO 6000
Blackwell **Workstation** Edition this time, driver 595.112.07): COMPLETED,
MP4 832×480 124 frames H.264 + AAC 32 kHz (3.0 MB, ffprobe). fv-serve load
156.0 s (h3-base from the volume, FastH3 4-step LoRA, TAEH3), denoise 36.1 s
by fv-serve's metrics (its 4 steps + TAEH3 decode; the worker reports
fv-serve's numbers, `decode_s` 0), execution 195.6 s, cold worker start
170 s. With that, every engine of the catalog has produced a clip through
the endpoint.

## 2026-11-19 05:10 UTC: image `sha-ca445df` (Rust worker, one CUDA 13 runtime, SASS kernels)

The image is 1.09 GB compressed (15 layers, ghcr manifest): ubuntu:22.04,
the CUDA 13.4 libraries + cuDNN, fv-serve, llama-server, Ubuntu's ffmpeg,
`h3 @ 697fb9a` and `fastwan @ fa6a318` built with `--features cuda-13` and
loading their kernels as NVRTC-compiled SASS, `studio-worker` (Rust) as the
entrypoint, no Python. Worker: RTX PRO 6000 Blackwell **Server** Edition,
driver 595.112.07, compute capability 12.0, pod `f3q4rq4bbi50zx`.

### `op: check` (the engines' conformance checkers on the card)

| engine | result | time |
|---|---|---|
| `h3 check --backend cuda` | **all 180 checks pass** (tolerance 3e-2) | 17.4 s |
| `fastwan check --backend cuda` | 63 / 66: the three VSA cases fail at rel 0.28 (`vsa (21, 17, 30) h2 d128`, `vsa (21, 30, 52) h1 d128` and the small d128 case) | |

The fastwan failures are the known check-input problem, not a kernel bug and
not CUDA 13's: fastwan-rs's `docs/runs/2026-11-18-vast-4090-upscale-everywhere`
records the same three cases at rel 0.28–0.29 on an RTX 4090 under CUDA 12.8
(random inputs rounded to bf16 by the upload flip top-k picks; the CPU oracle
fed the rounded inputs moves 0.21–0.41 from its f32 self). The fix (bf16-
representable inputs on both sides, fastwan-rs `e72ff08`, extended to the
small d128 case) is on the fastwan-rs branch of zaitrarrio/fastwan-rs#7; the
next image picks it up. Every other fastwan check, including the tiny DiT
VSA / int8 / int4 rollouts, the causal rollouts and the upscaler, passes.
So: **NVRTC 13.4 → SASS for sm_120 gives the same results as the CUDA 12.8
PTX builds on both engines.**

### `endpoint.py smoke` (cold worker each time: idle timeout 5 s)

| job | result | stage | load | encode | denoise | decode | mux | execution |
|---|---|---|---|---|---|---|---|---|
| `h3-plug4` T2V 832×480×124, seed 7 | COMPLETED, MP4 832×480 124 fr H.264 + AAC 32 kHz (2.09 MB) | 56.5 s | 125.1 s | 0.1 s | 7.5 s (4 steps: 2.03 / 1.82 / 1.82 / 1.82) | 3.6 s | 4.8 s | 200.0 s |
| `fastwan-1.3b` T2V 832×480×81, seed 7 | COMPLETED, MP4 832×480 81 fr (1.13 MB) | 134.7 s (R2 fetch of the FastWan tree) | 10.0 s | 1.3 s | 1.55 s (3 steps of 0.51–0.53 s) | 0.3 s | 0.2 s | 150.1 s |
| `fasth3-4step-vsa-480p-taeh3` (fastvideo-rs) T2V 832×480×124, seed 7 | COMPLETED, MP4 832×480 124 fr H.264 + AAC (3.01 MB) | 0 s | 104.1 s (fv-serve) | 0.1 s | 6.2 s (fv-serve's metric: its 4 steps + TAEH3) | — | — | 115.5 s |

All three MP4s downloaded and ffprobe-verified. The h3 load (125 s) now
includes opening the Qwen3-VL encoder resident (h3-rs's 2026-11-18 text-
encoder placement: `serve` opens it last, ~35 s on this card), so the prompt
encodes in 0.13 s; the earlier image's h3 streamed the encoder per prompt
(load 110 + encode 79 s). fastwan's 0.51 s
step is fastwan-rs's 0.54 s on this card class; the fastvideo-rs H3 draft
ran 6.2 s on the Server Edition against 36.1 s on the Workstation Edition
pod of 00:30 (fv-serve load 104 against 156 s).

### Prompt expansion (`options.prompt: balanced`), `h3-plug4` T2V 832×480×124, seed 7

Job `92938c5f…`, cold worker. COMPLETED in 301 s wall (execution 294.8 s),
MP4 832×480 124 frames H.264 + AAC 32 kHz (2.09 MB, ffprobe).

| phase | time |
|---|---|
| staging the h3 trees | 55 s |
| staging the GGUF tree `qwen3-vl-4b-instruct-gguf` from R2 (5.1 GB, first use on this worker) | 82 s |
| llama-server start + the rewrite + `h3 serve` load (resident encoder) | 139 s (`load_s` 137.1) |
| h3 encode / denoise (4 steps) / decode / mux | 1.0 / 7.5 (2.05, 1.82, 1.82, 1.83) / 3.7 / 3.2 s |

The job's `expanded_prompt` (Qwen3-VL-4B through the H3 "balanced" system
prompt ported from fastvideo-rs), passed to `h3 generate` as its `--prompt`:

> integrated_multimodal_description: A red fox with glossy russet fur and
> white chest trots through untouched snow, paws crunching softly; dawn light
> glows pale gold through pine branches. The fox glances left, ears twitching,
> nose twitching. \<d\>The fox sniffs the air.\</d\> Close-up, low-angle
> tracking shot, shallow depth of field, golden-hour backlighting, cinematic
> realism.
> overall_soundscape: Distant pine rustle, crunching snow, faint birdsong.
> non_diegetic_music: None.

That is the H3 multimodal format FastVideo's pipeline expects (description
with a `<d>` dialogue tag, soundscape, music line). The GGUF staging and the
sidecar start are paid once per worker; on a warm worker the rewrite is the
only extra cost (not yet measured separately: the worker's `load_s` folds the
sidecar start and the rewrite into the h3 load).

## 2026-11-19 05:40 UTC: image `sha-f52591f` (fastwan at `b50c842`, the check-input fix)

Same image otherwise (1.09 GB compressed). Worker this time: RTX PRO 6000
Blackwell **Workstation** Edition, driver 595.112.07.

| engine | result | time |
|---|---|---|
| `fastwan check --backend cuda` | **all 66 checks pass** (tolerance 3e-2): the three VSA cases now at rel 2.4e-3 / 2.9e-3 / 2.8e-3 | 34.5 s |
| `h3 check --backend cuda` | **all 180 checks pass** | 16.0 s |

So the third failing case of `sha-ca445df` was the small `vsa (5, 6, 9) h2
d128` one, as guessed, and both engines' conformance suites are green on
CUDA 13.4 with NVRTC-compiled SASS on both Blackwell editions. This is the
image the endpoint runs.

## 2026-11-19 13:35 UTC: one fv-serve per process group (image `sha-db8ffb0`)

Two fastvideo-engine jobs of different H3 models submitted 3 s apart, so
the second queued behind the first on the same worker (RTX PRO 6000 Server
Edition):

| job | stage | load (`load_s`) | denoise | `total_s` | execution | queue delay |
|---|---|---|---|---|---|---|
| `fasth3-4step-vsa-480p-taeh3` T2V 832×480×124, seed 7 | 0.02 s | 159.7 s (fv-serve started for the `h3` group with this model resident) | 41.4 s (fv-serve's metric: 4 steps + TAEH3) | 203.9 s | 205.5 s | 177 s (cold worker) |
| `sol-h3`, same prompt and seed | 0.02 s | **0** (the group's fv-serve kept) | 38.9 s | 151.3 s | 152.7 s | 380 s (behind job 1) |

Before this change the second job restarted fv-serve and paid its load
again. Now the process stays and fv-serve loads `sol-h3` on demand inside
it (`[engine] swap = true`): that on-demand load is the ~110 s between the
second job's `denoise_s` and its `total_s`, which the worker's timings do
not name (fv-serve reports `load_s` only for the process start). The saving
against a restart is the process start and the model's `resident` load
path, about 50 s here; the real win is in the pod mode, where the group's
models stay loaded across jobs (`docs/runs/2026-11-19-runpod-6000w-pod-mode`).
