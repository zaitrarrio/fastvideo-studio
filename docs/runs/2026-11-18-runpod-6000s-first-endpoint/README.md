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
