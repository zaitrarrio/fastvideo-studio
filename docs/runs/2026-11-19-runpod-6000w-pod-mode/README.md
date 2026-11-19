# Pod mode, phase 1 on a RunPod RTX PRO 6000 (2026-11-19)

The pod backend of docs/pod-mode.md, first run: `scripts/runpod/pod.py up`
on image `sha-77c7d22` (the worker image, `studio-worker serve` as the
container command), 1-hour cap.

- **Stock.** EU-RO-1 had no RTX PRO 6000 of any edition for pods at 13:33
  UTC (`POST /v2/pods`: "no longer any instances available"); a hunt loop
  over the three editions every 45 s got a **Workstation Edition** at
  13:34:43 ($2.79/h; the Server Edition the endpoint usually gets is
  $3.69/h). Pod `2forhou5o0vinr`, driver 595.112.07.
- **Boot.** RunPod's proxy answered 404 while the image pulled; the serve
  answered `/health` 21 s after the container started (`uptime_s` 18.2), the
  volume mounted with the H3 / LTX trees.

## Two `h3-plug4` T2V 832×480×124 jobs, back to back

| job | stage | load | encode | denoise (4 forwards) | decode | mux | `total_s` | wall from submit |
|---|---|---|---|---|---|---|---|---|
| 1, seed 7 (cold: the trees staged to the container disk, `h3 serve` started) | 54.8 s | 121.6 s | 0.09 s | 8.45 s (2.28 / 2.05 / 2.06 / 2.07) | 2.66 s | 0.42 s | 188.8 s | 191.5 s |
| 2, seed 11, another prompt (warm) | 0.13 s | 0 | 0.04 s | 8.46 s (2.25 / 2.07 / 2.07 / 2.07) | 2.58 s | 0.50 s | 12.55 s | **16.2 s** |

Both MP4s uploaded to R2 (832×480, 124 frames, H.264 + AAC). `delayTime`
0 ms on both: no queue. The warm job is the pod mode's point: the same clip
costs 16 s end to end against the endpoint's 190–300 s (cold worker, staging
and load every time, plus the queue delay). The denoise matches the
endpoint's Server Edition numbers (7.5–7.8 s) within the Workstation
Edition's lower clocks.

`/health` after the two jobs: `busy: false, queued: 0, jobs_done: 2`, the
`h3` sidecar resident with its key. The pod was terminated from the script
at 14:03 (`DELETE /v2/pods/{id}`), 29 minutes in: about $1.35.

## Not yet run here

Live streams and segments (phases 2 and 3): the pod had no Cloudflare
Stream credentials (`CF_ACCOUNT_ID` / `CF_STREAM_API_TOKEN` on the Worker,
`FV_CF_*` in the pod env). A fastvideo-engine job on the pod (fv-serve
resident for a process group) was not timed: the same path ran on the
endpoint the same afternoon (`docs/runs/2026-11-18-runpod-6000s-first-endpoint`,
the per-process-group section).
