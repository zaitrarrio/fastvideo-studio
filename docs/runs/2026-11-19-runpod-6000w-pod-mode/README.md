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

## 2026-11-21: three pod-only faults, from the first real stream and segment

The owner's pods (`sha-91c0ff4`, then `sha-8e0ced8`) failed at every live
stream and every segment session. Three faults in a row, each hidden by the
one before it, all of them pod-only (the serverless endpoint never hits
them):

1. **fv-serve's port.** `studio-worker serve` listens on `0.0.0.0:8000` and
   the image's fv-serve config bound `127.0.0.1:8000`: the sidecar died at
   start with `Address already in use (os error 98)`, reported as
   "fastvideo sidecar exited with code 1 while loading". `serve_argv` now
   rewrites `[server] bind` to the worker's fastvideo port (8010 by default,
   `STUDIO_FV_PORT`) in the config it starts fv-serve with, and the serve
   refuses to start on that port (8e0ced8).
2. **No staging on the stream path.** `fastvideo_serve` (`/streams`,
   `/segments`) went straight to `ensure_serve`, so fv-serve started with no
   weights linked or staged, its resident model's load failed and its
   `/ping` answered 503 `{"status":"failed"}` → "fastvideo failed to start".
   The job path had always staged first; this one does too now (8db1408).
   The workaround on a running pod is one clip job for the model.
3. **fv-serve's artifact URLs.** With the weights staged by a clip job, a
   480p H3 clip loaded in 130 s and denoised, then the job failed:
   `HTTP 404 from https://<pod>-8010.proxy.runpod.net/files/…`. fv-serve
   discovers the pod's public base (`Discovery::public_base_url`) and builds
   artifact URLs on **its own** port, which Runpod's proxy only answers for
   the one exposed port — and which sent the worker's own download out over
   the internet. It now gets `FV_PUBLIC_BASE_URL=http://127.0.0.1:<its
   port>` (9ea0d36).

Also: the Studio Worker showed the pod error's **last** log line, which was
fv-serve's own request trace ("response failed classification=Status code:
503"); it now skips those and picks the line that names the cause. And
`CF_ACCOUNT_ID` / `CF_STREAM_API_TOKEN` added in the dashboard as plain
variables were wiped by the next `wrangler deploy`: `keep_vars = true`
(2de0202), with both named as secrets in the README.

Not yet seen on a pod: a live stream reaching `publishing`, and a segment
MP4 mirrored to R2. The pod was gone before `sha-9ea0d36` existed.
