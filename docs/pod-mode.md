# Pod mode: Fast Video Studio on a persistent pod

Owner decision (2026-11-19): the Studio runs two backends. **Serverless**
(the RunPod queue endpoint, `docs/design.md`): one job, one MP4, a worker
that idles out after 5 s; cold starts, no state between jobs. **Pod**: one
persistent RunPod pod running the same worker image in `serve` mode, with
every sidecar kept resident, and three ways to use it, all on it at once:

| mode | what the page does | what the pod does |
|---|---|---|
| **Warm clip jobs** | the form as today, `backend: pod` | the job pipeline over HTTP (`POST /jobs`), no queue delay, no cold start: a second job of the same model skips staging and load |
| **Live stream** (clip queue / causal) | Start a stream, watch it over WHEP, steer prompts, pause / resume / reset / stop | fv-serve's `/fv/v1/streams` with Cloudflare Stream as the output: a clip model (H3, LTX, Wan) as a clip queue, a causal one (SF-Wan, LongLive) as one continuous rollout |
| **Segments** | ask for 5–10 s segments of one stream, a prompt per segment, play them back to back | fv-serve's `/v1/segments`: one continuous rollout (causal) or clips chained on the last frame (clip models), each segment an MP4 the pod uploads to R2 |

The serverless backend stays exactly as it is; the pod backend is opt-in
per page (the choice persists in the browser).

## 1. The pod

- **Image**: the worker image (`ghcr.io/zaitrarrio/fastvideo-studio-worker`),
  started as `studio-worker serve` (the container's CMD; the serverless
  entrypoint is the same binary with no arguments). Same engines, same
  catalog, same R2 staging; the sidecars (`h3 serve`, `fastwan serve`, one
  `fv-serve` per process group) stay resident for the pod's life.
- **Created by the Studio Worker** (Cloudflare) through RunPod's REST API v2
  (`POST /v2/pods`): `image`, `cloud: SECURE`, `gpu {id, count: 1}` from
  `POD_GPU` (default the RTX PRO 6000 Blackwell Server Edition), `disk`
  (`POD_DISK_GB`, 300: the staged trees), `ports ["8000/http"]`, the EU
  weights volume (`POD_VOLUME`, `bz0ofkfx45`) at `/runpod-volume` in its data
  centre (`POD_DC`, EU-RO-1), `cmd ["serve"]`, and the env of the serverless
  endpoint (the `{{ RUNPOD_SECRET_… }}` R2 references, `STUDIO_IMAGE`) plus
  `STUDIO_SERVE_TOKEN` (a per-pod random bearer token the Worker mints),
  `STUDIO_POD_HOURS` (the wall-clock cap), `RUNPOD_API_KEY` + `RUNPOD_POD_ID`
  (so the pod can terminate itself at the cap), and for streams
  `FV_CF_ACCOUNT_ID` / `FV_CF_STREAM_API_TOKEN` (Cloudflare Stream, Worker
  secrets `CF_ACCOUNT_ID` / `CF_STREAM_API_TOKEN`).
- **Reached at** `https://<pod>-8000.proxy.runpod.net` (RunPod's HTTPS
  proxy), only from the Worker, with the pod token. The page never holds it.
- **Record**: the Worker keeps the one current pod in the R2 bucket at
  `studio/pod.json` (`id`, `url`, `token`, `image`, `created_at`); `GET
  /api/pod` joins it with RunPod's pod status and the pod's `/health`.
- **Lifetime**: the Worker starts and stops it on request (`POST` / `DELETE
  /api/pod`); the pod exits and terminates itself at `STUDIO_POD_HOURS`
  (default 4) whatever the Worker does, the serverless scripts' backstop
  rule. One pod at a time (one GPU; a second `POST` is 409).

## 2. `studio-worker serve`

A small HTTP server (`tiny_http`, no async runtime, the crate's style) on
`STUDIO_SERVE_BIND` (`0.0.0.0:8000`), bearer `STUDIO_SERVE_TOKEN` on every
route (`STUDIO_SERVE_NO_AUTH=1` only for tests).

| route | behaviour |
|---|---|
| `GET /health` | `{ok, uptime_s, busy, queued, jobs_done, pod}` |
| `GET /info` | `op: info` (the catalog, trees, resident sidecars, GPU) |
| `POST /jobs` | the serverless job input (`docs/jobs.md`; `{"input": …}` or the input itself) → 202 `{id, status: IN_QUEUE}`. Jobs run one at a time, FIFO, on the pod's one GPU |
| `GET /jobs/{id}` | RunPod's status shape, so the Worker's normaliser is shared: `output` is the last progress record while `IN_PROGRESS` and the result once `COMPLETED`; `delayTime` / `executionTime` in ms |
| `POST /jobs/{id}/cancel` | queued → `CANCELLED`; running → the engine sidecar is stopped and the job ends `CANCELLED` |
| `GET /jobs` | the recent jobs (the last 64) |
| `POST /streams`, `GET /streams/{id}`, `POST /streams/{id}/commands`, `DELETE /streams/{id}` | phase 2: fv-serve's `/fv/v1/streams` on the model's process group, `output.kind = cloudflare`; the page gets `playback.whep_url` |
| `POST /segments`, `GET /segments/{id}`, `GET` / `DELETE /segments/streams/{id}` | phase 3: fv-serve's `/v1/segments`; a finished segment's MP4 is uploaded to R2 and answered as a presigned `video_url` |

Jobs, streams and segments share the GPU: a live stream or a segment session
holds its fv-serve, so a job for another process group waits in the queue
until the stream is stopped (`make_room` cannot stop a session the page is
watching); the status says so (`stage: queued`, detail "waiting for the
live stream").

## 3. The Worker and the page

- `POST /api/jobs` takes `backend: "serverless" | "pod"` (default
  serverless); pod job ids are `pod_<id>` and `GET /api/jobs/pod_…` /
  `cancel` route to the pod. `GET /api/pod`, `POST /api/pod`, `DELETE
  /api/pod` manage the pod. `/api/health` adds the pod's summary.
- The page gets a **Run on** switch (Serverless queue | Pod) with the pod's
  state and Start / Stop. In pod mode the form is unchanged for clip jobs
  (warm), and two more modes appear: **Live** (model, opening prompt,
  Start; a WHEP player; a steer box with Switch / Enqueue, Pause, Resume,
  Reset, Stop; the stream's state line) and **Segments** (a prompt per
  segment, duration, Next segment; two `<video>` elements alternating so the
  next MP4 is loaded while one plays).
- Everything the page sends still goes through the Worker with the studio
  token; the pod token never reaches the browser.

## 4. Validation

- Synthetic: `cargo test -p fastvideo-studio-worker` runs the serve routes
  (auth, a job through the pipeline, cancel, status shapes) and the Studio's
  `node --test` runs the Worker's pod routes against fakes of RunPod v2 and
  of the pod.
- On a pod (RTX PRO 6000, EU-RO-1): warm jobs (the second job of a model
  reports `stage_s` 0 and a small `load_s`), a clip-queue stream of
  `fasth3-4step-vsa-480p-taeh3` and a causal one of `sfwan21-1.3b` (the
  Cloudflare live input reaches `publishing`, WHEP playback answers), three
  segments of `sfwan21-1.3b` played back to back. Recorded under
  `docs/runs/`.

## 5. Phases

1. `studio-worker serve` + the Worker's backend switch and pod lifecycle +
   warm clip jobs in the page.
2. Live streams (clip queue / causal) with WHEP playback and steering.
3. Segments.
