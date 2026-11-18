# Job schema (RunPod `input` / `output`)

Every request to the endpoint is `POST /run` (or `/runsync` for `info`)
with `{"input": …}`. The worker validates `input` against
`catalog/catalog.json` and refuses anything it does not list.

## `op: "generate"` (default)

```json
{
  "op": "generate",
  "engine": "h3" | "fastwan" | "fastvideo",
  "model": "fasth3-v2",                 // catalog model id (without engine prefix)
  "task": "t2v" | "fl2v" | "i2v" | "ref2v" | "a2v" | "long" | "extend" | "retake" | "regen2k",
  "prompt": "…",                         // ≤ 2000 chars
  "negative_prompt": "…",                // fastvideo only
  "width": 1344, "height": 768,          // one of the model's sizes
  "frames": 124,                         // on the model's grid
  "seed": 7,                             // optional, random when absent
  "inputs": {                            // per task, see catalog.tasks[task].inputs
    "first_image": <media>, "last_image": <media>,
    "references": [<media>, …],          // image / video / audio
    "audio": <media>, "video": <media>, "draft": <media>
  },
  "options": { "weights": "mxfp8", "pv8": true, … },   // only knobs the model lists
  "upscale": 0 | 2 | 3 | 4,
  "client": { "studio": "…", "job": "…" }               // echoed back, free-form
}
```

`<media>` is one of
`{"url": "https://…"}`, `{"b64": "…", "mime": "image/png"}` (≤ 6 MiB inline),
`{"r2": "uploads/<key>"}` (an object in the Studio's bucket the worker can read).

### Progress (`/status` while IN_PROGRESS)

```json
{ "stage": "queued|staging|loading|encoding|preparing|denoising|decoding|upscaling|muxing|uploading|done|failed",
  "detail": "block 3/12", "step": 3, "total_steps": 8, "frames": 48, "total_frames": 124,
  "elapsed_s": 41.2, "engine": "h3", "model": "fasth3-v2" }
```

### Output (COMPLETED)

```json
{
  "engine": "h3", "model": "fasth3-v2", "task": "t2v", "seed": 7,
  "width": 1344, "height": 768, "frames": 124, "fps": 24, "audio": true,
  "video_url": "https://…/jobs/<id>/video.mp4?…",   // presigned, 24 h
  "audio_url": "https://…/audio.wav?…",             // h3 only
  "thumbnail_url": "https://…/thumb.jpg?…",
  "upscale": { "factor": 2, "width": 2688, "height": 1536 } | null,
  "timings": { "stage_s": 12.1, "load_s": 118.0, "encode_s": 3.2, "denoise_s": 42.6, "decode_s": 6.3, "upscale_s": 0, "mux_s": 1.1, "total_s": 183.3, "steps_s": [5.3, …] },
  "command": "h3 generate --backend cuda --variant fasth3-v2 …",   // reproducible CLI line
  "worker": { "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "driver": "580.xx", "image": "sha-…" },
  "expires_at": "2026-11-19T12:00:00Z",
  "client": { … }
}
```

A refused or failed job returns `{"error": "…", "where": "validate|stage|engine|mux|upload"}`.

## `op: "info"`

Returns the worker's GPU, driver, the three binaries' versions, which
catalog trees are present on the volume / staged, which sidecars are
resident, and the catalog version. Used by the Studio's health panel and
the smoke test.

## `op: "warm"`

`{"op": "warm", "engine": "h3", "model": "fasth3-v2"}` stages the trees
and starts the sidecar, returning once `/api/info` reports ready.
