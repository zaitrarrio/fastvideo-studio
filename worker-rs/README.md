# fastvideo-studio worker (Rust)

`studio-worker` is the RunPod serverless worker of `docs/design.md`: one
container, three engines (`h3 serve` / `fastwan serve` / `fv-serve` as
resident sidecars, the engines' CLIs per job for the tasks serve lacks, a
`llama-server` sidecar for prompt expansion), driven by the job loop in
`src/rp.rs`. It is a module-for-module port of the Python worker
(`worker/handler.py` + `worker/fvstudio`), which stays in the tree as the
reference until it is removed: the contract (`docs/jobs.md`,
`catalog/catalog.json`), the environment variables, the ports, the staging
rule, the argv builders, the progress records, the output JSON and the R2
signatures are the same, and the tests below hold them to it.

**No Python, no OpenSSL, no async runtime.** The crate depends on `serde` /
`serde_json`, `ureq` (blocking HTTP, rustls through its `tls` feature),
`hmac` + `sha2` + `hex` (S3 SigV4), `roxmltree` (ListObjectsV2), `indexmap`,
`anyhow` / `thiserror`. Sidecars, ffmpeg and nvidia-smi are `std::process`;
signals and process groups are two `extern "C"` declarations (`signal`,
`kill`). The image's final stage needs only libc.

## Build and test

```sh
cargo build --release                 # target/release/studio-worker (stripped by the profile)
cargo test                            # 108 tests: unit tests in every module + the fakes in tests/
cargo test --release -- --ignored e2e # the real engines in --synthetic mode on the CPU (below)
```

The e2e test is the port of `worker/tests/test_e2e.py`: it builds `h3`
(`cargo build --release -p h3-cli` in `$H3_RS`, default `/home/user/h3-rs`)
and `fastwan` (`-p fastwan-cli` in `$FASTWAN_RS`), starts
`h3 serve --synthetic --backend cpu` and `fastwan serve --synthetic
--backend cpu` through the worker, a fake fv-serve and a directory as the
object store, and asserts what the Python test asserts: the MP4s through
ffprobe, the stages seen, the resident `h3 serve` taking FL2VA, sidecar
reuse and crash recovery, the `where: engine` + `log_tail` shape when a
sidecar dies mid-job, validation errors, the fastwan ×2 upscale, the
fastvideo fake job (its `job.json` in the store), `op: warm`. The
scenarios run in order inside one `#[test]` (they share the h3 sidecar).
It needs ffmpeg / ffprobe for the mux assertions.

## Running

```sh
studio-worker                          # RunPod mode: RUNPOD_WEBHOOK_GET_JOB etc. set by the platform
studio-worker --test-input job.json    # local mode: one job ({"input": …}), the job-done body on stdout
```

Exit codes in local mode: 0 completed, 1 the job failed (`error` in the
body), 2 no job file. Without `RUNPOD_WEBHOOK_GET_JOB` the worker is in
local mode and looks for `test_input.json` in the working directory.

## The ops (unchanged from `docs/jobs.md`)

| op | what |
|---|---|
| `generate` (default) | validate → stage → expand the prompt (when `options.prompt` asks) → run the engine → mux → upload `jobs/<id>/{video.mp4, audio.wav, thumb.jpg, job.json}` → presigned URLs; `{"error", "where": validate\|stage\|engine\|mux\|upload, "log_tail"?}` on failure |
| `info` | `gpu, driver, vram_*, compute_cap, image, catalog_version, versions, sidecars, upsampler, trees, staged, volume, uptime_s, jobs_done, pod, endpoint` |
| `warm` | stage a model's trees and start its sidecar: `stage_s, fetched, staged, load_s, total_s, sidecars` |
| `r2` | can the worker write the Studio bucket and read the weights store (names and outcomes, never a secret) |
| `check` | `h3 check` / `fastwan check` on this card: `results.<engine>.{exit, seconds, tail, command}` |

Progress records (`{stage, detail?, step?, total_steps?, frames?,
total_frames?, elapsed_s, engine, model}`) go out as RunPod's
`{"status": "IN_PROGRESS", "output": …}` from a sender thread; a new stage
always goes out, detail updates within a stage at most every 0.5 s.

## Environment

The same variables as the Python worker (`worker/README.md`,
"Environment"): `STUDIO_*` (catalog, bin dir, fv-serve, llama-server, ports
7861 / 7862 / 8000 / 18080, volume / stage / link roots, jobs dir, backend,
synthetic, image, URL TTL, ESRGAN / TAEH3 weights, `STUDIO_R2_LOCAL_DIR`,
`STUDIO_STAGE_DRY_RUN`, `STUDIO_KEEP_JOB_DIRS`), `FV_R2_*` /
`STUDIO_R2_*` for R2, `RUNPOD_*` from the platform.

## How it maps to the Python modules

| Rust | Python | notes |
|---|---|---|
| `src/main.rs` | `handler.py` | `--test-input`, local mode, the loop with SIGTERM / SIGINT |
| `src/worker.rs` | `fvstudio/worker.py` | `Config`, `Worker::{handle, generate, info, warm, r2check, check, expand_prompt}`, `_finish`, `_redact_input` |
| `src/catalog.rs` | `fvstudio/catalog.py` | `Catalog::validate` and the same messages; `Job`, `Knob::coerce`, `FrameGrid`, `Media`; base64 |
| `src/flags.rs` | `fvstudio/flags.py` | `parse_flags` / `emit` / `knob_overlay`, the h3 / fastwan argv builders, `fastvideo_body`, `command_line` (`shlex.join`) |
| `src/weights.rs` | `fvstudio/weights.py` | `Roots`, `TreeEntry`, `WeightsResolver` (volume link → R2 manifest fetch → `stage TREE[:SUB,…]` with 8 parallel `cp --update=none`, size-verified, add-only) |
| `src/sidecar.rs` | `fvstudio/sidecar.py` | `Sidecar` (process group, log ring, probes), `SidecarManager::{ensure, make_room, stop_all, resident}` |
| `src/engines/{common,h3,fastwan,fastvideo}.rs` | `fvstudio/engines/*.py` | the serve job loop (`/api/generate`, `/api/jobs/<id>`, `/frames`, `/audio`, `/hd`), the CLI runner (`step i/n`, `block b:` lines), the three drivers |
| `src/frames.rs`, `src/mux.rs`, `src/media.rs` | `frames.py`, `mux.py`, `media.py` | frame stream → raw RGB → y4m through ffmpeg; mux / thumbnail / probe / to_y4m / to_wav; job media → files |
| `src/progress.rs`, `src/gpu.rs`, `src/httpc.rs` | `progress.py`, `gpu.py`, `httpc.py` | `Reporter` + `Timings`; nvidia-smi; the loopback client |
| `src/r2.rs` | `fvstudio/r2.py` | `SigV4` (header auth + presign), `R2Store` (retries on 429 / 5xx and transport errors), `LocalStore` |
| `src/rp.rs` | `fvstudio/rp.py` | `RunpodEnv`, `RunpodClient` (take / progress / done / ping), `run_job`, `serve`, `run_local`, signal handlers |
| `src/upsampler.rs` | `fvstudio/upsampler.py` | the llama-server argv / probe, the family system prompts, cleaning, `Upsampler::expand` |
| `src/py.rs` | — | Python-compatible `json.dumps`, `repr(float)`, `shlex`, `quote`, `strftime`, `which`, `realpath` |
| `tests/common/mod.rs` | `tests/fake_fv_serve.py`, `tests/fake_llama.py`, the fakes in `test_rp.py` / `test_r2.py` | tiny_http fakes |
| `tests/{rp,r2,fastvideo_driver,upsampler,local_mode}.rs` | `test_rp.py`, `test_r2.py`, `test_fastvideo_driver.py`, `test_upsampler.py`, `test_rp.py::test_handler_main…` | |
| `tests/e2e.rs` | `tests/test_e2e.py` | `cargo test --release -- --ignored e2e` |

The SigV4 tests hold the signer to the values the Python signer produces
for `test_r2.py`'s cases (which that test holds byte-identical to
botocore's) plus the AWS docs' signing-key vector; the flag tests hold the
argv builders to the same run-log lines `test_flags.py` uses.

## Known differences from the Python worker

- An unexpected failure inside a job is reported as `{"error": "<text>",
  "where": "engine", "log_tail"}` like the Python catch-all, but the text is
  the Rust error's (`OSError: …` for I/O), not a Python exception class name.
- A handler panic becomes the SDK-style report (`error_type` is
  `<class 'panic'>`, `error_traceback` a Rust backtrace).
- `mimetypes.guess_type` is a fixed table of the media types a job carries
  (Python also reads the system's `mime.types`).
- `round(x, 3)` on timings is `(x·1000).round()/1000`, which can differ from
  Python's correctly rounded `round()` in the last digit of a half-way value.
