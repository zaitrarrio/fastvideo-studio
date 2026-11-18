# Runbook: first deploy and first real run

Nothing below has been run on a GPU yet. Each step says what to check.

## 0. Prerequisites

- RunPod account with the EU network volume `bz0ofkfx45`
  (`fv-weights-h3-ltx-500`, EU-RO-1) and the secrets fastvideo-rs defines
  (`fv_r2_endpoint`, `fv_r2_access_key_id`, `fv_r2_secret_access_key`,
  `fv_r2_bucket`; `fastvideo-rs/docs/ops/runpod-secrets.md`). Add
  `studio_output_bucket` (an R2 bucket for job outputs and uploads; the
  Studio Worker binds the same bucket).
- The volume must hold the trees the catalog names. Missing today for the
  h3-rs engine: nothing for the V2 / plug4 / pika / omni / regen recipes
  (`h3-rs/docs/runs/2026-11-12-runpod-volume-fill`); `h3-rs/cfg` (the
  h3-cfg adapter) and the FastH3 V1 LoRA path are flagged `verify` in the
  catalog. FastWan and Self-Forcing are fetched from R2 / the public bucket
  on first use. Never run two writers on the volume.
- `RUNPOD_API_KEY` in the environment of whoever runs
  `scripts/runpod/endpoint.py`.

## 1. Build the image

`.github/workflows/worker-image.yml` builds
`ghcr.io/zaitrarrio/fastvideo-studio-worker:sha-<sha>` from the pinned
`H3_REF` / `FASTWAN_REF` (the Dockerfile's defaults) on top of
`ghcr.io/zaitrarrio/fastvideo-rs-serve:cuda-stable`. Run it with
`workflow_dispatch`; if the engine repos are private, set the
`REPO_READ_TOKEN` secret. Check: the workflow's last step prints the three
binaries' `--version` / `--help` heads.

## 2. Create the endpoint

RunPod's REST v2 (`https://api.runpod.io/v2`, bearer `RUNPOD_API_KEY`;
`rest.runpod.io` v1 no longer resolves). A v2 endpoint takes the image
directly (no template), chooses GPUs by **pool** (`gpus` lists the catalog:
the RTX PRO 6000 family is `BLACKWELL_96`, the RTX 5090 `ADA_32_PRO`) minus
excluded type ids, and pulls a private ghcr image through a
container-registry credential (`registry`; the account's are named
`fv-ghcr-<timestamp>`, `--registry-name 'fv-ghcr-*'` picks the newest).

```bash
export RUNPOD_API_KEY=…
export STUDIO_IMAGE=ghcr.io/zaitrarrio/fastvideo-studio-worker:sha-<sha>
python3 scripts/runpod/endpoint.py gpus --pool BLACKWELL_96        # ids and serverless $/h in the pool
python3 scripts/runpod/endpoint.py plan --registry-name 'fv-ghcr-*' # the create body, nothing sent
python3 scripts/runpod/endpoint.py up --name studio --disk 300 --registry-name 'fv-ghcr-*'
python3 scripts/runpod/endpoint.py backstop 180                     # `down --yes` later, whatever happens
```

`up` first checks the pool against `GET /v2/catalog/gpus` (printing each
GPU id and its serverless price), refuses a name that an endpoint already
carries (prints its id), prints the body, POSTs `/v2/serverless`, prints
the id and `requestUrls`, and writes `.runpod-endpoint.json` (id, name,
createdAt, image) in the working directory; `status`, `health`, `down`
and `backstop` read it (or `--id`). Defaults: workers 0 / 1 (more needs
`--i-know`), idle 5 s, FlashBoot off (`--flashboot` turns it on), request
timeout 30 min, `minCudaVersion 13.0` (fv-serve's floor), pool
`BLACKWELL_96` minus the Max-Q Workstation Edition (`--exclude`, repeatable;
`--exclude ''` for none), volume `bz0ofkfx45` mounted at `/runpod-volume`
in EU-RO-1. The 300 GB container disk is for staging (h3 V2 recipe 103 GB,
base MXFP8 + text side ~60 GB, FastWan ~15 GB, fastvideo-rs trees as
needed). Extra worker env goes in with `--env K=V`.

## 3. Smoke

```bash
python3 scripts/runpod/endpoint.py smoke
```

Runs `op: info` (prints GPU, driver, binaries, trees present), then one
480p draft job per engine: `h3-plug4` 832×480×124, `fastwan-1.3b`
832×480×81, `fastwan22-ti2v-5b` 832×480×49. Check: each returns a
`video_url` that plays, `timings` that match the engines' measured numbers
within reason (h3 draft ~9 s denoise after load on a 600 W PRO 6000,
FastWan clip ~2 s), and the worker's `command` line reproduces on a rented
box. The cold job pays worker start + staging + load; the second job of
an engine must not.

Then the full matrix, one job each, in `endpoint.py run` form:
`fasth3-v2` 768p T2V, `h3-plug4` FL2VA with two images (goes through
`h3 serve` once the serve-inputs change is in the image, else the CLI),
`h3-turbo4-ref` Ref2VA, `pika-flex` FL2VA, `regen-2k` from the draft,
`self-forcing-1.3b` 321 frames, `ltx25-distill-sol` T2V and A2V,
`h3-ref2v-turbo` on fastvideo-rs. Record a `docs/runs/<date>-<box>/`
like the engines do: command, timings, GPU, driver, image sha.

## 4. Deploy the Studio

```bash
cd studio
npx wrangler@4 r2 bucket create fastvideo-studio
npx wrangler@4 secret put RUNPOD_API_KEY
npx wrangler@4 secret put RUNPOD_ENDPOINT_ID
npx wrangler@4 secret put STUDIO_TOKEN
npx wrangler@4 deploy
```

Open the Worker URL, unlock with the token, pick H3 → FastH3 V2, generate.
The endpoint's `op: info` is under the session panel.

## 5. Tear down

```bash
python3 scripts/runpod/endpoint.py down --yes
```

Outputs in R2 expire by URL only; prune `jobs/` with a lifecycle rule.
