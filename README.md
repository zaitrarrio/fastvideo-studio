# fastvideo-studio

One RunPod serverless endpoint for three engines, and the Studio that
drives it.

| Part | Where | What |
|---|---|---|
| Catalog | `catalog/catalog.json` | Every model the endpoint serves: engine, tasks, sizes, frame grid, knobs, recipe, weight trees. The one contract the worker and the page share. |
| Worker | `worker/` | The RunPod queue handler: dispatches a job to `h3` (h3-rs), `fastwan` (fastwan-rs) or `fv-serve` (fastvideo-rs), keeps the engines resident, stages weights, muxes MP4s, uploads to R2. One image, `ghcr.io/zaitrarrio/fastvideo-studio-worker`. |
| Endpoint tooling | `scripts/runpod/endpoint.py` | Create / smoke / run / tear down the endpoint through RunPod's REST API, with the money guards the engines' repos use. |
| Studio | `studio/` | A Cloudflare Worker (holds the RunPod key, proxies jobs, takes uploads into R2) and the page: Prompt Studio's layout and tokens, one form generated from the catalog. |
| Docs | `docs/design.md`, `docs/jobs.md`, `docs/runbook.md` | Architecture and invariants, the job schema, how to deploy and verify. |

The engines are separate repositories: [h3-rs](https://github.com/zaitrarrio/h3-rs),
[fastwan-rs](https://github.com/zaitrarrio/fastwan-rs),
[fastvideo-rs](https://github.com/zaitrarrio/fastvideo-rs). This repo adds no
kernels and no model code; it builds their binaries at pinned commits and
drives their `serve` / jobs APIs.

Status: written and tested against the engines' synthetic CPU modes and a
fake RunPod loop. Not yet run on a GPU; `docs/runbook.md` lists the first
real run step by step.
