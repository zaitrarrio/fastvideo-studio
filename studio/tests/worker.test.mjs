// Unit tests for studio/worker.js: `node --test studio/tests/`.
// RunPod is a fake fetch; R2 is an in-memory stub.
import { test, describe, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import worker, { normaliseStatus, uploadKey, podCreateBody, podEnv, podUrl } from "../worker.js";
import catalog from "../../catalog/catalog.json" with { type: "json" };

const ORIGIN = "https://studio.example.test";
const TOKEN = "s3cret-token";

class FakeR2 {
  constructor() { this.objects = new Map(); }
  async put(key, body, opts) {
    const buf = body instanceof ArrayBuffer ? body : await new Response(body).arrayBuffer();
    this.objects.set(key, { body: buf, opts });
    return { key };
  }
  async get(key) {
    const o = this.objects.get(key);
    return o ? { text: async () => new TextDecoder().decode(o.body) } : null;
  }
  async delete(key) { this.objects.delete(key); }
}

function env(extra = {}) {
  return {
    RUNPOD_API_KEY: "rp_key", RUNPOD_ENDPOINT_ID: "ep123", STUDIO_TOKEN: TOKEN, ENDPOINT_LABEL: "test endpoint",
    UPLOADS: new FakeR2(),
    ASSETS: { fetch: async () => new Response("asset", { status: 200 }) },
    ...extra,
  };
}

// A RunPod double: records calls, answers from `routes` (path suffix → body or fn).
function fakeRunpod(routes) {
  const calls = [];
  const fn = async (url, init = {}) => {
    const u = new URL(url);
    assert.equal(u.origin, "https://api.runpod.ai");
    assert.equal(init.headers.Authorization, "Bearer rp_key");
    const path = u.pathname.replace("/v2/ep123", "");
    calls.push({ path, method: init.method || "GET", body: init.body ? JSON.parse(init.body) : undefined, signal: init.signal });
    const hit = Object.entries(routes).find(([p]) => path === p || path.startsWith(p));
    if (!hit) return new Response(JSON.stringify({ error: "no route " + path }), { status: 404 });
    const v = typeof hit[1] === "function" ? hit[1](calls.at(-1)) : hit[1];
    if (v instanceof Response) return v;
    return new Response(JSON.stringify(v), { status: 200, headers: { "Content-Type": "application/json" } });
  };
  fn.calls = calls;
  return fn;
}

function req(path, { method = "GET", body, headers = {}, auth = true, origin = ORIGIN } = {}) {
  const h = new Headers(headers);
  if (auth) h.set("Authorization", `Bearer ${TOKEN}`);
  if (origin) h.set("Origin", origin);
  if (body !== undefined && typeof body !== "string" && !(body instanceof ArrayBuffer) && !(body instanceof FormData)) {
    h.set("Content-Type", "application/json"); body = JSON.stringify(body);
  }
  return new Request(ORIGIN + path, { method, headers: h, body });
}

const realFetch = globalThis.fetch;
let rp;
beforeEach(() => { rp = fakeRunpod({}); globalThis.fetch = rp; });
afterEach(() => { globalThis.fetch = realFetch; });

describe("auth and routing", () => {
  test("the catalog is public and is the repo's file", async () => {
    const r = await worker.fetch(req("/api/catalog", { auth: false, origin: null }), env());
    assert.equal(r.status, 200);
    assert.equal(r.headers.get("Cache-Control"), "no-store");
    const c = await r.json();
    assert.equal(c.version, catalog.version);
    assert.equal(c.models.length, catalog.models.length);
  });

  test("every other API route needs the bearer token", async () => {
    for (const [path, method] of [["/api/health", "GET"], ["/api/jobs", "POST"], ["/api/jobs/abc", "GET"], ["/api/jobs/abc/cancel", "POST"], ["/api/uploads", "POST"], ["/api/info", "POST"]]) {
      const r = await worker.fetch(req(path, { method, auth: false, body: method === "POST" ? {} : undefined }), env());
      assert.equal(r.status, 401, `${method} ${path}`);
      assert.equal(r.headers.get("Cache-Control"), "no-store");
    }
    const wrong = await worker.fetch(req("/api/health", { headers: { Authorization: "Bearer nope" }, auth: false }), env());
    assert.equal(wrong.status, 401);
    assert.equal(rp.calls.length, 0, "nothing reached RunPod");
  });

  test("a missing STUDIO_TOKEN refuses rather than opens", async () => {
    const r = await worker.fetch(req("/api/health"), env({ STUDIO_TOKEN: undefined }));
    assert.equal(r.status, 503);
  });

  test("a foreign Origin on a POST is rejected; same-origin and origin-less pass", async () => {
    rp = fakeRunpod({ "/run": { id: "j1", status: "IN_QUEUE" } }); globalThis.fetch = rp;
    const body = { engine: "h3", model: "fasth3-v2", prompt: "a fox" };
    const foreign = await worker.fetch(req("/api/jobs", { method: "POST", body, origin: "https://evil.example" }), env());
    assert.equal(foreign.status, 403);
    assert.equal(rp.calls.length, 0);
    const same = await worker.fetch(req("/api/jobs", { method: "POST", body }), env());
    assert.equal(same.status, 202);
    const none = await worker.fetch(req("/api/jobs", { method: "POST", body, origin: null }), env());
    assert.equal(none.status, 202);
  });

  test("non-API paths go to the asset binding", async () => {
    const r = await worker.fetch(new Request(ORIGIN + "/"), env());
    assert.equal(await r.text(), "asset");
  });

  test("unknown API routes are 404, wrong methods 405", async () => {
    assert.equal((await worker.fetch(req("/api/nope"), env())).status, 404);
    assert.equal((await worker.fetch(req("/api/jobs"), env())).status, 405);
    assert.equal((await worker.fetch(req("/api/health", { method: "POST", body: {} }), env())).status, 405);
  });
});

describe("health", () => {
  test("passes the endpoint's counts through with the label", async () => {
    rp = fakeRunpod({ "/health": { jobs: { inQueue: 2, inProgress: 1 }, workers: { idle: 1, running: 1 } } }); globalThis.fetch = rp;
    const r = await worker.fetch(req("/api/health"), env());
    assert.equal(r.status, 200);
    assert.deepEqual(await r.json(), { label: "test endpoint", workers: { idle: 1, running: 1 }, jobs: { inQueue: 2, inProgress: 1 }, pod: { state: "none" } });
    assert.equal(rp.calls[0].path, "/health");
  });

  test("an unreachable RunPod is a 502 with a message", async () => {
    globalThis.fetch = async () => { throw new TypeError("fetch failed"); };
    const r = await worker.fetch(req("/api/health"), env());
    assert.equal(r.status, 502);
    assert.match((await r.json()).error, /unreachable/);
  });
});

describe("jobs", () => {
  test("submit maps the input to /run with the execution policy", async () => {
    rp = fakeRunpod({ "/run": { id: "job-1", status: "IN_QUEUE" } }); globalThis.fetch = rp;
    const input = { op: "generate", engine: "h3", model: "fasth3-v2", task: "t2v", prompt: "a fox", width: 1344, height: 768, frames: 124, seed: 7, options: { weights: "mxfp8" } };
    const r = await worker.fetch(req("/api/jobs", { method: "POST", body: input }), env());
    assert.equal(r.status, 202);
    assert.deepEqual(await r.json(), { id: "job-1", status: "IN_QUEUE", backend: "serverless" });
    assert.equal(rp.calls[0].path, "/run");
    assert.equal(rp.calls[0].method, "POST");
    assert.deepEqual(rp.calls[0].body, { input, policy: { executionTimeout: 1800000 } });
  });

  test("submit refuses a non-JSON body, an unknown model and an empty prompt before RunPod", async () => {
    const e = env();
    const notJson = await worker.fetch(new Request(ORIGIN + "/api/jobs", { method: "POST", headers: { Authorization: `Bearer ${TOKEN}`, "Content-Type": "text/plain" }, body: "hi" }), e);
    assert.equal(notJson.status, 415);
    const unknown = await worker.fetch(req("/api/jobs", { method: "POST", body: { engine: "h3", model: "nope", prompt: "x" } }), e);
    assert.equal(unknown.status, 400);
    const empty = await worker.fetch(req("/api/jobs", { method: "POST", body: { engine: "h3", model: "fasth3-v2", prompt: " " } }), e);
    assert.equal(empty.status, 400);
    assert.equal(rp.calls.length, 0);
  });

  test("status while IN_PROGRESS carries the progress object", async () => {
    const progress = { stage: "denoising", detail: "block 3/12", step: 3, total_steps: 8, elapsed_s: 41.2, engine: "h3", model: "fasth3-v2" };
    rp = fakeRunpod({ "/status/job-1": { id: "job-1", status: "IN_PROGRESS", output: progress, delayTime: 1200, executionTime: 41000, workerId: "w1" } }); globalThis.fetch = rp;
    const r = await worker.fetch(req("/api/jobs/job-1"), env());
    assert.equal(r.status, 200);
    assert.deepEqual(await r.json(), { id: "job-1", status: "IN_PROGRESS", progress, delayTime: 1200, executionTime: 41000, workerId: "w1" });
  });

  test("status when COMPLETED carries the output; a refused job is FAILED", async () => {
    const output = { engine: "h3", model: "fasth3-v2", video_url: "https://r2/jobs/job-1/video.mp4", timings: { total_s: 50 } };
    rp = fakeRunpod({
      "/status/job-1": { id: "job-1", status: "COMPLETED", output, delayTime: 5, executionTime: 50000 },
      "/status/job-2": { id: "job-2", status: "COMPLETED", output: { error: "frames off the grid", where: "validate" } },
      "/status/job-3": { id: "job-3", status: "FAILED", error: "CUDA out of memory" },
      "/status/job-4": { id: "job-4", status: "IN_QUEUE" },
    }); globalThis.fetch = rp;
    const done = await (await worker.fetch(req("/api/jobs/job-1"), env())).json();
    assert.deepEqual(done, { id: "job-1", status: "COMPLETED", output, delayTime: 5, executionTime: 50000 });
    const refused = await (await worker.fetch(req("/api/jobs/job-2"), env())).json();
    assert.equal(refused.status, "FAILED");
    assert.equal(refused.error, "frames off the grid (validate)");
    const failed = await (await worker.fetch(req("/api/jobs/job-3"), env())).json();
    assert.deepEqual(failed, { id: "job-3", status: "FAILED", error: "CUDA out of memory" });
    const queued = await (await worker.fetch(req("/api/jobs/job-4"), env())).json();
    assert.deepEqual(queued, { id: "job-4", status: "IN_QUEUE" });
  });

  test("normaliseStatus maps unknown statuses to IN_QUEUE and keeps a failure message", () => {
    assert.equal(normaliseStatus("x", { status: "weird" }).status, "IN_QUEUE");
    assert.equal(normaliseStatus("x", { status: "TIMED_OUT" }).status, "TIMED_OUT");
    assert.equal(normaliseStatus("x", { status: "FAILED" }).error, "the job failed without a message");
  });

  test("cancel posts to /cancel/:id", async () => {
    rp = fakeRunpod({ "/cancel/job-1": { id: "job-1", status: "CANCELLED" } }); globalThis.fetch = rp;
    const r = await worker.fetch(req("/api/jobs/job-1/cancel", { method: "POST" }), env());
    assert.equal(r.status, 200);
    assert.deepEqual(await r.json(), { id: "job-1", status: "CANCELLED" });
    assert.equal(rp.calls[0].method, "POST");
    assert.equal(rp.calls[0].path, "/cancel/job-1");
  });

  test("a RunPod error status is relayed with its message", async () => {
    rp = fakeRunpod({ "/status/gone": new Response(JSON.stringify({ error: "request does not exist" }), { status: 404 }) }); globalThis.fetch = rp;
    const r = await worker.fetch(req("/api/jobs/gone"), env());
    assert.equal(r.status, 404);
    assert.match((await r.json()).error, /request does not exist/);
  });
});

describe("info", () => {
  test("runs op=info through /runsync with a timeout and returns the output", async () => {
    const out = { gpu: "NVIDIA RTX PRO 6000", driver: "580.1", sidecars: [] };
    rp = fakeRunpod({ "/runsync": { id: "sync-1", status: "COMPLETED", output: out } }); globalThis.fetch = rp;
    const r = await worker.fetch(req("/api/info", { method: "POST" }), env());
    assert.equal(r.status, 200);
    assert.deepEqual(await r.json(), out);
    assert.deepEqual(rp.calls[0].body, { input: { op: "info" } });
    assert.ok(rp.calls[0].signal instanceof AbortSignal, "runsync carries a timeout signal");
  });

  test("an info job that failed is a 502", async () => {
    rp = fakeRunpod({ "/runsync": { id: "sync-1", status: "FAILED", error: "boom" } }); globalThis.fetch = rp;
    const r = await worker.fetch(req("/api/info", { method: "POST" }), env());
    assert.equal(r.status, 502);
    assert.equal((await r.json()).error, "boom");
  });
});

describe("uploads", () => {
  test("a raw body lands in R2 under uploads/<yyyy-mm>/<random>-<name>", async () => {
    const e = env();
    const bytes = new Uint8Array([1, 2, 3, 4, 5]);
    const r = await worker.fetch(req("/api/uploads", { method: "POST", body: bytes.buffer, headers: { "Content-Type": "audio/wav", "X-File-Name": "my clip (1).wav" } }), e);
    assert.equal(r.status, 201);
    const v = await r.json();
    const ym = new Date().toISOString().slice(0, 7);
    assert.match(v.r2, new RegExp(`^uploads/${ym}/[0-9a-f]{16}-my-clip-1-.wav$`));
    assert.equal(v.bytes, 5);
    assert.equal(v.mime, "audio/wav");
    const stored = e.UPLOADS.objects.get(v.r2);
    assert.deepEqual(new Uint8Array(stored.body), bytes);
    assert.equal(stored.opts.httpMetadata.contentType, "audio/wav");
    assert.equal(rp.calls.length, 0, "uploads never touch RunPod");
  });

  test("a multipart body with one file works too", async () => {
    const e = env();
    const form = new FormData();
    form.append("file", new File([new Uint8Array(10)], "draft.mp4", { type: "video/mp4" }));
    const r = await worker.fetch(req("/api/uploads", { method: "POST", body: form }), e);
    assert.equal(r.status, 201);
    const v = await r.json();
    assert.match(v.r2, /^uploads\/\d{4}-\d{2}\/[0-9a-f]{16}-draft\.mp4$/);
    assert.equal(v.bytes, 10);
    assert.equal(v.mime, "video/mp4");
  });

  test("the size cap and empty uploads are refused", async () => {
    const e = env();
    const big = new Request(ORIGIN + "/api/uploads", { method: "POST", headers: { Authorization: `Bearer ${TOKEN}`, "Content-Type": "video/mp4", "X-File-Name": "x.mp4", "Content-Length": String(600 * 1024 * 1024) }, body: new Uint8Array(1) });
    assert.equal((await worker.fetch(big, e)).status, 413);
    const empty = await worker.fetch(req("/api/uploads", { method: "POST", body: new ArrayBuffer(0), headers: { "Content-Type": "image/png", "X-File-Name": "e.png" } }), e);
    assert.equal(empty.status, 400);
    assert.equal(e.UPLOADS.objects.size, 0);
  });

  test("uploadKey sanitises the name and keeps the month", () => {
    const k = uploadKey("Ünïcode name!.PNG", new Date(Date.UTC(2026, 10, 18)), "abc123");
    assert.equal(k, "uploads/2026-11/abc123-Unicode-name-.PNG");
    assert.equal(uploadKey("../../etc/passwd", new Date(Date.UTC(2026, 0, 1)), "r"), "uploads/2026-01/r-etc-passwd");
  });
});

// ---- the pod backend ---------------------------------------------------------------
// A fetch that answers RunPod's queue API (api.runpod.ai), its REST v2
// (api.runpod.io) and the pod itself (<id>-8000.proxy.runpod.net).
function fakeWorld({ queue = {}, rest = {}, pod = {} } = {}) {
  const calls = [];
  const fn = async (url, init = {}) => {
    const u = new URL(url);
    const auth = init.headers?.Authorization;
    let table, path;
    if (u.origin === "https://api.runpod.ai") { assert.equal(auth, "Bearer rp_key"); table = queue; path = u.pathname.replace("/v2/ep123", ""); }
    else if (u.origin === "https://api.runpod.io") { assert.equal(auth, "Bearer rp_key"); table = rest; path = u.pathname.replace("/v2", ""); }
    else if (/-8000\.proxy\.runpod\.net$/.test(u.host)) { table = pod; path = u.pathname; }
    else throw new Error("unexpected host " + u.host);
    const call = { host: u.host, path, method: init.method || "GET", body: init.body ? JSON.parse(init.body) : undefined, auth };
    calls.push(call);
    const hit = Object.entries(table).find(([p]) => path === p || path.startsWith(p));
    if (!hit) return new Response(JSON.stringify({ error: "no route " + path }), { status: 404 });
    const v = typeof hit[1] === "function" ? hit[1](call) : hit[1];
    if (v instanceof Response) return v;
    return new Response(JSON.stringify(v), { status: 200, headers: { "Content-Type": "application/json" } });
  };
  fn.calls = calls;
  return fn;
}

describe("pod backend", () => {
  const podEnvVars = { POD_IMAGE: "ghcr.io/x/studio:sha-1", STUDIO_R2_ENDPOINT: "https://acct.r2.cloudflarestorage.com", CF_ACCOUNT_ID: "cf1", CF_STREAM_API_TOKEN: "cf-tok" };

  test("the create body is RunPod REST v2's shape with the serve command and the endpoint's env", () => {
    const e = env(podEnvVars);
    const body = podCreateBody(e, { image: "img", gpu: "GPU X", dc: "EU-RO-1", volume: "vol1", disk: 300, hours: 4, minCuda: "13.0" }, "tok", "studio-pod-1");
    assert.deepEqual(Object.keys(body).sort(), ["cloud", "cmd", "dataCenterIds", "disk", "env", "gpu", "image", "mounts", "name", "ports"]);
    assert.deepEqual(body.gpu, { id: "GPU X", count: 1, minCudaVersion: "13.0" });
    assert.deepEqual(body.cmd, ["serve"]);
    assert.deepEqual(body.mounts, { network: [{ volumeId: "vol1", path: "/runpod-volume" }] });
    assert.equal(body.env.STUDIO_SERVE_TOKEN, "tok");
    assert.equal(body.env.STUDIO_POD_HOURS, "4");
    assert.equal(body.env.RUNPOD_API_KEY, "rp_key");
    assert.equal(body.env.FV_R2_ACCESS_KEY_ID, "{{ RUNPOD_SECRET_fv_r2_access_key_id }}");
    assert.equal(body.env.STUDIO_R2_ENDPOINT, "https://acct.r2.cloudflarestorage.com");
    assert.equal(body.env.FV_CF_STREAM_API_TOKEN, "cf-tok");
    assert.equal(podEnv(env(), { image: "i", hours: 2 }, "t").FV_CF_ACCOUNT_ID, undefined);
    assert.equal(podUrl("abc"), "https://abc-8000.proxy.runpod.net");
  });

  test("no pod: status says none, jobs for the pod are 409, delete is 404", async () => {
    const e = env(podEnvVars);
    globalThis.fetch = fakeWorld();
    let r = await worker.fetch(req("/api/pod"), e);
    assert.deepEqual(await r.json(), { state: "none" });
    r = await worker.fetch(req("/api/jobs", { method: "POST", body: { backend: "pod", engine: "h3", model: catalog.models[0].id, prompt: "a fox" } }), e);
    assert.equal(r.status, 409);
    r = await worker.fetch(req("/api/pod", { method: "DELETE" }), e);
    assert.equal(r.status, 404);
    r = await worker.fetch(req("/api/jobs", { method: "POST", body: { backend: "nope", engine: "h3", model: catalog.models[0].id, prompt: "a fox" } }), e);
    assert.equal(r.status, 400);
  });

  test("create records the pod, status joins RunPod's state with the pod's health, delete terminates", async () => {
    const e = env(podEnvVars);
    const world = fakeWorld({
      rest: {
        "/pods/p1": (c) => c.method === "DELETE" ? new Response(null, { status: 204 }) : { id: "p1", status: "RUNNING", cost: 3.69, dataCenterId: "EU-RO-1", gpu: { id: "NVIDIA RTX PRO 6000 Blackwell Server Edition", count: 1 } },
        "/pods": (c) => { assert.equal(c.method, "POST"); return { id: "p1", status: "PROVISIONING", cost: 3.69 }; },
      },
      pod: { "/health": { ok: true, mode: "pod", busy: false, queued: 0, jobs_done: 2 } },
    });
    globalThis.fetch = world;
    let r = await worker.fetch(req("/api/pod", { method: "POST", body: { hours: 2 } }), e);
    const created = await r.json();
    assert.equal(r.status, 201, JSON.stringify(created));
    assert.equal(created.id, "p1"); assert.equal(created.state, "provisioning"); assert.equal(created.hours, 2);
    const create = world.calls.find((c) => c.path === "/pods" && c.method === "POST");
    assert.equal(create.body.image, "ghcr.io/x/studio:sha-1");
    assert.equal(create.body.env.STUDIO_POD_HOURS, "2");
    assert.match(create.body.name, /^studio-pod-\d{12}$/);
    const rec = JSON.parse(new TextDecoder().decode(e.UPLOADS.objects.get("studio/pod.json").body));
    assert.equal(rec.id, "p1"); assert.equal(rec.url, "https://p1-8000.proxy.runpod.net"); assert.equal(rec.token.length, 32);
    // A second create while one exists is 409.
    r = await worker.fetch(req("/api/pod", { method: "POST" }), e);
    assert.equal(r.status, 409);
    // Status.
    r = await worker.fetch(req("/api/pod"), e);
    const st = await r.json();
    assert.equal(st.state, "running"); assert.equal(st.ready, true); assert.equal(st.health.jobs_done, 2); assert.equal(st.cost, 3.69);
    assert.equal(world.calls.find((c) => c.path === "/health").auth, `Bearer ${rec.token}`);
    assert.equal(st.token, undefined);
    // Health carries the pod summary.
    globalThis.fetch = fakeWorld({ queue: { "/health": { workers: { idle: 1 }, jobs: {} } }, rest: { "/pods/p1": { id: "p1", status: "EXITED" } } });
    r = await worker.fetch(req("/api/health"), e);
    assert.equal((await r.json()).pod.state, "exited");
    // Delete.
    globalThis.fetch = world;
    r = await worker.fetch(req("/api/pod", { method: "DELETE" }), e);
    assert.deepEqual(await r.json(), { state: "terminated", id: "p1" });
    assert.equal(e.UPLOADS.objects.has("studio/pod.json"), false);
  });

  test("pod jobs go to the pod with its token; ids are pod_…; status and cancel follow", async () => {
    const e = env(podEnvVars);
    await e.UPLOADS.put("studio/pod.json", JSON.stringify({ id: "p1", url: "https://p1-8000.proxy.runpod.net", token: "podtok" }));
    const world = fakeWorld({
      pod: {
        "/jobs/j1/cancel": { id: "j1", status: "CANCELLED" },
        "/jobs/j1": { id: "j1", status: "IN_PROGRESS", output: { stage: "denoising", elapsed_s: 3 } },
        "/jobs": (c) => { assert.equal(c.method, "POST"); assert.equal(c.body.input.backend, undefined); assert.equal(c.body.input.prompt, "a fox"); return { id: "j1", status: "IN_QUEUE" }; },
      },
    });
    globalThis.fetch = world;
    let r = await worker.fetch(req("/api/jobs", { method: "POST", body: { backend: "pod", engine: "h3", model: catalog.models[0].id, prompt: "a fox" } }), e);
    const sub = await r.json();
    assert.equal(r.status, 202, JSON.stringify(sub));
    assert.deepEqual(sub, { id: "pod_j1", status: "IN_QUEUE", backend: "pod" });
    assert.equal(world.calls.at(-1).auth, "Bearer podtok");
    r = await worker.fetch(req("/api/jobs/pod_j1"), e);
    const s = await r.json();
    assert.equal(s.id, "pod_j1"); assert.equal(s.status, "IN_PROGRESS"); assert.equal(s.progress.stage, "denoising"); assert.equal(s.backend, "pod");
    r = await worker.fetch(req("/api/jobs/pod_j1/cancel", { method: "POST" }), e);
    assert.deepEqual(await r.json(), { id: "pod_j1", status: "CANCELLED", backend: "pod" });
    // Serverless jobs still go to the queue endpoint.
    globalThis.fetch = fakeWorld({ queue: { "/run": { id: "q1", status: "IN_QUEUE" } } });
    r = await worker.fetch(req("/api/jobs", { method: "POST", body: { engine: "h3", model: catalog.models[0].id, prompt: "a fox" } }), e);
    assert.deepEqual(await r.json(), { id: "q1", status: "IN_QUEUE", backend: "serverless" });
  });
});

describe("live streams on the pod", () => {
  const podEnvVars = { POD_IMAGE: "ghcr.io/x/studio:sha-1" };
  const fvModel = catalog.models.find((m) => m.engine === "fastvideo").id;

  test("no pod: 409; bad bodies: 400 before the pod is asked", async () => {
    const e = env(podEnvVars);
    globalThis.fetch = fakeWorld();
    let r = await worker.fetch(req("/api/streams", { method: "POST", body: { model: fvModel, prompt: "a fox" } }), e);
    assert.equal(r.status, 409);
    r = await worker.fetch(req("/api/streams", { method: "POST", body: { model: "h3-plug4", prompt: "a fox" } }), e);
    assert.equal(r.status, 400);
    r = await worker.fetch(req("/api/streams", { method: "POST", body: { model: fvModel } }), e);
    assert.equal(r.status, 400);
    r = await worker.fetch(req("/api/streams/s1/commands", { method: "POST", body: { nope: 1 } }), e);
    assert.equal(r.status, 400);
  });

  test("open, status, command and stop go to the pod with its token", async () => {
    const e = env(podEnvVars);
    await e.UPLOADS.put("studio/pod.json", JSON.stringify({ id: "p1", url: "https://p1-8000.proxy.runpod.net", token: "podtok" }));
    const obj = { id: "fvstream_1", mode: "clip", status: { state: "running", output_state: "publishing" }, playback: { whep_url: "https://cf/whep" } };
    const world = fakeWorld({
      pod: {
        "/streams/fvstream_1/commands": (c) => { assert.equal(c.body.type, "set_prompt"); return { reply: { ok: true } }; },
        "/streams/fvstream_1": (c) => c.method === "DELETE" ? new Response(null, { status: 204 }) : obj,
        "/streams": (c) => { assert.equal(c.method, "POST"); assert.equal(c.body.model, fvModel); assert.equal(c.body.max_seconds, 60); return new Response(JSON.stringify(obj), { status: 201, headers: { "Content-Type": "application/json" } }); },
      },
    });
    globalThis.fetch = world;
    let r = await worker.fetch(req("/api/streams", { method: "POST", body: { model: fvModel, prompt: "a fox", max_seconds: 60 } }), e);
    const opened = await r.json();
    assert.equal(r.status, 201, JSON.stringify(opened));
    assert.equal(opened.playback.whep_url, "https://cf/whep");
    assert.equal(world.calls.at(-1).auth, "Bearer podtok");
    r = await worker.fetch(req("/api/streams/fvstream_1"), e);
    assert.equal((await r.json()).status.state, "running");
    r = await worker.fetch(req("/api/streams/fvstream_1/commands", { method: "POST", body: { type: "set_prompt", data: { prompt: "dusk" } } }), e);
    assert.equal((await r.json()).reply.ok, true);
    r = await worker.fetch(req("/api/streams/fvstream_1", { method: "DELETE" }), e);
    assert.deepEqual(await r.json(), { id: "fvstream_1", state: "closed" });
  });
});

describe("segments on the pod", () => {
  const podEnvVars = { POD_IMAGE: "ghcr.io/x/studio:sha-1" };
  const fvModel = catalog.models.find((m) => m.engine === "fastvideo").id;

  test("validation before the pod; no pod is 409", async () => {
    const e = env(podEnvVars);
    globalThis.fetch = fakeWorld();
    const ok = { stream_id: "str_1", segment_index: 0, model: fvModel, prompt: "a harbour", duration_sec: 6 };
    let r = await worker.fetch(req("/api/segments", { method: "POST", body: ok }), e);
    assert.equal(r.status, 409);
    for (const bad of [{ ...ok, stream_id: "no spaces" }, { ...ok, segment_index: -1 }, { ...ok, model: "h3-plug4" }, { ...ok, duration_sec: 99 }, { ...ok, prompt: "" }]) {
      r = await worker.fetch(req("/api/segments", { method: "POST", body: bad }), e);
      assert.equal(r.status, 400, JSON.stringify(bad));
    }
  });

  test("next, status, session and close go to the pod", async () => {
    const e = env(podEnvVars);
    await e.UPLOADS.put("studio/pod.json", JSON.stringify({ id: "p1", url: "https://p1-8000.proxy.runpod.net", token: "podtok" }));
    const world = fakeWorld({
      pod: {
        "/segments/streams/str_1": (c) => c.method === "DELETE" ? new Response(null, { status: 204 }) : { stream_id: "str_1", path: "causal", segments: 1 },
        "/segments/seg_1": { id: "seg_1", status: "succeeded", video_url: "https://r2/segments/str_1/0000-seg_1.mp4?sig" },
        "/segments": (c) => { assert.equal(c.method, "POST"); assert.equal(c.body.segment_index, 0); return new Response(JSON.stringify({ id: "seg_1", status: "queued" }), { status: 202, headers: { "Content-Type": "application/json" } }); },
      },
    });
    globalThis.fetch = world;
    let r = await worker.fetch(req("/api/segments", { method: "POST", body: { stream_id: "str_1", segment_index: 0, model: fvModel, prompt: "a harbour", duration_sec: 6 } }), e);
    const a = await r.json();
    assert.equal(r.status, 202, JSON.stringify(a));
    assert.equal(a.id, "seg_1");
    assert.equal(world.calls.at(-1).auth, "Bearer podtok");
    r = await worker.fetch(req("/api/segments/seg_1"), e);
    assert.equal((await r.json()).video_url, "https://r2/segments/str_1/0000-seg_1.mp4?sig");
    r = await worker.fetch(req("/api/segments/streams/str_1"), e);
    assert.equal((await r.json()).path, "causal");
    r = await worker.fetch(req("/api/segments/streams/str_1", { method: "DELETE" }), e);
    assert.deepEqual(await r.json(), { stream_id: "str_1", state: "closed" });
  });
});
