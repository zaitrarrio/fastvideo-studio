// FastVideo Studio Worker: serves the page from the asset store and proxies
// the RunPod endpoint behind one shared token. No framework, no runtime deps.
//
//   GET  /api/catalog           the catalog (public)
//   GET  /api/health            the endpoint's workers / queue counts
//   POST /api/jobs              body = jobs.md `input` → RunPod /run
//   GET  /api/jobs/:id          RunPod /status/:id, normalised
//   POST /api/jobs/:id/cancel   RunPod /cancel/:id
//   POST /api/uploads           a file into R2 under uploads/<yyyy-mm>/…
//   POST /api/info              RunPod /runsync with {"op": "info"}
//   GET  /api/pod               the pod backend (docs/pod-mode.md): its record, RunPod state, health
//   POST /api/pod               create the pod (RunPod REST v2 POST /v2/pods, the worker image in `serve` mode)
//   DELETE /api/pod             terminate it
//
// A job input with `backend: "pod"` goes to the pod's /jobs instead of the
// queue endpoint; its id is `pod_<id>` and status / cancel follow it.
//
// Secrets: RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID, STUDIO_TOKEN; for the pod
// CF_ACCOUNT_ID + CF_STREAM_API_TOKEN (live streams). Vars: ENDPOINT_LABEL,
// POD_IMAGE, POD_GPU, POD_DC, POD_VOLUME, POD_DISK_GB, POD_HOURS, POD_MIN_CUDA,
// STUDIO_R2_ENDPOINT, STUDIO_R2_BUCKET.
import catalog from "../catalog/catalog.json" with { type: "json" };

const RUNPOD = "https://api.runpod.ai/v2";
const RUNPOD_REST = "https://api.runpod.io/v2"; // REST v2 (v1 at rest.runpod.io is retired)
const POD_RECORD = "studio/pod.json";
const POD_DEFAULTS = { gpu: "NVIDIA RTX PRO 6000 Blackwell Server Edition", dc: "EU-RO-1", volume: "bz0ofkfx45", disk: 300, hours: 4, minCuda: "13.0" };
const POD_HEALTH_TIMEOUT_MS = 8_000;
const EXECUTION_TIMEOUT_MS = 1_800_000; // 30 min: the longest jobs (Regen-2K) run ~8 min
const INFO_TIMEOUT_MS = 60_000;
const STATUSES = new Set(["IN_QUEUE", "IN_PROGRESS", "COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"]);
const UPLOAD_LIMIT = catalog.upload_limits?.r2_upload_bytes ?? 536_870_912;

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (!url.pathname.startsWith("/api/")) return env.ASSETS ? env.ASSETS.fetch(request) : text("not found", 404);
    try {
      return await handleApi(request, env, url);
    } catch (e) {
      if (e instanceof HttpError) return json({ error: e.message }, e.status);
      return json({ error: `studio: ${e?.message || e}` }, 502);
    }
  },
};

class HttpError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}

// ---- routing ---------------------------------------------------------------
async function handleApi(request, env, url) {
  const path = url.pathname.replace(/\/+$/, "");
  const method = request.method.toUpperCase();
  if (method === "OPTIONS") return new Response(null, { status: 204, headers: noStore() });
  if (path === "/api/catalog") return method === "GET" ? json(catalog) : methodNotAllowed("GET");
  requireToken(request, env);
  if (method === "POST") requireSameOrigin(request, url);

  if (path === "/api/health") return method === "GET" ? health(env) : methodNotAllowed("GET");
  if (path === "/api/jobs") return method === "POST" ? submitJob(request, env) : methodNotAllowed("POST");
  if (path === "/api/uploads") return method === "POST" ? upload(request, env) : methodNotAllowed("POST");
  if (path === "/api/info") return method === "POST" ? info(env) : methodNotAllowed("POST");
  if (path === "/api/pod") {
    if (method === "GET") return podStatus(env);
    if (method === "POST") return podCreate(request, env);
    if (method === "DELETE") return podDelete(env);
    return methodNotAllowed("GET, POST, DELETE");
  }
  const job = path.match(/^\/api\/jobs\/([A-Za-z0-9_-]+)(\/cancel)?$/);
  if (job && !job[2]) return method === "GET" ? jobStatus(job[1], env) : methodNotAllowed("GET");
  if (job) return method === "POST" ? cancelJob(job[1], env) : methodNotAllowed("POST");
  throw new HttpError(404, "no such route");
}

// ---- guards ----------------------------------------------------------------
function requireToken(request, env) {
  const expected = env.STUDIO_TOKEN;
  if (!expected) throw new HttpError(503, "STUDIO_TOKEN is not set on this Worker");
  const given = (request.headers.get("Authorization") || "").replace(/^Bearer\s+/i, "");
  if (!given || !sameSecret(given, expected)) throw new HttpError(401, "unlock the studio with its token");
}

// Constant-time string compare (the Worker has no timingSafeEqual on strings).
function sameSecret(a, b) {
  const x = new TextEncoder().encode(a), y = new TextEncoder().encode(b);
  let diff = x.length ^ y.length;
  for (let i = 0; i < Math.max(x.length, y.length); i++) diff |= (x[i % x.length] ?? 0) ^ (y[i % y.length] ?? 0);
  return diff === 0;
}

// The engines' rule: a cross-site form POST skips the preflight, so a foreign
// Origin is refused outright. Same-origin and origin-less (curl) requests pass.
function requireSameOrigin(request, url) {
  const origin = request.headers.get("Origin");
  if (origin && origin !== url.origin) throw new HttpError(403, `foreign origin ${origin}`);
}

function requireEndpoint(env) {
  if (!env.RUNPOD_API_KEY || !env.RUNPOD_ENDPOINT_ID) throw new HttpError(503, "RUNPOD_API_KEY / RUNPOD_ENDPOINT_ID are not set on this Worker");
}

// ---- RunPod ---------------------------------------------------------------
async function runpod(env, path, { method = "GET", body, timeoutMs } = {}) {
  requireEndpoint(env);
  const init = {
    method,
    headers: { Authorization: `Bearer ${env.RUNPOD_API_KEY}`, Accept: "application/json" },
  };
  if (body !== undefined) { init.headers["Content-Type"] = "application/json"; init.body = JSON.stringify(body); }
  if (timeoutMs) init.signal = AbortSignal.timeout(timeoutMs);
  let r;
  try { r = await fetch(`${RUNPOD}/${env.RUNPOD_ENDPOINT_ID}${path}`, init); }
  catch (e) { throw new HttpError(502, `RunPod unreachable: ${e?.name === "TimeoutError" ? "timed out" : e?.message || e}`); }
  const textBody = await r.text();
  let data;
  try { data = textBody ? JSON.parse(textBody) : {}; } catch { data = { raw: textBody }; }
  if (!r.ok) throw new HttpError(r.status === 401 || r.status === 403 ? 502 : r.status, `RunPod ${path} answered ${r.status}: ${data.error || data.raw || textBody}`.slice(0, 400));
  return data;
}

async function health(env) {
  const [h, pod] = await Promise.all([runpod(env, "/health"), podSummary(env).catch((e) => ({ error: e.message }))]);
  return json({ label: env.ENDPOINT_LABEL || "", workers: h.workers || {}, jobs: h.jobs || {}, pod });
}

async function submitJob(request, env) {
  const input = await readJson(request);
  if (!input || typeof input !== "object" || Array.isArray(input)) throw new HttpError(400, "the body must be a JSON object (the job input)");
  const backend = input.backend ?? "serverless";
  delete input.backend;
  if (backend !== "serverless" && backend !== "pod") throw new HttpError(400, `backend must be serverless or pod, not ${backend}`);
  if ((input.op ?? "generate") === "generate") {
    if (typeof input.prompt !== "string" || !input.prompt.trim()) throw new HttpError(400, "prompt is required");
    if (input.prompt.length > 2000) throw new HttpError(400, "prompt is over 2000 characters");
    if (!catalog.engines[input.engine]) throw new HttpError(400, `unknown engine ${input.engine}`);
    if (!catalog.models.some((m) => m.engine === input.engine && m.id === input.model)) throw new HttpError(400, `unknown model ${input.model} for ${input.engine}`);
  }
  if (backend === "pod") {
    const rec = await podRecord(env);
    if (!rec) throw new HttpError(409, "no pod is running: start one first (POST /api/pod)");
    const r = await pod(env, rec, "/jobs", { method: "POST", body: { input } });
    if (!r.id) throw new HttpError(502, "the pod gave no job id");
    return json({ id: `pod_${r.id}`, status: normalStatus(r.status), backend: "pod" }, 202);
  }
  const r = await runpod(env, "/run", { method: "POST", body: { input, policy: { executionTimeout: EXECUTION_TIMEOUT_MS } } });
  if (!r.id) throw new HttpError(502, "RunPod /run gave no job id");
  return json({ id: r.id, status: normalStatus(r.status), backend: "serverless" }, 202);
}

async function jobStatus(id, env) {
  if (id.startsWith("pod_")) {
    const rec = await podRecord(env);
    if (!rec) throw new HttpError(404, "no pod is running: the job is gone with it");
    const r = await pod(env, rec, `/jobs/${id.slice(4)}`);
    const s = normaliseStatus(id.slice(4), r);
    s.id = id; s.backend = "pod";
    return json(s);
  }
  const r = await runpod(env, `/status/${id}`);
  return json(normaliseStatus(id, r));
}

async function cancelJob(id, env) {
  if (id.startsWith("pod_")) {
    const rec = await podRecord(env);
    if (!rec) throw new HttpError(404, "no pod is running");
    const r = await pod(env, rec, `/jobs/${id.slice(4)}/cancel`, { method: "POST" });
    return json({ id, status: normalStatus(r.status || "CANCELLED"), backend: "pod" });
  }
  const r = await runpod(env, `/cancel/${id}`, { method: "POST" });
  return json({ id, status: normalStatus(r.status || "CANCELLED") });
}

// ---- the pod backend (docs/pod-mode.md) -------------------------------------
// One pod at a time, its record in the bucket. RunPod's REST v2 creates and
// terminates it; the pod's own `serve` answers jobs with its per-pod token.
async function runpodRest(env, path, { method = "GET", body, timeoutMs = 60_000 } = {}) {
  if (!env.RUNPOD_API_KEY) throw new HttpError(503, "RUNPOD_API_KEY is not set on this Worker");
  const init = { method, headers: { Authorization: `Bearer ${env.RUNPOD_API_KEY}`, Accept: "application/json" }, signal: AbortSignal.timeout(timeoutMs) };
  if (body !== undefined) { init.headers["Content-Type"] = "application/json"; init.body = JSON.stringify(body); }
  let r;
  try { r = await fetch(`${RUNPOD_REST}${path}`, init); }
  catch (e) { throw new HttpError(502, `RunPod unreachable: ${e?.name === "TimeoutError" ? "timed out" : e?.message || e}`); }
  const textBody = await r.text();
  let data;
  try { data = textBody ? JSON.parse(textBody) : {}; } catch { data = { raw: textBody }; }
  if (!r.ok) throw new HttpError(r.status === 401 || r.status === 403 ? 502 : r.status, `RunPod ${method} ${path} answered ${r.status}: ${data.error || data.message || data.raw || textBody}`.slice(0, 400));
  return data;
}

async function pod(env, rec, path, { method = "GET", body, timeoutMs = 60_000 } = {}) {
  const init = { method, headers: { Authorization: `Bearer ${rec.token}`, Accept: "application/json" }, signal: AbortSignal.timeout(timeoutMs) };
  if (body !== undefined) { init.headers["Content-Type"] = "application/json"; init.body = JSON.stringify(body); }
  let r;
  try { r = await fetch(`${rec.url}${path}`, init); }
  catch (e) { throw new HttpError(502, `the pod is unreachable: ${e?.name === "TimeoutError" ? "timed out" : e?.message || e}`); }
  const textBody = await r.text();
  let data;
  try { data = textBody ? JSON.parse(textBody) : {}; } catch { data = { raw: textBody }; }
  if (!r.ok) throw new HttpError(r.status === 401 ? 502 : r.status, `the pod answered ${r.status} on ${path}: ${data.error || data.raw || textBody}`.slice(0, 400));
  return data;
}

async function podRecord(env) {
  if (!env.UPLOADS) throw new HttpError(503, "no UPLOADS bucket is bound to this Worker");
  const obj = await env.UPLOADS.get(POD_RECORD);
  if (!obj) return null;
  try { return JSON.parse(await obj.text()); } catch { return null; }
}

async function savePodRecord(env, rec) {
  await env.UPLOADS.put(POD_RECORD, JSON.stringify(rec), { httpMetadata: { contentType: "application/json" } });
}

export function podUrl(id) { return `https://${id}-8000.proxy.runpod.net`; }

function podSettings(env) {
  return {
    image: env.POD_IMAGE || "",
    gpu: env.POD_GPU || POD_DEFAULTS.gpu,
    dc: env.POD_DC || POD_DEFAULTS.dc,
    volume: env.POD_VOLUME || POD_DEFAULTS.volume,
    disk: Number(env.POD_DISK_GB) || POD_DEFAULTS.disk,
    hours: Number(env.POD_HOURS) || POD_DEFAULTS.hours,
    minCuda: env.POD_MIN_CUDA || POD_DEFAULTS.minCuda,
  };
}

// The pod's env: the serverless endpoint's (the account's R2 secrets by
// reference, the image), the per-pod serve token, the cap, the key the pod
// terminates itself with, and Cloudflare Stream for live streams.
export function podEnv(env, settings, token) {
  const e = {
    FV_R2_ENDPOINT: "{{ RUNPOD_SECRET_fv_r2_endpoint }}",
    FV_R2_ACCESS_KEY_ID: "{{ RUNPOD_SECRET_fv_r2_access_key_id }}",
    FV_R2_SECRET_ACCESS_KEY: "{{ RUNPOD_SECRET_fv_r2_secret_access_key }}",
    FV_R2_BUCKET: "{{ RUNPOD_SECRET_fv_r2_bucket }}",
    STUDIO_R2_ACCESS_KEY_ID: "{{ RUNPOD_SECRET_studio_r2_access_key_id }}",
    STUDIO_R2_SECRET_ACCESS_KEY: "{{ RUNPOD_SECRET_studio_r2_secret_access_key }}",
    STUDIO_R2_BUCKET: env.STUDIO_R2_BUCKET || "fastvideo-studio",
    STUDIO_IMAGE: settings.image,
    STUDIO_SERVE_TOKEN: token,
    STUDIO_POD_HOURS: String(settings.hours),
    RUNPOD_API_KEY: env.RUNPOD_API_KEY,
  };
  if (env.STUDIO_R2_ENDPOINT) e.STUDIO_R2_ENDPOINT = env.STUDIO_R2_ENDPOINT;
  if (env.CF_ACCOUNT_ID && env.CF_STREAM_API_TOKEN) { e.FV_CF_ACCOUNT_ID = env.CF_ACCOUNT_ID; e.FV_CF_STREAM_API_TOKEN = env.CF_STREAM_API_TOKEN; }
  return e;
}

// RunPod REST v2 `POST /v2/pods` (CreatePodRequest).
export function podCreateBody(env, settings, token, name) {
  return {
    name,
    image: settings.image,
    cloud: "SECURE",
    gpu: { id: settings.gpu, count: 1, minCudaVersion: settings.minCuda },
    disk: settings.disk,
    ports: ["8000/http"],
    cmd: ["serve"],
    env: podEnv(env, settings, token),
    mounts: { network: [{ volumeId: settings.volume, path: "/runpod-volume" }] },
    dataCenterIds: [settings.dc],
  };
}

async function podSummary(env) {
  const rec = await podRecord(env);
  if (!rec) return { state: "none" };
  const out = { state: "unknown", id: rec.id, image: rec.image, created_at: rec.created_at, hours: rec.hours, gpu: rec.gpu };
  try {
    const p = await runpodRest(env, `/pods/${rec.id}`, { timeoutMs: 15_000 });
    out.state = String(p.status || "unknown").toLowerCase();
    out.cost = p.cost; out.dc = p.dataCenterId; out.gpu = p.gpu?.id || rec.gpu;
  } catch (e) {
    if (e instanceof HttpError && e.status === 404) out.state = "gone";
    else out.error = e.message;
  }
  if (out.state === "running") {
    try { out.health = await pod(env, rec, "/health", { timeoutMs: POD_HEALTH_TIMEOUT_MS }); out.ready = true; }
    catch (e) { out.ready = false; out.health_error = e.message; }
  }
  return out;
}

async function podStatus(env) {
  return json(await podSummary(env));
}

async function podCreate(request, env) {
  const settings = podSettings(env);
  const body = (await readJson(request).catch(() => null)) || {};
  if (body.hours != null) { const h = Number(body.hours); if (!(h > 0 && h <= 24)) throw new HttpError(400, "hours must be in (0, 24]"); settings.hours = h; }
  if (!settings.image) throw new HttpError(503, "POD_IMAGE is not set on this Worker");
  const existing = await podSummary(env);
  if (existing.state !== "none" && existing.state !== "gone" && existing.state !== "terminated") throw new HttpError(409, `a pod is already ${existing.state} (${existing.id}); stop it first`);
  const token = randomId() + randomId();
  const name = `studio-pod-${new Date().toISOString().slice(0, 16).replace(/[-:T]/g, "")}`;
  const p = await runpodRest(env, "/pods", { method: "POST", body: podCreateBody(env, settings, token, name), timeoutMs: 90_000 });
  if (!p.id) throw new HttpError(502, "RunPod gave the pod no id");
  const rec = { id: p.id, url: podUrl(p.id), token, image: settings.image, gpu: settings.gpu, dc: settings.dc, hours: settings.hours, created_at: new Date().toISOString() };
  await savePodRecord(env, rec);
  return json({ state: String(p.status || "provisioning").toLowerCase(), id: p.id, image: rec.image, gpu: rec.gpu, dc: rec.dc, hours: rec.hours, created_at: rec.created_at, cost: p.cost }, 201);
}

async function podDelete(env) {
  const rec = await podRecord(env);
  if (!rec) throw new HttpError(404, "no pod is running");
  let state = "terminated";
  try { await runpodRest(env, `/pods/${rec.id}`, { method: "DELETE", timeoutMs: 60_000 }); }
  catch (e) { if (!(e instanceof HttpError && e.status === 404)) throw e; state = "gone"; }
  await env.UPLOADS.delete(POD_RECORD);
  return json({ state, id: rec.id });
}

async function info(env) {
  const r = await runpod(env, "/runsync", { method: "POST", body: { input: { op: "info" } }, timeoutMs: INFO_TIMEOUT_MS });
  const s = normaliseStatus(r.id || "info", r);
  if (s.status !== "COMPLETED") throw new HttpError(502, s.error || `info job ${s.status}`);
  return json(s.output || {});
}

function normalStatus(s) {
  const u = String(s || "").toUpperCase();
  return STATUSES.has(u) ? u : "IN_QUEUE";
}

// RunPod's /status: `output` holds the SDK's last progress_update while the
// job runs and the handler's result once it is COMPLETED. A handler that
// refuses a job returns {"error", "where"} as its output: that is a failure.
export function normaliseStatus(id, r) {
  const out = { id: r.id || id, status: normalStatus(r.status) };
  const output = r.output && typeof r.output === "object" ? r.output : null;
  if (out.status === "IN_PROGRESS" || out.status === "IN_QUEUE") {
    if (output && (output.stage || output.elapsed_s != null)) out.progress = output;
  } else if (out.status === "COMPLETED") {
    if (output && output.error && !output.video_url) {
      out.status = "FAILED";
      out.error = output.where ? `${output.error} (${output.where})` : String(output.error);
    } else if (output) out.output = output;
  }
  if (r.error) out.error = typeof r.error === "string" ? r.error : JSON.stringify(r.error);
  if (out.status === "FAILED" && !out.error) out.error = "the job failed without a message";
  if (r.delayTime != null) out.delayTime = r.delayTime;
  if (r.executionTime != null) out.executionTime = r.executionTime;
  if (r.workerId) out.workerId = r.workerId;
  return out;
}

// ---- uploads ---------------------------------------------------------------
// Raw body + X-File-Name (what the page sends), or multipart with one file.
async function upload(request, env) {
  if (!env.UPLOADS) throw new HttpError(503, "no UPLOADS bucket is bound to this Worker");
  const ctype = request.headers.get("Content-Type") || "application/octet-stream";
  let name, mime, body, bytes;
  if (/^multipart\/form-data/i.test(ctype)) {
    const form = await request.formData();
    const file = [...form.values()].find((v) => typeof v !== "string");
    if (!file) throw new HttpError(400, "the multipart body holds no file");
    name = file.name; mime = file.type || "application/octet-stream"; bytes = file.size;
    if (bytes > UPLOAD_LIMIT) throw new HttpError(413, `the file is over ${UPLOAD_LIMIT} bytes`);
    body = await file.arrayBuffer();
  } else {
    name = decodeURIComponent(request.headers.get("X-File-Name") || "upload.bin");
    mime = ctype.split(";")[0].trim();
    const len = Number(request.headers.get("Content-Length"));
    if (Number.isFinite(len) && len > 0) {
      if (len > UPLOAD_LIMIT) throw new HttpError(413, `the file is over ${UPLOAD_LIMIT} bytes`);
      bytes = len; body = request.body;
    } else {
      body = await request.arrayBuffer(); bytes = body.byteLength;
      if (bytes > UPLOAD_LIMIT) throw new HttpError(413, `the file is over ${UPLOAD_LIMIT} bytes`);
    }
  }
  if (!bytes) throw new HttpError(400, "the upload is empty");
  const key = uploadKey(name, new Date());
  await env.UPLOADS.put(key, body, { httpMetadata: { contentType: mime }, customMetadata: { name } });
  return json({ r2: key, bytes, mime, name }, 201);
}

export function uploadKey(name, now = new Date(), random = randomId()) {
  const safe = String(name).normalize("NFKD").replace(/[\u0300-\u036f]/g, "").replace(/[^\w.-]+/g, "-").replace(/^[-.]+|[-.]+$/g, "").slice(0, 80) || "upload";
  const ym = `${now.getUTCFullYear()}-${String(now.getUTCMonth() + 1).padStart(2, "0")}`;
  return `uploads/${ym}/${random}-${safe}`;
}

function randomId() {
  const b = new Uint8Array(8); crypto.getRandomValues(b);
  return [...b].map((x) => x.toString(16).padStart(2, "0")).join("");
}

// ---- helpers ---------------------------------------------------------------
async function readJson(request) {
  const ctype = request.headers.get("Content-Type") || "";
  if (!/^application\/json\b/i.test(ctype)) throw new HttpError(415, "send application/json");
  try { return await request.json(); } catch { throw new HttpError(400, "the body is not valid JSON"); }
}

function noStore(extra = {}) {
  return { "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", ...extra };
}

function json(data, status = 200) {
  return new Response(JSON.stringify(data), { status, headers: noStore({ "Content-Type": "application/json; charset=utf-8" }) });
}

function text(body, status = 200) {
  return new Response(body, { status, headers: noStore({ "Content-Type": "text/plain; charset=utf-8" }) });
}

function methodNotAllowed(allow) {
  return new Response(JSON.stringify({ error: `use ${allow}` }), { status: 405, headers: noStore({ "Content-Type": "application/json; charset=utf-8", Allow: allow }) });
}
