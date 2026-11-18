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
//
// Secrets: RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID, STUDIO_TOKEN. Vars: ENDPOINT_LABEL.
import catalog from "../catalog/catalog.json" with { type: "json" };

const RUNPOD = "https://api.runpod.ai/v2";
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
  const h = await runpod(env, "/health");
  return json({ label: env.ENDPOINT_LABEL || "", workers: h.workers || {}, jobs: h.jobs || {} });
}

async function submitJob(request, env) {
  const input = await readJson(request);
  if (!input || typeof input !== "object" || Array.isArray(input)) throw new HttpError(400, "the body must be a JSON object (the job input)");
  if ((input.op ?? "generate") === "generate") {
    if (typeof input.prompt !== "string" || !input.prompt.trim()) throw new HttpError(400, "prompt is required");
    if (input.prompt.length > 2000) throw new HttpError(400, "prompt is over 2000 characters");
    if (!catalog.engines[input.engine]) throw new HttpError(400, `unknown engine ${input.engine}`);
    if (!catalog.models.some((m) => m.engine === input.engine && m.id === input.model)) throw new HttpError(400, `unknown model ${input.model} for ${input.engine}`);
  }
  const r = await runpod(env, "/run", { method: "POST", body: { input, policy: { executionTimeout: EXECUTION_TIMEOUT_MS } } });
  if (!r.id) throw new HttpError(502, "RunPod /run gave no job id");
  return json({ id: r.id, status: normalStatus(r.status) }, 202);
}

async function jobStatus(id, env) {
  const r = await runpod(env, `/status/${id}`);
  return json(normaliseStatus(id, r));
}

async function cancelJob(id, env) {
  const r = await runpod(env, `/cancel/${id}`, { method: "POST" });
  return json({ id, status: normalStatus(r.status || "CANCELLED") });
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
