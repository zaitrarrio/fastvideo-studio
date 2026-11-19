// The Studio Worker's API: one shared token (localStorage `studio.token`),
// JSON calls, uploads. A 401 throws AuthError so the page can re-lock.

export const TOKEN_KEY = "studio.token";

export class AuthError extends Error {}
export class ApiError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}

export function store(k, v) {
  try {
    if (v === undefined) return localStorage.getItem(k);
    if (v === null) localStorage.removeItem(k); else localStorage.setItem(k, v);
  } catch { return null; }
  return v;
}

export function getToken() { return store(TOKEN_KEY) || ""; }
export function setToken(t) { store(TOKEN_KEY, t ? t.trim() : null); }

async function call(path, { method = "GET", body, headers = {}, raw = false, signal } = {}) {
  const h = { Accept: "application/json", ...headers };
  const token = getToken();
  if (token && path !== "/api/catalog") h.Authorization = `Bearer ${token}`;
  if (body !== undefined && !raw) { h["Content-Type"] = "application/json"; body = JSON.stringify(body); }
  let r;
  try { r = await fetch(path, { method, headers: h, body, cache: "no-store", signal }); }
  catch (e) { throw new ApiError(0, `The studio is unreachable (${e?.message || e}).`); }
  let data = null;
  try { data = await r.json(); } catch { data = null; }
  if (r.status === 401) throw new AuthError(data?.error || "unlock the studio with its token");
  if (!r.ok) throw new ApiError(r.status, data?.error || `The studio answered ${r.status}.`);
  return data;
}

export const getCatalog = () => call("/api/catalog");
export const getHealth = () => call("/api/health");
export const submitJob = (input) => call("/api/jobs", { method: "POST", body: input });
export const getJob = (id) => call(`/api/jobs/${encodeURIComponent(id)}`);
export const cancelJob = (id) => call(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" });
export const getInfo = () => call("/api/info", { method: "POST", signal: AbortSignal.timeout(70_000) });
export const getPod = () => call("/api/pod", { signal: AbortSignal.timeout(30_000) });
export const startPod = (hours) => call("/api/pod", { method: "POST", body: hours ? { hours } : {}, signal: AbortSignal.timeout(100_000) });
export const stopPod = () => call("/api/pod", { method: "DELETE", signal: AbortSignal.timeout(70_000) });
export const openStream = (body) => call("/api/streams", { method: "POST", body, signal: AbortSignal.timeout(620_000) });
export const getStream = (id) => call(`/api/streams/${encodeURIComponent(id)}`, { signal: AbortSignal.timeout(40_000) });
export const streamCommand = (id, cmd) => call(`/api/streams/${encodeURIComponent(id)}/commands`, { method: "POST", body: cmd, signal: AbortSignal.timeout(70_000) });
export const stopStream = (id) => call(`/api/streams/${encodeURIComponent(id)}`, { method: "DELETE", signal: AbortSignal.timeout(70_000) });

// A file into the R2 bucket: raw body with its name and type.
export function uploadFile(file) {
  return call("/api/uploads", {
    method: "POST", raw: true, body: file,
    headers: { "Content-Type": file.type || "application/octet-stream", "X-File-Name": encodeURIComponent(file.name || "upload.bin") },
  });
}

// Turns a picked File into a jobs.md <media>: images under the inline limit go
// as {b64, mime}, everything else through the bucket as {r2}.
export async function mediaFor(file, inlineLimit) {
  if (file.type.startsWith("image/") && file.size <= inlineLimit) return { b64: await toBase64(file), mime: file.type };
  const up = await uploadFile(file);
  return { r2: up.r2 };
}

export function toBase64(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result).replace(/^data:[^,]*,/, ""));
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });
}
