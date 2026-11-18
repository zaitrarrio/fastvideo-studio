// "This session": the last 24 jobs in localStorage `studio.session` (ids,
// the request, the output JSON), rendered as the history grid.
import { store } from "./api.js";
import { isTerminal } from "./player.js";

export const SESSION_KEY = "studio.session";
export const SESSION_MAX = 24;

export function loadSession() {
  try {
    const v = JSON.parse(store(SESSION_KEY) || "[]");
    return Array.isArray(v) ? v.filter((x) => x && x.id) : [];
  } catch { return []; }
}

export function saveSession(items) {
  // Thumbnails are URLs (presigned, 24 h), never data: strings, so the list stays small.
  store(SESSION_KEY, JSON.stringify(items.slice(0, SESSION_MAX)));
}

// Prepends (or replaces) an item and returns the new list.
export function addHistory(items, item) {
  const rest = items.filter((x) => x.id !== item.id);
  return [item, ...rest].slice(0, SESSION_MAX);
}

export function updateHistory(items, id, patch) {
  return items.map((x) => (x.id === id ? { ...x, ...patch } : x));
}

export function findItem(items, id) {
  return items.find((x) => x.id === id) || null;
}

// The presigned URLs last 24 h (output.expires_at).
export function isExpired(item, now = Date.now()) {
  const at = item?.output?.expires_at;
  if (!at) return false;
  const t = Date.parse(at);
  return Number.isFinite(t) && t <= now;
}

// Items whose clip can still be played: the "Use a clip from this session" list.
export function playableClips(items, now = Date.now()) {
  return items.filter((x) => x.status === "COMPLETED" && x.output?.video_url && !isExpired(x, now));
}

export function statusKind(item) {
  if (item.status === "COMPLETED") return isExpired(item) ? "muted" : "ok";
  if (isTerminal(item.status)) return "bad";
  return "wait";
}

export function statusText(item) {
  if (item.status === "COMPLETED") return isExpired(item) ? "expired" : "done";
  if (item.status === "FAILED") return "failed";
  if (item.status === "CANCELLED") return "cancelled";
  if (item.status === "TIMED_OUT") return "timed out";
  if (item.status === "IN_PROGRESS") return item.progress?.stage || "running";
  return "queued";
}

const esc = (t) => String(t ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

export function renderHistory(root, items, currentId) {
  if (!items.length) { root.innerHTML = `<p class="hint">Finished clips appear here. The last 24 jobs are kept in this browser; their links expire after 24 hours.</p>`; return; }
  root.innerHTML = items.map((it) => {
    const expired = isExpired(it);
    const thumb = it.output?.thumbnail_url && !expired ? `<img src="${esc(it.output.thumbnail_url)}" alt="" loading="lazy">` : `<span>${esc(statusText(it))}</span>`;
    const w = it.request?.width || it.output?.width || 16, h = it.request?.height || it.output?.height || 9;
    return `<button type="button" class="clip${it.id === currentId ? " current" : ""}" data-id="${esc(it.id)}" title="${esc(it.prompt)}">
      <span class="shot" style="aspect-ratio:${w} / ${h}">${thumb}</span>
      <span class="text">${esc(it.prompt)}</span>
      <span class="meta"><span title="${esc(it.modelLabel || it.model)}">${esc(it.modelLabel || it.model)}</span><span class="status ${statusKind(it)}"><span class="dot ${statusKind(it)}"></span>${esc(statusText(it))}</span></span>
    </button>`;
  }).join("");
}
