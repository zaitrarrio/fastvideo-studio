// The Clip panel: the video, the stages grid from the worker's progress
// stream, the summary line. Pure functions take DOM nodes as arguments.

// jobs.md stages, in order, with the grid's labels.
export const STAGES = [
  ["queued", "Queue"], ["staging", "Stage weights"], ["loading", "Load"], ["encoding", "Encode"], ["preparing", "Prepare"],
  ["denoising", "Denoise"], ["decoding", "Decode"], ["upscaling", "Upscale"], ["muxing", "Mux"], ["uploading", "Upload"],
];
const INDEX = Object.fromEntries(STAGES.map(([k], i) => [k, i]));
// output.timings keys per stage (the queue comes from RunPod's delayTime).
const TIMING = { staging: "stage_s", loading: "load_s", encoding: "encode_s", denoising: "denoise_s", decoding: "decode_s", upscaling: "upscale_s", muxing: "mux_s" };

export const TERMINAL = new Set(["COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"]);
export const isTerminal = (status) => TERMINAL.has(status);

export function fmt(x) {
  if (x == null || !Number.isFinite(Number(x))) return "–";
  x = Number(x);
  return (x < 10 ? x.toFixed(2) : x.toFixed(1)) + " s";
}

// Client-side stage clock: records the elapsed_s at which each stage was
// first seen, so a finished stage can show its seconds before the output's
// timings arrive. `track` lives on the session item.
export function noteStage(track, progress) {
  const t = track || { seen: {} };
  if (progress?.stage && !(progress.stage in t.seen)) t.seen[progress.stage] = Number(progress.elapsed_s) || 0;
  return t;
}

function seenSeconds(track, key) {
  if (!track?.seen || !(key in track.seen)) return null;
  const later = STAGES.map(([k]) => k).filter((k) => INDEX[k] > INDEX[key] && k in track.seen).map((k) => track.seen[k]);
  if (!later.length) return null;
  return Math.max(0, Math.min(...later) - track.seen[key]);
}

function activeDetail(key, p) {
  if (key === "denoising" && p.total_steps) return `${p.step ?? 0} / ${p.total_steps}`;
  if (key === "decoding" && p.total_frames) return `${p.frames ?? 0} / ${p.total_frames}`;
  if (key === "upscaling" && p.total_frames) return `${p.frames ?? 0} / ${p.total_frames}`;
  if (p.detail) return String(p.detail);
  return p.elapsed_s != null ? fmt(p.elapsed_s) : "running";
}

// `view`: { status, progress, output, error, delayTime, track, upscale }
export function renderStages(root, view) {
  const v = view || {};
  const status = v.status || null;
  const done = status === "COMPLETED";
  const failed = status === "FAILED" || status === "TIMED_OUT" || status === "CANCELLED";
  const p = v.progress || null;
  const timings = v.output?.timings || null;
  let current = -1;
  if (status === "IN_QUEUE") current = INDEX.queued;
  else if (status === "IN_PROGRESS") current = p?.stage in INDEX ? INDEX[p.stage] : INDEX.staging;
  else if (done) current = STAGES.length;
  else if (failed) current = p?.stage in INDEX ? INDEX[p.stage] : Math.max(0, ...Object.keys(v.track?.seen || {}).map((k) => INDEX[k] ?? 0));
  const upscale = v.upscale ? Number(v.upscale) : 0;
  root.innerHTML = STAGES.map(([key, label], i) => {
    let cls = "", detail = "–";
    if (key === "upscaling" && !upscale) { detail = "off"; if (status) cls = i < current ? "" : ""; }
    else if (status && i < current) {
      cls = "done";
      let secs = null;
      if (key === "queued" && v.delayTime != null) secs = v.delayTime / 1000;
      else if (TIMING[key] && timings && timings[TIMING[key]] != null) secs = timings[TIMING[key]];
      else secs = seenSeconds(v.track, key);
      detail = secs != null ? fmt(secs) : "done";
    } else if (status && i === current) {
      if (failed) { cls = "failed"; detail = status === "CANCELLED" ? "cancelled" : status === "TIMED_OUT" ? "timed out" : "failed"; }
      else { cls = "active"; detail = key === "queued" ? "waiting" : activeDetail(key, p || {}); }
    } else if (failed && i > current) detail = "–";
    if (key === "upscaling" && upscale && cls === "active") label = `Upscale ×${upscale}`;
    return `<div class="stage ${cls}"><b>${label}</b><span>${detail}</span></div>`;
  }).join("");
}

// "model · task · 1344×768 · 124 frames · seed 7 · 45.0 s · RTX PRO 6000"
export function summaryText(item) {
  const o = item?.output;
  if (!o) return "";
  const parts = [item.modelLabel || o.model, item.taskLabel || o.task, `${o.width}×${o.height}`, `${o.frames} frames`];
  if (o.seed != null) parts.push(`seed ${o.seed}`);
  if (o.upscale?.factor) parts.push(`×${o.upscale.factor} to ${o.upscale.width}×${o.upscale.height}`);
  if (o.timings?.total_s != null) parts.push(`${fmt(o.timings.total_s)} total`);
  if (o.worker?.gpu) parts.push(shortGpu(o.worker.gpu));
  return parts.join(" · ");
}

export function shortGpu(name) {
  return String(name).replace(/^NVIDIA\s+/i, "").replace(/\s+Blackwell/i, "").replace(/GeForce\s+/i, "");
}

// Points the player at the finished clip (or clears it).
export function showClip(screen, video, empty, output, aspect) {
  if (aspect) screen.style.aspectRatio = aspect;
  if (output?.video_url) {
    if (video.getAttribute("src") !== output.video_url) { video.src = output.video_url; video.load(); }
    if (output.thumbnail_url) video.poster = output.thumbnail_url; else video.removeAttribute("poster");
    video.hidden = false; empty.hidden = true;
  } else {
    video.pause(); video.removeAttribute("src"); video.removeAttribute("poster"); video.load();
    video.hidden = true; empty.hidden = false;
  }
}

export function aspectOf(w, h) {
  return w && h ? `${w} / ${h}` : "16 / 9";
}
