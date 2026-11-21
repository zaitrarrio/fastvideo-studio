// FastVideo Studio: wiring. State at the top, small named functions, fetch
// polling; the form is generated from the catalog (form.js), the clip panel
// is player.js, the history is session.js, the Worker's API is api.js.
import * as api from "./api.js";
import { families, familyById, modelById, defaultModel, taskLabel, taskSpec, knobDef, inlineLimit } from "./catalog.js";
import { knobDefaults, buildRequest, CARD_LIMIT, cliLine, hintFor, missingInputs, readKnob, renderFamilies, renderModels, renderTasks, renderSizes, renderLengths, renderKnobs, updateKnobVisibility, renderInputs, isUpscaleKnob } from "./form.js";
import { renderStages, summaryText, showClip, aspectOf, apiRatio, isTerminal, noteStage, fmt } from "./player.js";
import { loadSession, saveSession, addHistory, updateHistory, findItem, isExpired, playableClips, renderHistory } from "./session.js";
import { playWhep, stopPlayer, liveText, isLiveTerminal, commandFor } from "./live.js";

const $ = (id) => document.getElementById(id);
const FORM_KEY = "studio.form";
const INFO_KEY = "studio.info";
const BACKEND_KEY = "studio.backend";
const MODE_KEY = "studio.mode";
const LIVE_MS = 2000;
const POD_MS = 10000;
const POLL_MS = 2000;
const HEALTH_MS = 15000;
const EXAMPLES = [
  ["Fox at dawn", "A red fox trots through fresh snow in a birch forest at dawn, its breath steaming, low golden sunlight, shallow depth of field; snow crunches softly under its paws and a distant woodpecker taps.", "nature"],
  ["Rain street", "Cinematic tracking shot along a rain-soaked Tokyo side street at night, neon signs doubled in the puddles, a woman in a yellow raincoat walking away from camera, anamorphic flares, 35 mm film grain.", "cinematic"],
  ["Watch macro", "Product shot: a matte-black mechanical watch rotating slowly on a dark slate surface, a single soft key light sweeping across the dial, dust motes in the beam, extreme macro detail.", "product"],
  ["Paper boat", "Hand-drawn 2D animation of a paper boat drifting down a rain gutter river, bold ink outlines, flat pastel colours, a cat watching from the kerb, gentle camera pan.", "animation"],
  ["Two at the bar", "A dimly lit jazz bar; a bartender slides a glass across the counter and says \"Last one, then I close\". The man in the grey coat answers \"One is all I need\" and smiles; brushed snare and a double bass in the background.", "dialogue"],
  ["Cliff waves", "Aerial drone shot sweeping along green sea cliffs as waves crash into white foam, overcast sky, cinematic, gulls circling.", "nature"],
];

const S = {
  cat: null, family: null, model: null, task: null,
  size: null, frames: null, values: {},
  inputs: {},            // name → [{file?, url?, name, mime, kind, thumb?}]
  items: loadSession(),  // "This session"
  current: null,         // the item shown in the Clip panel
  polling: null,         // the job id being polled
  pollTimer: 0, healthTimer: 0,
  locked: !api.getToken(),
  busy: false,
  backend: api.store(BACKEND_KEY) === "pod" ? "pod" : "serverless",
  pod: null,             // the last /api/pod summary
  podTimer: 0, podBusy: false,
  mode: ["live", "segments"].includes(api.store(MODE_KEY)) ? api.store(MODE_KEY) : "clip",
  seg: null,             // the segment stream: {streamId, index, model, items: [...], current}
  segTimer: 0, segBusy: false,
  stream: null,          // the live stream object being followed
  streamTimer: 0, streamBusy: false, whepPlaying: false,
  cardLimit: CARD_LIMIT, // 0 = every model of the family
};

// ---- status line --------------------------------------------------------------------
function setEngine(kind, text) { $("eng-dot").className = "dot " + kind; $("eng-text").textContent = text; }

function scheduleHealth(ms) { clearTimeout(S.healthTimer); S.healthTimer = setTimeout(refreshHealth, ms); }
async function refreshHealth() {
  clearTimeout(S.healthTimer);
  if (S.locked) { setEngine("", "Locked: enter the studio token"); return; }
  let h;
  try { h = await api.getHealth(); }
  catch (e) {
    if (e instanceof api.AuthError) { lock("That token was refused."); return; }
    setEngine("bad", `Endpoint unreachable: ${e.message}`); scheduleHealth(HEALTH_MS); return;
  }
  const w = h.workers || {}, j = h.jobs || {};
  const up = ["idle", "ready", "running", "initializing"].reduce((a, k) => a + (w[k] || 0), 0);
  const queue = j.inQueue || 0;
  const label = h.label ? `${h.label} · ` : "";
  if (h.pod) renderPod(h.pod);
  if (S.backend === "pod") {
    const p = S.pod || {};
    if (p.state === "running" && p.ready) setEngine("ok", `${label}pod ${p.id} ready: ${p.health?.busy ? "busy" : "idle"}, ${p.health?.queued || 0} queued, ${p.health?.jobs_done || 0} done`);
    else if (!p.state || ["none", "gone", "terminated"].includes(p.state)) setEngine("", `${label}no pod: start one below (the queue endpoint still answers)`);
    else setEngine("wait", `${label}pod ${p.id || ""} ${p.state}${p.state === "running" ? ", serve not ready yet" : ""}`);
  } else {
    setEngine("ok", `${label}ready: ${up} worker${up === 1 ? "" : "s"}${up ? "" : " (the first job starts one)"}, ${queue} in queue${w.throttled ? `, ${w.throttled} throttled` : ""}`);
  }
  updateGo();
  scheduleHealth(HEALTH_MS);
}

// ---- the pod backend (docs/pod-mode.md) --------------------------------------------
function pickBackend(b) {
  S.backend = b === "pod" ? "pod" : "serverless";
  api.store(BACKEND_KEY, S.backend);
  for (const el of $("backends").querySelectorAll("[data-backend]")) el.setAttribute("aria-pressed", String(el.dataset.backend === S.backend));
  $("podbox").hidden = S.backend !== "pod";
  if (S.backend === "pod") refreshPod(); else clearTimeout(S.podTimer);
  renderMode();
  updateGo(); if (!S.locked) refreshHealth();
}
function podReady() { return S.backend !== "pod" || (S.pod?.state === "running" && !!S.pod?.ready); }

// ---- modes in the pod backend: clip jobs or a live stream --------------------------
function pickMode(m) {
  S.mode = ["live", "segments"].includes(m) ? m : "clip";
  api.store(MODE_KEY, S.mode);
  renderMode();
  updateGo();
}
function liveMode() { return S.backend === "pod" && S.mode === "live"; }
function segMode() { return S.backend === "pod" && S.mode === "segments"; }
function renderMode() {
  const pod = S.backend === "pod";
  const special = liveMode() || segMode();
  $("modes").hidden = !pod;
  $("mode-hint").hidden = !liveMode();
  for (const el of $("modes").querySelectorAll("[data-mode]")) el.setAttribute("aria-pressed", String(el.dataset.mode === S.mode));
  $("live").hidden = !liveMode();
  $("segments").hidden = !segMode();
  $("out-h").closest("section").hidden = special;
  $("length").closest(".field").hidden = special;
  $("adv").hidden = special;
  $("steps-hint").hidden = special;
  if (liveMode()) renderLive();
  if (segMode()) renderSegments();
}

// ---- segments: one stream, a prompt per segment, played back to back -----------
function segText() {
  const g = S.seg;
  if (!g) return "No stream yet. Pick a fastvideo-rs model, write the first segment's prompt and press Next segment: each one continues the last (a causal model as one rollout, a clip model chained on its last frame) and they play back to back.";
  const done = g.items.filter((i) => i.status === "succeeded").length, pending = g.items.filter((i) => !["succeeded", "failed"].includes(i.status)).length;
  return `Stream ${g.streamId} on ${g.model}: ${g.items.length} segment${g.items.length === 1 ? "" : "s"}, ${done} ready${pending ? `, ${pending} rendering` : ""}${g.restarted ? " · the session restarted (a cut)" : ""}.`;
}
function renderSegments() {
  const g = S.seg;
  $("seg-text").textContent = segText();
  $("seg-summary").textContent = g?.items.at(-1)?.load_s ? `fv-serve up in ${fmt(g.items.at(-1).load_s)}` : "";
  $("seg-next").disabled = S.segBusy || !podReady() || !(g ? true : liveModelOk());
  $("seg-next").textContent = S.segBusy ? "Asking…" : g ? `Next segment (#${g.index})` : "First segment";
  $("seg-close").disabled = !g || S.segBusy;
  $("seg-list").innerHTML = (g?.items || []).map((it) => `<div class="item${it.id === g.current ? " current" : ""}" data-seg="${esc(it.id)}"><span class="st ${it.status === "succeeded" ? "ok" : it.status === "failed" ? "bad" : "wait"}"></span><span class="t">#${it.index} · ${esc(it.status)}${it.duration_sec ? ` · ${fmt(it.duration_sec)}` : ""}${it.reset ? " · cut" : ""}</span><span class="p">${esc(it.error ? `${it.prompt.slice(0, 40)} — ${it.error}` : it.prompt.slice(0, 80))}</span></div>`).join("");
}
function showSegError(m) { $("seg-err").textContent = m || ""; $("seg-err").hidden = !m; }
// fv-serve's errors are objects ({message, type, param}), not strings.
function errText(e) {
  if (e == null) return "";
  if (typeof e === "string") return e;
  if (typeof e !== "object") return String(e);
  const m = e.message || e.error || e.detail;
  return m ? `${m}${e.param ? ` (${e.param})` : ""}` : JSON.stringify(e);
}
async function nextSegment() {
  if (S.segBusy || !podReady()) return;
  const prompt = $("prompt").value.trim();
  if (!prompt) { showSegError("Write the segment's prompt first."); $("prompt").focus(); return; }
  if (!S.seg && !liveModelOk()) { showSegError("Pick a fastvideo-rs model first."); return; }
  showSegError("");
  if (!S.seg) S.seg = { streamId: `str_${Math.random().toString(36).slice(2, 10)}${Date.now().toString(36)}`, index: 0, model: S.model.id, items: [], current: null, restarted: false };
  const g = S.seg;
  const body = { stream_id: g.streamId, segment_index: g.index, model: g.model, prompt, duration_sec: Number($("seg-seconds").value) || 6 };
  if ($("seg-reset").value === "1") { body.reset = true; $("seg-reset").value = ""; }
  const seed = Number($("seed").value); if (Number.isFinite(seed) && $("seed").value !== "") body.seed = seed;
  if (S.size && g.index === 0) { body.resolution = `${Math.min(S.size[0], S.size[1])}p`; body.aspect_ratio = apiRatio(S.size[0], S.size[1]); }
  S.segBusy = true; renderSegments(); updateGo();
  try {
    const r = await api.nextSegment(body);
    g.index += 1;
    g.items.push({ id: r.id, index: body.segment_index, status: r.status || "queued", prompt, reset: !!body.reset, load_s: r.load_s, video_url: null, duration_sec: r.planned_duration_sec || null });
    scheduleSegPoll(200);
  } catch (e) {
    if (e instanceof api.AuthError) lock("The studio token was refused."); else showSegError(e.message);
  }
  S.segBusy = false; renderSegments(); updateGo();
}
function scheduleSegPoll(ms) { clearTimeout(S.segTimer); S.segTimer = setTimeout(pollSegments, ms); }
async function pollSegments() {
  const g = S.seg; if (!g) return;
  const pending = g.items.filter((i) => !["succeeded", "failed"].includes(i.status));
  for (const it of pending) {
    try {
      const r = await api.getSegment(it.id);
      it.status = r.status || it.status;
      if (r.video_url) it.video_url = r.video_url;
      if (r.duration_sec) it.duration_sec = r.duration_sec;
      if (r.session?.restarted) g.restarted = true;
      if (r.status === "failed") it.error = errText(r.error) || "failed";
      if (r.mirror_error) it.error = errText(r.mirror_error);
    } catch (e) {
      if (e instanceof api.AuthError) { lock("The studio token was refused."); return; }
      showSegError(`Segment ${it.index}: ${e.message}`);
    }
  }
  renderSegments();
  playNextSegment();
  if (g.items.some((i) => !["succeeded", "failed"].includes(i.status))) scheduleSegPoll(POLL_MS);
}
// Two <video> elements: the one playing stays on screen while the next ready
// segment is loaded into the other; on `ended` they swap.
function segVideos() { const a = $("seg-video-a"), b = $("seg-video-b"); return a.hidden ? [b, a] : [a, b]; }
function playNextSegment() {
  const g = S.seg; if (!g) return;
  const [shown, other] = segVideos();
  const ready = g.items.filter((i) => i.status === "succeeded" && i.video_url);
  if (!g.current) {
    const first = ready[0]; if (!first) return;
    g.current = first.id; shown.src = first.video_url; shown.hidden = false; other.hidden = true; shown.play?.().catch(() => {});
    renderSegments(); return;
  }
  const at = ready.findIndex((i) => i.id === g.current);
  const next = at >= 0 ? ready[at + 1] : null;
  if (next && other.dataset.seg !== next.id) { other.src = next.video_url; other.dataset.seg = next.id; other.preload = "auto"; other.load?.(); }
  if (next && g.waiting && shown.ended) { g.waiting = false; onSegmentEnded({ target: shown }); }
}
function onSegmentEnded(ev) {
  const g = S.seg; if (!g) return;
  const [shown, other] = segVideos();
  if (ev.target !== shown) return;
  const ready = g.items.filter((i) => i.status === "succeeded" && i.video_url);
  const at = ready.findIndex((i) => i.id === g.current);
  const next = at >= 0 ? ready[at + 1] : null;
  if (!next) { g.waiting = true; return; }
  if (other.dataset.seg !== next.id) { other.src = next.video_url; other.dataset.seg = next.id; }
  g.current = next.id; other.hidden = false; shown.hidden = true; other.play?.().catch(() => {});
  renderSegments();
  playNextSegment();
}
async function closeSegments() {
  const g = S.seg; if (!g || S.segBusy) return;
  S.segBusy = true; renderSegments();
  try { await api.closeSegmentStream(g.streamId); } catch (e) { if (e instanceof api.AuthError) lock("The studio token was refused."); else showSegError(e.message); }
  clearTimeout(S.segTimer);
  for (const v of [$("seg-video-a"), $("seg-video-b")]) { v.pause?.(); v.removeAttribute("src"); delete v.dataset.seg; }
  S.seg = null; S.segBusy = false; renderSegments(); updateGo();
}
function liveModelOk() { return !!S.model && S.model.engine === "fastvideo"; }
function renderLive() {
  const s = S.stream;
  const live = !!s && !isLiveTerminal(s);
  $("live-text").textContent = s ? liveText(s) : "No stream. Pick a fastvideo-rs model, write the opening prompt and press Start stream.";
  $("live-summary").textContent = s ? `${s.studio_model || s.model || ""}${s.load_s ? ` · fv-serve up in ${fmt(s.load_s)}` : ""}` : "";
  const clip = s?.mode === "clip";
  $("steer-label").textContent = clip ? "Next clip prompt (queued behind the clips already generating)" : "New prompt (applied at the next block boundary)";
  $("steer-seconds-row").hidden = !clip;
  $("live-steer").textContent = clip ? "Enqueue clip" : "Switch prompt";
  $("live-pause").textContent = clip ? "Pause (autoplay off)" : "Pause";
  $("live-resume").textContent = clip ? "Resume (autoplay on)" : "Resume";
  for (const id of ["live-steer", "live-pause", "live-resume", "live-reset", "live-stop"]) $(id).disabled = !live || S.streamBusy;
}
function showLiveError(m) { $("live-err").textContent = m || ""; $("live-err").hidden = !m; }
async function startStream() {
  if (S.streamBusy || !podReady() || !liveModelOk()) return;
  const prompt = $("prompt").value.trim();
  if (!prompt) { showError("Write the opening prompt first."); $("prompt").focus(); return; }
  showError(""); showLiveError("");
  S.streamBusy = true; updateGo();
  const body = { model: S.model.id, prompt };
  const seed = Number($("seed").value); if (Number.isFinite(seed) && $("seed").value !== "") body.seed = seed;
  if (S.size) body.resolution = `${Math.min(S.size[0], S.size[1])}p`;
  if (S.size) body.aspect_ratio = apiRatio(S.size[0], S.size[1]);
  try {
    S.stream = await api.openStream(body);
    S.whepPlaying = false;
    $("steer").value = "";
    followStream();
  } catch (e) {
    if (e instanceof api.AuthError) lock("The studio token was refused.");
    else showError(e.message);
  }
  S.streamBusy = false; updateGo(); renderLive();
}
function followStream() {
  clearTimeout(S.streamTimer);
  pollStream();
}
async function pollStream() {
  const id = S.stream?.id; if (!id) return;
  try { S.stream = await api.getStream(id); }
  catch (e) {
    if (e instanceof api.AuthError) { lock("The studio token was refused."); return; }
    if (e.status === 404) { S.stream = { ...S.stream, status: { ...(S.stream.status || {}), state: "closed" } }; }
    else showLiveError(`Stream status: ${e.message}`);
  }
  renderLive();
  const s = S.stream;
  const whep = s?.playback?.whep_url;
  if (!S.whepPlaying && whep && s?.status?.output_state === "publishing") {
    S.whepPlaying = true;
    try { await playWhep($("live-video"), whep); showLiveError(""); }
    catch (e) { S.whepPlaying = false; showLiveError(`WHEP playback: ${e.message}`); }
  }
  if (isLiveTerminal(s)) { await stopPlayer($("live-video")); S.whepPlaying = false; renderLive(); updateGo(); return; }
  S.streamTimer = setTimeout(pollStream, LIVE_MS);
}
async function liveCommand(action) {
  const s = S.stream; if (!s || S.streamBusy) return;
  const data = { prompt: $("steer").value.trim(), seconds: Number($("steer-seconds").value) };
  if (action === "steer" && !data.prompt) { showLiveError("Write the next prompt first."); return; }
  S.streamBusy = true; renderLive();
  try { const r = await api.streamCommand(s.id, commandFor(s.mode, action, data)); showLiveError(""); $("live-text").textContent = `${action}: ${JSON.stringify(r.reply ?? r).slice(0, 160)}`; }
  catch (e) { if (e instanceof api.AuthError) lock("The studio token was refused."); else showLiveError(e.message); }
  S.streamBusy = false; renderLive();
}
async function stopLive() {
  const s = S.stream; if (!s || S.streamBusy) return;
  S.streamBusy = true; renderLive();
  try { await api.stopStream(s.id); } catch (e) { if (e instanceof api.AuthError) lock("The studio token was refused."); else showLiveError(e.message); }
  clearTimeout(S.streamTimer);
  await stopPlayer($("live-video")); S.whepPlaying = false;
  S.stream = { ...s, status: { ...(s.status || {}), state: "closed", output_state: "ended" } };
  S.streamBusy = false; renderLive(); updateGo();
}
function podLive(p) { return !!p?.state && !["none", "gone", "terminated"].includes(p.state); }
function renderPod(p) {
  S.pod = p;
  const live = podLive(p);
  $("pod-start").hidden = live; $("pod-stop").hidden = !live;
  const h = p.health || {};
  let text;
  if (!live && p.id) text = `Pod ${p.id} is ${p.state}.`;
  else if (!live) text = "No pod. Start one to run every engine warm: about a minute to boot, then the first job of a model stages its weights once.";
  else if (p.state === "running" && p.ready) text = `Pod ${p.id} (${p.gpu || "GPU"}${p.dc ? `, ${p.dc}` : ""}${p.cost ? `, $${p.cost}/h` : ""}) is ready: ${h.busy ? "running a job" : "idle"}, ${h.queued || 0} queued, ${h.jobs_done || 0} done, up ${fmt(h.uptime_s || 0)}; it stops itself after ${p.hours || h.pod_hours || "?"} h.`;
  else if (p.state === "running") text = `Pod ${p.id} is running; its serve is not answering yet${p.health_error ? ` (${p.health_error})` : ""}.`;
  else text = `Pod ${p.id} is ${p.state}${p.error ? ` (${p.error})` : ""}…`;
  $("pod-text").textContent = text;
  updateGo();
}
async function refreshPod() {
  clearTimeout(S.podTimer);
  if (S.locked || S.backend !== "pod") return;
  try { renderPod(await api.getPod()); }
  catch (e) { if (e instanceof api.AuthError) { lock("The studio token was refused."); return; } $("pod-text").textContent = `Pod status failed: ${e.message}`; }
  const p = S.pod || {};
  const settling = podLive(p) && !(p.state === "running" && p.ready);
  S.podTimer = setTimeout(refreshPod, settling ? POD_MS : HEALTH_MS * 4);
}
async function startPod() {
  if (S.podBusy) return;
  S.podBusy = true; $("pod-start").disabled = true; $("pod-text").textContent = "Asking RunPod for a pod…";
  try { renderPod(await api.startPod()); clearTimeout(S.podTimer); S.podTimer = setTimeout(refreshPod, POD_MS); }
  catch (e) { if (e instanceof api.AuthError) lock("The studio token was refused."); else $("pod-text").textContent = `Start failed: ${e.message}`; }
  S.podBusy = false; $("pod-start").disabled = false;
}
async function stopPod() {
  if (S.podBusy || !confirm("Stop the pod? Its resident engines and any running job are lost.")) return;
  S.podBusy = true; $("pod-stop").disabled = true; $("pod-text").textContent = "Terminating the pod…";
  try { await api.stopPod(); renderPod({ state: "terminated", id: S.pod?.id }); }
  catch (e) { if (e instanceof api.AuthError) lock("The studio token was refused."); else $("pod-text").textContent = `Stop failed: ${e.message}`; }
  S.podBusy = false; $("pod-stop").disabled = false;
  if (!S.locked) refreshHealth();
}

// ---- lock / unlock ------------------------------------------------------------------
function lock(message) {
  S.locked = true;
  $("unlock").hidden = false;
  $("unlock-err").textContent = message || ""; $("unlock-err").hidden = !message;
  setEngine("", "Locked: enter the studio token");
  updateGo();
}
async function unlock() {
  const t = $("token").value.trim();
  if (!t) { lock("Enter the token."); return; }
  api.setToken(t);
  S.locked = false; $("unlock-err").hidden = true;
  setEngine("wait", "Checking the endpoint…");
  try { await api.getHealth(); }
  catch (e) {
    if (e instanceof api.AuthError) { api.setToken(null); lock("That token was refused."); return; }
    setEngine("bad", `Endpoint unreachable: ${e.message}`);
  }
  $("unlock").hidden = true; $("token").value = "";
  refreshHealth();
  resumePolling();
  if (S.backend === "pod") refreshPod();
}

// ---- the form ------------------------------------------------------------------------
function loadForm() { try { return JSON.parse(api.store(FORM_KEY) || "{}") || {}; } catch { return {}; } }
function saveForm() {
  const f = loadForm();
  f.family = S.family; f.model = S.model?.id; f.prompt = $("prompt").value; f.negative = $("negative").value;
  f.byModel = f.byModel || {};
  if (S.model) f.byModel[S.model.id] = { task: S.task, size: S.size, frames: S.frames, values: S.values, seed: $("seed").value };
  api.store(FORM_KEY, JSON.stringify(f));
}

function pickFamily(id, keepModel) {
  S.family = familyById(S.cat, id) ? id : families(S.cat)[0].id;
  renderFamilies($("families"), S.cat, S.family);
  S.cardLimit = CARD_LIMIT;
  const saved = loadForm();
  const want = keepModel || (saved.family === S.family ? saved.model : null);
  pickModel(modelById(S.cat, want) && modelById(S.cat, want).family === S.family ? want : defaultModel(S.cat, S.family).id);
}

function pickModel(id) {
  const m = modelById(S.cat, id); if (!m) return;
  S.model = m;
  renderModels($("models"), S.cat, S.family, m.id, S.cardLimit);
  const saved = loadForm().byModel?.[m.id] || {};
  S.values = { ...knobDefaults(S.cat, m), ...(saved.values || {}) };
  for (const k of Object.keys(S.values)) if (!(m.knobs || []).includes(k)) delete S.values[k];
  S.size = saved.size && m.sizes.some(([w, h]) => w === saved.size[0] && h === saved.size[1]) ? saved.size : m.default_size || m.sizes[0];
  S.frames = saved.frames && (m.frames.choices || []).includes(saved.frames) ? saved.frames : m.frames.default;
  if (saved.seed != null) $("seed").value = saved.seed;
  pickTask(m.tasks.includes(saved.task) ? saved.task : m.tasks[0], true);
  renderSizes($("size"), m, S.size);
  renderLengths($("length"), m, S.frames);
  renderKnobs($("knobs"), S.cat, m, S.values);
  $("adv-sub").textContent = `· ${(m.knobs || []).length} knob${(m.knobs || []).length === 1 ? "" : "s"}`;
  $("neg-row").hidden = !m.negative_prompt;
  $("steps-hint").textContent = hintFor(m);
  renderExamples();
  refreshForm();
}

function pickTask(t, silent) {
  S.task = t;
  renderTasks($("tasks"), S.cat, S.model, t);
  for (const name of Object.keys(S.inputs)) if (!(taskSpec(S.cat, t).inputs || []).includes(name)) dropInput(name);
  refreshInputs();
  if (!silent) refreshForm();
}

function refreshForm() {
  updateCommand();
  updateGo();
  saveForm();
}

function updateGo() {
  if (segMode()) {
    const ok = !S.locked && !S.segBusy && podReady() && (S.seg ? true : liveModelOk());
    $("go").disabled = !ok;
    $("go").textContent = S.segBusy ? "Asking…" : !podReady() ? "Start the pod first" : !S.seg && !liveModelOk() ? "Pick a fastvideo-rs model" : S.seg ? `Next segment (#${S.seg.index})` : "First segment";
    return;
  }
  if (liveMode()) {
    const live = !!S.stream && !isLiveTerminal(S.stream);
    const ok = liveModelOk() && !S.locked && !S.streamBusy && podReady() && !live;
    $("go").disabled = !ok;
    $("go").textContent = S.streamBusy ? "Starting…" : live ? "Stream running" : !podReady() ? "Start the pod first" : !liveModelOk() ? "Pick a fastvideo-rs model" : "Start stream";
    return;
  }
  const ok = !!S.model && !S.locked && !S.busy && podReady();
  $("go").disabled = !ok;
  $("go").textContent = S.busy ? "Submitting…" : S.backend === "pod" ? (podReady() ? "Generate clip on the pod" : "Start the pod first") : "Generate clip";
}

function currentState() {
  return {
    task: S.task, prompt: $("prompt").value, negativePrompt: $("negative").value,
    size: S.size, frames: S.frames, seed: $("seed").value,
    inputs: S.inputs, values: S.values,
  };
}

function updateCommand() {
  if (!S.model) return;
  const st = currentState();
  st.prompt = st.prompt.trim() || "…";
  const req = buildRequest(S.cat, S.model, st);
  $("cmd").textContent = cliLine(S.cat, S.model, req, st);
}

function renderExamples() {
  const audio = !!S.model?.audio;
  const list = EXAMPLES.filter(([, , kind]) => audio || kind !== "dialogue").slice(0, 5);
  $("examples").innerHTML = list.map(([l, p]) => `<button type="button" class="chip" data-prompt="${p.replace(/"/g, "&quot;")}" aria-pressed="${$("prompt").value.trim() === p}" title="${p.replace(/"/g, "&quot;")}">${l}</button>`).join("");
}
function markChip() {
  const cur = $("prompt").value.trim();
  document.querySelectorAll("#examples .chip").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.prompt === cur)));
}

// ---- knobs ----------------------------------------------------------------------------
function onKnobInput(e) {
  const el = e.target.closest("[data-knob]"); if (!el) return;
  const key = el.dataset.knob, def = knobDef(S.cat, key);
  S.values[key] = readKnob(def, el);
  if (el.type === "range") el.parentElement.querySelector("output").textContent = el.value;
  updateKnobVisibility($("knobs"), S.cat, S.model, S.values);
  refreshForm();
}
function resetKnobs() {
  S.values = knobDefaults(S.cat, S.model);
  renderKnobs($("knobs"), S.cat, S.model, S.values);
  refreshForm();
}

// ---- inputs ---------------------------------------------------------------------------
function refreshInputs() {
  renderInputs($("inputs"), S.cat, S.model, S.task, S.inputs, playableClips(S.items));
}
function addFiles(name, files) {
  const multi = name === "references";
  const list = multi ? (S.inputs[name] || []) : [];
  if (!multi) dropInput(name);
  for (const f of files) {
    const kind = (f.type || "").split("/")[0] || "image";
    list.push({ file: f, name: f.name, mime: f.type, kind, thumb: kind === "image" ? URL.createObjectURL(f) : null });
  }
  S.inputs[name] = list;
  refreshInputs(); refreshForm();
}
function useSessionClip(name, id) {
  const it = findItem(S.items, id); if (!it?.output?.video_url) return;
  dropInput(name);
  S.inputs[name] = [{ url: it.output.video_url, name: `session clip ${it.prompt.slice(0, 32)}${it.prompt.length > 32 ? "…" : ""}`, mime: "video/mp4", kind: "video", thumb: it.output.thumbnail_url || null, session: true }];
  refreshInputs(); refreshForm();
}
function removeInput(name, index) {
  const list = S.inputs[name] || [];
  const [m] = list.splice(index, 1);
  if (m?.thumb && !m.session) URL.revokeObjectURL(m.thumb);
  if (!list.length) delete S.inputs[name];
  refreshInputs(); refreshForm();
}
function dropInput(name) {
  for (const m of S.inputs[name] || []) if (m.thumb && !m.session) URL.revokeObjectURL(m.thumb);
  delete S.inputs[name];
}
// Picked files → jobs.md <media>: inline images, R2 for the rest.
async function resolveInputs() {
  const out = {};
  const limit = inlineLimit(S.cat);
  for (const [name, list] of Object.entries(S.inputs)) {
    out[name] = [];
    for (const m of list) {
      if (m.url) { out[name].push({ url: m.url, name: m.name, mime: m.mime, kind: m.kind }); continue; }
      $("go").textContent = `Uploading ${m.name}…`;
      const media = await api.mediaFor(m.file, limit);
      out[name].push({ ...media, name: m.name, mime: m.mime, kind: m.kind });
    }
  }
  return out;
}
// Strips the media payloads before an input list goes to the request or the session.
function bareMedia(inputs) {
  const out = {};
  for (const [name, list] of Object.entries(inputs)) out[name] = list.map(({ name: n, mime, kind, url, b64, r2 }) => (url ? { url } : b64 ? { b64, mime } : { r2 })).map((m, i) => ({ ...m, name: list[i].name, mime: list[i].mime, kind: list[i].kind }));
  return out;
}
function cleanRequest(req) {
  const inputs = {};
  for (const [name, v] of Object.entries(req.inputs || {})) {
    const strip = (m) => ({ name: m.name, mime: m.mime, kind: m.kind, ...(m.url ? { url: m.url } : {}), ...(m.r2 ? { r2: m.r2 } : {}) });
    inputs[name] = Array.isArray(v) ? v.map(strip) : strip(v);
  }
  return { ...req, inputs };
}
function wireMedia(req) {
  // The job carries plain <media> objects; the names stay in the session copy only.
  const inputs = {};
  for (const [name, v] of Object.entries(req.inputs || {})) {
    const plain = (m) => (m.url ? { url: m.url } : m.b64 ? { b64: m.b64, mime: m.mime } : { r2: m.r2 });
    inputs[name] = Array.isArray(v) ? v.map(plain) : plain(v);
  }
  return { ...req, inputs };
}

// ---- generation -------------------------------------------------------------------------
async function generate(e) {
  e?.preventDefault();
  if (liveMode()) return startStream();
  if (segMode()) return nextSegment();
  if (!S.model || S.locked || S.busy) return;
  showError("");
  const st = currentState();
  if (!st.prompt.trim()) { showError("Write a prompt first."); $("prompt").focus(); return; }
  const missing = missingInputs(S.cat, S.task, S.inputs);
  if (missing.length) { showError(`This task needs: ${missing.map((n) => n.replace("_", " ")).join(", ")}.`); return; }
  S.busy = true; updateGo();
  const localId = `s${Date.now().toString(36)}`;
  try {
    st.inputs = bareMedia(await resolveInputs());
    const req = buildRequest(S.cat, S.model, st, { studio: "fastvideo-studio", job: localId });
    const v = await api.submitJob({ ...wireMedia(req), backend: S.backend });
    const item = {
      id: v.id, at: new Date().toISOString(), status: v.status || "IN_QUEUE", backend: v.backend || S.backend,
      request: cleanRequest(req), prompt: req.prompt, model: S.model.id, modelLabel: S.model.label, engine: S.model.engine,
      task: S.task, taskLabel: taskLabel(S.cat, S.task), upscale: req.upscale, progress: null, output: null, error: null, track: { seen: {} },
    };
    S.items = addHistory(S.items, item); saveSession(S.items);
    openItem(item.id);
    startPolling(item.id);
  } catch (err) {
    if (err instanceof api.AuthError) lock("The studio token was refused: unlock again.");
    else showError(err.message);
  } finally {
    S.busy = false; updateGo();
  }
}
function showError(m) { $("err").textContent = m; $("err").hidden = !m; }

// ---- polling ------------------------------------------------------------------------------
function startPolling(id) {
  clearTimeout(S.pollTimer);
  S.polling = id;
  pollJob();
}
function stopPolling() { clearTimeout(S.pollTimer); S.polling = null; }
async function pollJob() {
  const id = S.polling; if (!id) return;
  let s;
  try { s = await api.getJob(id); }
  catch (e) {
    if (e instanceof api.AuthError) { stopPolling(); lock("The studio token was refused: unlock again to keep following the job."); return; }
    if (e.status === 404) { S.items = updateHistory(S.items, id, { status: "FAILED", error: "The endpoint no longer knows this job." }); saveSession(S.items); stopPolling(); renderAll(); return; }
    setEngine("bad", `Polling failed: ${e.message}`);
    S.pollTimer = setTimeout(pollJob, POLL_MS * 2); return;
  }
  if (S.polling !== id) return;
  const it = findItem(S.items, id) || {};
  const patch = { status: s.status, delayTime: s.delayTime ?? it.delayTime, executionTime: s.executionTime ?? it.executionTime };
  if (s.progress) { patch.progress = s.progress; patch.track = noteStage(it.track, s.progress); }
  if (s.output) patch.output = s.output;
  if (s.error) patch.error = s.error;
  S.items = updateHistory(S.items, id, patch); saveSession(S.items);
  renderAll();
  if (isTerminal(s.status)) { stopPolling(); refreshHealth(); return; }
  S.pollTimer = setTimeout(pollJob, POLL_MS);
}
// After a reload, keep following the newest job that has not finished.
function resumePolling() {
  if (S.locked || S.polling) return;
  const open = S.items.find((it) => !isTerminal(it.status));
  if (open) { openItem(open.id); startPolling(open.id); }
}
async function cancel() {
  const id = S.current; if (!id) return;
  $("cancel").disabled = true;
  try { await api.cancelJob(id); S.items = updateHistory(S.items, id, { status: "CANCELLED" }); saveSession(S.items); if (S.polling === id) stopPolling(); renderAll(); }
  catch (e) { if (e instanceof api.AuthError) lock("The studio token was refused."); else showJobError(e.message); }
  $("cancel").disabled = false;
}

// ---- the clip panel -----------------------------------------------------------------------
function openItem(id) {
  S.current = id;
  renderAll();
}
async function reopen(id) {
  const it = findItem(S.items, id); if (!it) return;
  restoreForm(it);
  openItem(id);
  if (!isTerminal(it.status) && !S.locked) startPolling(id);
  else if (!S.locked && it.status === "COMPLETED" && !it.output) startPolling(id);
}
function restoreForm(it) {
  const r = it.request; if (!r) return;
  const m = modelById(S.cat, r.model, r.engine); if (!m) return;
  if (m.family !== S.family) { S.family = m.family; renderFamilies($("families"), S.cat, S.family); }
  pickModel(m.id);
  if (m.tasks.includes(r.task)) pickTask(r.task, true);
  $("prompt").value = r.prompt || ""; $("negative").value = r.negative_prompt || "";
  $("seed").value = r.seed ?? "";
  if (m.sizes.some(([w, h]) => w === r.width && h === r.height)) { S.size = [r.width, r.height]; renderSizes($("size"), m, S.size); }
  if ((m.frames.choices || []).includes(r.frames)) { S.frames = r.frames; renderLengths($("length"), m, S.frames); }
  for (const key of m.knobs || []) {
    const short = key.slice(key.indexOf(".") + 1);
    if (isUpscaleKnob(key)) S.values[key] = String(r.upscale || 0);
    else if (r.options && short in r.options) S.values[key] = r.options[short];
  }
  renderKnobs($("knobs"), S.cat, m, S.values);
  markChip();
  refreshForm();
}
function renderAll() {
  renderHistory($("history"), S.items, S.current);
  refreshInputs();
  renderClip(findItem(S.items, S.current));
}
function renderClip(it) {
  const o = it?.output;
  const expired = it ? isExpired(it) : false;
  const w = o?.width || it?.request?.width, h = o?.height || it?.request?.height;
  showClip($("screen"), $("video"), $("empty"), expired ? null : o, aspectOf(w, h));
  if (!it) $("empty").textContent = "Your clip plays here when the endpoint has uploaded it.";
  else if (expired) $("empty").textContent = "This clip's links expired (24 h). Generate it again from the form.";
  else if (!o) $("empty").textContent = it.status === "IN_QUEUE" ? "Waiting for a worker…" : isTerminal(it.status) ? (it.error || "No clip.") : `${it.progress?.stage || "starting"}${it.progress?.detail ? " · " + it.progress.detail : ""}`;
  $("summary").textContent = it ? summaryText(it) : "";
  renderStages($("stages"), it ? { status: it.status, progress: it.progress, output: o, error: it.error, delayTime: it.delayTime, track: it.track, upscale: it.upscale ?? it.request?.upscale } : null);
  $("cancel").hidden = !it || isTerminal(it.status);
  const links = [];
  if (o?.video_url && !expired) {
    links.push(`<a href="${o.video_url}" download="${slug(it)}.mp4">Download MP4</a>`);
    if (o.audio_url) links.push(`<a href="${o.audio_url}" download="${slug(it)}.wav">Download WAV</a>`);
    links.push(`<a href="${o.video_url}" target="_blank" rel="noopener">Open in new tab</a>`);
  }
  $("links").innerHTML = links.join("");
  const hint = [];
  if (it?.delayTime != null && it.status !== "IN_QUEUE") hint.push(`queue ${fmt(it.delayTime / 1000)}`);
  if (it?.status === "IN_PROGRESS" && it.progress?.elapsed_s != null) hint.push(`${fmt(it.progress.elapsed_s)} elapsed`);
  if (o?.expires_at && !expired) hint.push(`links expire ${new Date(o.expires_at).toLocaleString()}`);
  $("clip-hint").textContent = hint.join(" · ");
  showJobError(it && isTerminal(it.status) && it.status !== "COMPLETED" ? (it.error || `Job ${it.status.toLowerCase().replace("_", " ")}.`) : "");
  $("job-cmd-row").hidden = !o?.command;
  $("job-cmd").textContent = o?.command || "";
}
function showJobError(m) { $("job-err").textContent = m; $("job-err").hidden = !m; }
function slug(it) { return (it?.prompt || "clip").toLowerCase().replace(/[^a-z0-9]+/g, "-").slice(0, 40).replace(/-+$/, "") || "clip"; }

// ---- endpoint info -------------------------------------------------------------------------
const INFO_LABELS = { gpu: "GPU", driver: "Driver", image: "Image", catalog_version: "Catalog", versions: "Binaries", sidecars: "Resident sidecars", resident: "Resident sidecars", trees: "Trees", staged: "Staged", volume: "On the volume", vram_gib: "VRAM (GB)", free_vram_gib: "Free VRAM (GB)" };
function renderInfo(info, note) {
  const kv = $("info-kv");
  if (!info) { kv.innerHTML = ""; $("info-sub").textContent = note || ""; return; }
  const rows = [];
  for (const [k, v] of Object.entries(info)) {
    if (v == null || v === "") continue;
    let text;
    if (Array.isArray(v)) text = v.length ? v.map((x) => (typeof x === "object" ? JSON.stringify(x) : String(x))).join(", ") : "none";
    else if (typeof v === "object") text = Object.entries(v).map(([a, b]) => `${a}: ${typeof b === "object" ? JSON.stringify(b) : b}`).join(" · ");
    else text = String(v);
    rows.push(`<dt>${esc(INFO_LABELS[k] || k.replace(/_/g, " "))}</dt><dd>${esc(text)}</dd>`);
  }
  kv.innerHTML = rows.join("");
  $("info-sub").textContent = note || "";
}
async function refreshInfo() {
  if (S.locked) { lock(); return; }
  $("info-go").disabled = true; $("info-sub").textContent = "· asking a worker…";
  try {
    const info = await api.getInfo();
    api.store(INFO_KEY, JSON.stringify({ at: new Date().toISOString(), info }));
    renderInfo(info, `· ${new Date().toLocaleTimeString()}`);
  } catch (e) {
    if (e instanceof api.AuthError) lock("The studio token was refused.");
    else renderInfo(savedInfo()?.info, `· failed: ${e.message}`);
  }
  $("info-go").disabled = false;
}
function savedInfo() { try { return JSON.parse(api.store(INFO_KEY) || "null"); } catch { return null; } }

const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

async function copyText(text, button) {
  try { await navigator.clipboard.writeText(text); button.textContent = "Copied"; }
  catch { button.textContent = "Select and copy"; }
  setTimeout(() => { button.textContent = "Copy"; }, 1500);
}

// ---- boot -------------------------------------------------------------------------------------
async function boot() {
  try { S.cat = await api.getCatalog(); }
  catch (e) { setEngine("bad", `The catalog did not load: ${e.message}`); return; }
  const saved = loadForm();
  if (saved.prompt) $("prompt").value = saved.prompt; else $("prompt").value = EXAMPLES[0][1];
  if (saved.negative) $("negative").value = saved.negative;
  pickFamily(saved.family || families(S.cat)[0].id);
  renderHistory($("history"), S.items, S.current);
  renderInfo(savedInfo()?.info, savedInfo() ? `· ${new Date(savedInfo().at).toLocaleString()}` : "· not asked yet");
  renderStages($("stages"), null);
  for (const el of $("backends").querySelectorAll("[data-backend]")) el.setAttribute("aria-pressed", String(el.dataset.backend === S.backend));
  $("podbox").hidden = S.backend !== "pod";
  renderMode();
  if (S.locked) lock(); else { setEngine("wait", "Checking the endpoint…"); refreshHealth(); resumePolling(); if (S.backend === "pod") refreshPod(); }
  if (S.items[0]) openItem(S.items[0].id);
}

$("backends").onclick = (e) => { const b = e.target.closest("[data-backend]")?.dataset.backend; if (b && b !== S.backend) pickBackend(b); };
$("modes").onclick = (e) => { const m = e.target.closest("[data-mode]")?.dataset.mode; if (m && m !== S.mode) pickMode(m); };
$("live-steer").onclick = () => liveCommand("steer");
$("live-pause").onclick = () => liveCommand("pause");
$("live-resume").onclick = () => liveCommand("resume");
$("live-reset").onclick = () => liveCommand("reset");
$("live-stop").onclick = stopLive;
$("seg-next").onclick = nextSegment;
$("seg-close").onclick = closeSegments;
$("seg-video-a").addEventListener("ended", onSegmentEnded);
$("seg-video-b").addEventListener("ended", onSegmentEnded);
$("seg-video-a").addEventListener("loadeddata", () => { if (S.seg?.waiting) { S.seg.waiting = false; playNextSegment(); } });
$("seg-video-b").addEventListener("loadeddata", () => { if (S.seg?.waiting) { S.seg.waiting = false; playNextSegment(); } });
$("pod-start").onclick = startPod;
$("pod-stop").onclick = stopPod;
$("pod-refresh").onclick = refreshPod;

$("families").onclick = (e) => { const f = e.target.closest("[data-family]")?.dataset.family; if (f) pickFamily(f); };
$("models").onclick = (e) => {
  const more = e.target.closest("[data-more]");
  if (more) { S.cardLimit = more.dataset.more === "all" ? 0 : CARD_LIMIT; renderModels($("models"), S.cat, S.family, S.model?.id, S.cardLimit); return; }
  const m = e.target.closest("[data-model]")?.dataset.model; if (m && m !== S.model?.id) pickModel(m);
};
$("tasks").onclick = (e) => { const t = e.target.closest("[data-task]")?.dataset.task; if (t) pickTask(t); };
$("examples").onclick = (e) => { const p = e.target.closest("[data-prompt]")?.dataset.prompt; if (p) { $("prompt").value = p; markChip(); refreshForm(); $("prompt").focus(); } };
$("prompt").addEventListener("input", () => { markChip(); refreshForm(); });
$("negative").addEventListener("input", refreshForm);
$("prompt").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) generate(e); });
$("size").onchange = () => { S.size = $("size").value.split("x").map(Number); refreshForm(); };
$("length").onchange = () => { S.frames = Number($("length").value); refreshForm(); };
$("seed").oninput = refreshForm;
$("dice").onclick = () => { $("seed").value = Math.floor(Math.random() * 1e9); refreshForm(); };
$("knobs").addEventListener("input", onKnobInput);
$("knobs").addEventListener("change", onKnobInput);
$("reset").onclick = resetKnobs;
$("form").addEventListener("submit", generate);
$("copy").onclick = () => copyText($("cmd").textContent, $("copy"));
$("job-copy").onclick = () => copyText($("job-cmd").textContent, $("job-copy"));
$("cancel").onclick = cancel;
$("unlock-go").onclick = unlock;
$("token").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); unlock(); } });
$("history").onclick = (e) => { const id = e.target.closest("[data-id]")?.dataset.id; if (id) reopen(id); };
$("info-go").onclick = refreshInfo;
$("inputs").addEventListener("click", (e) => {
  const rm = e.target.closest("[data-remove]");
  if (rm) { e.preventDefault(); removeInput(rm.dataset.remove, Number(rm.dataset.index)); return; }
  if (e.target.closest("select, input")) return;
  const zone = e.target.closest("[data-input]"); if (!zone) return;
  zone.querySelector("input[type=file]").click();
});
$("inputs").addEventListener("keydown", (e) => { if ((e.key === "Enter" || e.key === " ") && e.target.matches("[data-input]")) { e.preventDefault(); e.target.querySelector("input[type=file]").click(); } });
$("inputs").addEventListener("change", (e) => {
  const file = e.target.closest("[data-file-for]");
  if (file && file.files?.length) { addFiles(file.dataset.fileFor, [...file.files]); return; }
  const clip = e.target.closest("[data-clip-for]");
  if (clip && clip.value) useSessionClip(clip.dataset.clipFor, clip.value);
});
for (const ev of ["dragenter", "dragover"]) $("inputs").addEventListener(ev, (e) => { const z = e.target.closest("[data-input]"); if (z) { e.preventDefault(); z.classList.add("over"); } });
$("inputs").addEventListener("dragleave", (e) => { e.target.closest("[data-input]")?.classList.remove("over"); });
$("inputs").addEventListener("drop", (e) => { const z = e.target.closest("[data-input]"); if (!z) return; e.preventDefault(); z.classList.remove("over"); if (e.dataTransfer?.files?.length) addFiles(z.dataset.input, [...e.dataTransfer.files]); });
window.addEventListener("storage", (e) => { if (e.key === api.TOKEN_KEY && !e.newValue) lock(); });

boot();
