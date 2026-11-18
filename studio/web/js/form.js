// The form: pure helpers (knob defaults, `when`, labels, the job input, the
// equivalent command line) and the renderers that build controls from the
// catalog. Only the renderers touch the DOM, and only inside functions, so
// the helpers run under `node --test` as they are.
import { knobDef, knobName, taskSpec, taskLabel, modelsForFamily, families } from "./catalog.js";

// ---- knobs --------------------------------------------------------------------
// Defaults for every knob the model lists: catalog.knobs[k].default, then
// model.knob_defaults[k] on top.
export function knobDefaults(cat, model) {
  const out = {};
  for (const key of model.knobs || []) {
    const def = knobDef(cat, key);
    if (!def) continue;
    out[key] = model.knob_defaults && key in model.knob_defaults ? model.knob_defaults[key] : def.default ?? null;
  }
  return out;
}

// `when: {attention: "vsa"}` names a sibling knob by its short name; the
// sibling is looked up with this knob's engine prefix ("h3.", "fw.", "fv.").
export function knobVisible(cat, key, values) {
  const def = knobDef(cat, key);
  if (!def?.when) return true;
  const prefix = key.includes(".") ? key.slice(0, key.indexOf(".") + 1) : "";
  return Object.entries(def.when).every(([k, want]) => {
    const have = values[prefix + k] ?? values[k];
    return Array.isArray(want) ? want.includes(have) : have === want;
  });
}

export function isUpscaleKnob(key) {
  return knobName(key) === "upscale_factor";
}

// The job's `options`: the model's knobs by short name, hidden ones and the
// upscale knob (a top-level field) left out, unset values (null) left out.
export function buildOptions(cat, model, values) {
  const out = {};
  for (const key of model.knobs || []) {
    if (isUpscaleKnob(key) || !knobVisible(cat, key, values)) continue;
    const v = values[key];
    if (v === null || v === undefined || v === "") continue;
    out[knobName(key)] = castKnob(knobDef(cat, key), v);
  }
  return out;
}

export function castKnob(def, v) {
  if (!def) return v;
  if (def.type === "bool") return v === true || v === "true" || v === 1 || v === "1";
  if (def.type === "int") return Math.round(Number(v));
  if (def.type === "float") return Number(v);
  return String(v);
}

export function upscaleOf(model, values) {
  const key = (model.knobs || []).find(isUpscaleKnob);
  return key ? Number(values[key] || 0) || 0 : 0;
}

// ---- labels -------------------------------------------------------------------
const ASPECTS = [["16:9", 16 / 9], ["9:16", 9 / 16], ["1:1", 1], ["4:3", 4 / 3], ["3:4", 3 / 4], ["21:9", 21 / 9], ["3:2", 3 / 2], ["2:3", 2 / 3]];

export function aspectLabel(w, h) {
  const r = w / h;
  const hit = ASPECTS.find(([, v]) => Math.abs(r - v) / v <= 0.03);
  if (hit) return hit[0];
  const g = gcd(w, h);
  return `${w / g}:${h / g}`;
}

function gcd(a, b) { return b ? gcd(b, a % b) : a; }

// "1344×768 · 16:9 · 768p"
export function sizeLabel([w, h]) {
  return `${w}×${h} · ${aspectLabel(w, h)} · ${Math.min(w, h)}p`;
}

// "124 frames · 5.2 s"
export function frameLabel(frames, fps) {
  const s = frames / (fps || 24);
  return `${frames} frames · ${fmtSeconds(s)} s`;
}

export function fmtSeconds(s) {
  const r = Math.round(s * 10) / 10;
  return Number.isInteger(r) ? String(r) : r.toFixed(1);
}

export function tierOf(model) {
  return model.tier || "standard";
}

export function hintFor(model) {
  const steps = model.steps ? `${model.steps} denoising step${model.steps === 1 ? "" : "s"}` : "steps set by the recipe";
  const audio = model.audio ? " with audio" : ", silent";
  return `${steps} · ${tierOf(model)} tier · ${model.fps} fps${audio}. Ctrl/⌘ + Enter generates.`;
}

// ---- the job input -------------------------------------------------------------
// `state`: { task, prompt, negativePrompt, size: [w, h], frames, seed, inputs: {name: [<media>, …]}, values: {knobKey: v} }
export function buildRequest(cat, model, state, client) {
  const spec = taskSpec(cat, state.task);
  const req = {
    op: "generate",
    engine: model.engine,
    model: model.id,
    task: state.task,
    prompt: state.prompt.trim(),
    width: state.size[0],
    height: state.size[1],
    frames: Number(state.frames),
  };
  if (model.negative_prompt && state.negativePrompt?.trim()) req.negative_prompt = state.negativePrompt.trim();
  if (state.seed !== "" && state.seed != null && Number.isFinite(Number(state.seed))) req.seed = Math.max(0, Math.floor(Number(state.seed)));
  const inputs = {};
  for (const name of spec.inputs || []) {
    const list = state.inputs?.[name] || [];
    if (!list.length) continue;
    inputs[name] = name === "references" ? list : list[0];
  }
  req.inputs = inputs;
  req.options = buildOptions(cat, model, state.values);
  req.upscale = upscaleOf(model, state.values);
  if (client) req.client = client;
  return req;
}

// Inputs the task needs that the form has not got.
export function missingInputs(cat, task, inputs) {
  const spec = taskSpec(cat, task);
  const optional = new Set(spec.optional || []);
  return (spec.inputs || []).filter((n) => !optional.has(n) && !(inputs?.[n]?.length));
}

// ---- the command line ------------------------------------------------------------
export function shellQuote(s) {
  return `"${String(s).replace(/(["\\$`])/g, "\\$1").replace(/\n/g, "\\n")}"`;
}

const H3_TASKS = { t2v: "t2va", fl2v: "fl2va", ref2v: "ref2va" };

// Recipe flags whose names a listed knob already covers are left to the
// knobs; the rest (`--causal`, a fixed `--attention dense`) pass through.
function recipeFlags(cat, model) {
  const covered = new Set((model.knobs || []).map((k) => knobDef(cat, k)?.cli).filter(Boolean));
  const out = [];
  let cur = null;
  for (const tok of model.recipe?.flags || []) {
    if (tok.startsWith("--")) { cur = covered.has(tok) ? null : tok; if (cur) out.push(cur); }
    else if (cur) out.push(tok);
  }
  return out;
}

function knobFlags(cat, model, values, engine) {
  const out = [];
  for (const key of model.knobs || []) {
    const def = knobDef(cat, key);
    if (!def || !knobVisible(cat, key, values)) continue;
    const v = values[key];
    if (v === null || v === undefined || v === "") continue;
    if (isUpscaleKnob(key)) {
      const k = Number(v) || 0;
      if (k) out.push("--upscale", "realesr-animevideov3.safetensors", engine === "fastwan" ? "--upscale-x" : "--upscale-factor", String(k));
      continue;
    }
    if (!def.cli) continue;
    if (def.type === "bool") { if (castKnob(def, v)) out.push(def.cli); continue; }
    if (def.type === "select" && v === "auto") continue;
    out.push(def.cli, String(castKnob(def, v)));
  }
  return out;
}

// A picked file's name for the line, or a placeholder.
function fileName(media, fallback) {
  const n = (media?.url ? media.url.split("?")[0].split("/").pop() : "") || media?.name || fallback;
  return n.replace(/[^\w.-]+/g, "_");
}

function h3Line(cat, model, req, state) {
  const r = model.recipe || {};
  const values = state.values;
  const inputs = state.inputs || {};
  if (req.task === "regen2k" || r.regen_dir) {
    const a = ["h3", "regen2k", "--backend", "cuda", "--regen-dir", r.regen_dir, "--model-dir", r.model_dir, "--draft", fileName(inputs.draft?.[0], "draft.y4m")];
    if (r.adaln_table) a.push("--adaln-table", r.adaln_table);
    a.push(...recipeFlags(cat, model), ...knobFlags(cat, model, values, "h3"));
    if (req.seed != null) a.push("--seed", String(req.seed));
    a.push("--prompt", shellQuote(req.prompt));
    return a.join(" ");
  }
  const a = ["h3", "generate", "--backend", "cuda", "--variant", r.variant || model.id, "--model-dir", r.model_dir];
  if (r.variant_dir) a.push("--variant-dir", r.variant_dir);
  if (r.transformer) a.push("--transformer", r.transformer);
  if (r.adapter) a.push("--adapter", r.adapter);
  const tablesKey = (model.knobs || []).find((k) => knobName(k) === "tables");
  const wantTables = tablesKey ? castKnob(knobDef(cat, tablesKey), values[tablesKey]) : false;
  const tables = req.task === "ref2v" && r.tables_ref ? r.tables_ref : r.tables;
  if (wantTables && tables) a.push("--tables", tables);
  if (req.task !== "t2v" && H3_TASKS[req.task]) a.push("--task", H3_TASKS[req.task]);
  a.push(...recipeFlags(cat, model), ...knobFlags(cat, model, values, "h3"));
  if (inputs.first_image?.[0]) a.push("--first-image", fileName(inputs.first_image[0], "first.png"));
  if (inputs.last_image?.[0]) a.push("--last-image", fileName(inputs.last_image[0], "last.png"));
  (inputs.references || []).forEach((m, i) => {
    const kind = m.kind || (m.mime || "").split("/")[0] || "image";
    a.push("--ref", `${kind === "video" ? "video" : kind === "audio" ? "audio" : "image"}:${fileName(m, `ref${i + 1}.${kind === "video" ? "y4m" : kind === "audio" ? "wav" : "png"}`)}`);
  });
  if (inputs.video?.[0]) a.push("--continue", fileName(inputs.video[0], "clip.y4m"));
  a.push("--width", String(req.width), "--height", String(req.height), "--frames", String(req.frames));
  if (req.seed != null) a.push("--seed", String(req.seed));
  a.push("--prompt", shellQuote(req.prompt));
  return a.join(" ");
}

function fastwanLine(cat, model, req, state) {
  const r = model.recipe || {};
  const dir = r.model_dir || "";
  const a = ["fastwan", "generate", "--backend", "cuda"];
  a.push("--transformer", r.transformer || `${dir}/transformer`, "--tae", r.tae || `${dir}/taew2_1.safetensors`);
  a.push("--text-encoder", r.text_encoder || `${dir}/text_encoder`, "--tokenizer", r.tokenizer || `${dir}/tokenizer/tokenizer.json`);
  a.push(...recipeFlags(cat, model), ...knobFlags(cat, model, state.values, "fastwan"));
  a.push("--width", String(req.width), "--height", String(req.height), "--frames", String(req.frames));
  if (req.seed != null) a.push("--seed", String(req.seed));
  a.push("--prompt", shellQuote(req.prompt));
  return a.join(" ");
}

const FV_INPUTS = { first_image: "image_url", last_image: "last_image_url", audio: "audio_url", video: "video_url", draft: "video_url" };

function fastvideoLine(cat, model, req, state) {
  const body = { model: model.recipe?.variant || model.id, prompt: req.prompt };
  if (req.negative_prompt) body.negative_prompt = req.negative_prompt;
  body.width = req.width; body.height = req.height; body.num_frames = req.frames;
  if (req.seed != null) body.seed = req.seed;
  for (const key of model.knobs || []) {
    const def = knobDef(cat, key);
    if (!def?.field || !knobVisible(cat, key, state.values)) continue;
    const v = state.values[key];
    if (v === null || v === undefined || v === "") continue;
    body[def.field] = def.type === "select" && /^\d+$/.test(String(v)) ? Number(v) : castKnob(def, v);
  }
  const inputs = state.inputs || {};
  for (const [name, field] of Object.entries(FV_INPUTS)) if (inputs[name]?.[0]) body[field] = `https://…/${fileName(inputs[name][0], name)}`;
  if (inputs.references?.length) body.reference_urls = inputs.references.map((m, i) => `https://…/${fileName(m, `ref${i + 1}`)}`);
  return `curl -X POST http://127.0.0.1:8080/fv/v1/jobs -H 'Content-Type: application/json' -d '${JSON.stringify(body).replace(/'/g, "'\\''")}'`;
}

export function cliLine(cat, model, req, state) {
  if (model.engine === "h3") return h3Line(cat, model, req, state);
  if (model.engine === "fastwan") return fastwanLine(cat, model, req, state);
  return fastvideoLine(cat, model, req, state);
}

// ---- renderers (DOM) -------------------------------------------------------------------
const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

export function renderFamilies(root, cat, current) {
  root.innerHTML = families(cat).map((f) => `<button type="button" data-family="${esc(f.id)}" aria-pressed="${f.id === current}">${esc(f.label)}</button>`).join("");
}

export function modelPills(model) {
  const pills = [];
  if (model.recommended) pills.push(`<span class="pill ok">Recommended</span>`);
  if (model.tier) pills.push(`<span class="pill">${esc(model.tier)}</span>`);
  if (model.audio) pills.push(`<span class="pill muted">audio</span>`);
  if (model.verify) pills.push(`<span class="pill warn" title="${esc(model.verify)}">unverified recipe</span>`);
  return pills.join("");
}

// The first `limit` cards (the current model always among them) and a
// "Show N more" toggle; `limit` 0 shows every model of the family.
export const CARD_LIMIT = 4;
export function renderModels(root, cat, familyId, currentId, limit = CARD_LIMIT) {
  const all = modelsForFamily(cat, familyId);
  let list = all;
  if (limit && all.length > limit) {
    list = all.slice(0, limit);
    const cur = all.find((m) => m.id === currentId);
    if (cur && !list.includes(cur)) list = [...list.slice(0, limit - 1), cur];
  }
  const more = all.length - list.length;
  const toggle = all.length > CARD_LIMIT ? `<button type="button" class="ghost small cards-toggle" data-more="${limit ? "all" : "fewer"}" aria-expanded="${!limit}">${limit ? `Show ${more} more model${more === 1 ? "" : "s"}` : "Show fewer"}</button>` : "";
  root.innerHTML = list.map((m) => `<button type="button" class="card" data-model="${esc(m.id)}" aria-pressed="${m.id === currentId}" title="${esc(m.blurb || "")}">
    <span class="title">${esc(m.label)} ${modelPills(m)}</span>
    <span class="blurb">${esc(m.blurb || "")}</span>
    <span class="meta"><span>${m.steps ? `${m.steps} steps` : "recipe steps"}</span><span>${m.fps} fps</span><span>${esc((m.tasks || []).map((t) => taskLabel(cat, t)).join(", "))}</span>${m.vram_gib ? `<span>${m.vram_gib} GB</span>` : ""}</span>
  </button>`).join("") + toggle;
}

export function renderTasks(root, cat, model, current) {
  root.innerHTML = (model.tasks || []).map((t) => `<button type="button" class="chip" data-task="${esc(t)}" aria-pressed="${t === current}" title="${esc(taskSpec(cat, t).help || "")}">${esc(taskLabel(cat, t))}</button>`).join("");
}

export function renderSizes(select, model, current) {
  const sizes = model.sizes || [];
  select.innerHTML = sizes.map(([w, h]) => `<option value="${w}x${h}">${esc(sizeLabel([w, h]))}</option>`).join("");
  const want = current ? `${current[0]}x${current[1]}` : `${model.default_size?.[0]}x${model.default_size?.[1]}`;
  if ([...select.options].some((o) => o.value === want)) select.value = want;
  else select.selectedIndex = 0;
}

export function renderLengths(select, model, current) {
  const f = model.frames || {};
  const choices = f.choices?.length ? f.choices : [f.default];
  select.innerHTML = choices.map((n) => `<option value="${n}">${esc(frameLabel(n, model.fps))}</option>`).join("");
  const want = String(current ?? f.default ?? choices[0]);
  if ([...select.options].some((o) => o.value === want)) select.value = want;
  else select.selectedIndex = 0;
}

// One row per knob, by type; `when` hides a row whose condition is not met.
export function renderKnobs(root, cat, model, values) {
  root.innerHTML = (model.knobs || []).map((key) => {
    const def = knobDef(cat, key);
    if (!def) return "";
    const id = `knob-${key.replace(/\W/g, "-")}`;
    const v = values[key];
    const hidden = knobVisible(cat, key, values) ? "" : " hidden";
    const help = def.help ? `<p class="help">${esc(def.help)}</p>` : "";
    const group = def.group ? `<span class="group">${esc(def.group)}</span>` : "";
    let control;
    if (def.type === "bool") {
      control = `<label class="check" for="${id}"><input type="checkbox" id="${id}" data-knob="${esc(key)}" ${castKnob(def, v) ? "checked" : ""} title="${esc(def.help || def.label)}"><span>${esc(def.label)}</span></label>`;
      return `<div class="knob${hidden}" data-knob-row="${esc(key)}"><div class="top">${control}${group}</div>${help}</div>`;
    }
    if (def.type === "select") {
      control = `<select id="${id}" data-knob="${esc(key)}" title="${esc(def.help || def.label)}">${(def.options || []).map(([val, lab]) => `<option value="${esc(val)}" ${String(v) === String(val) ? "selected" : ""}>${esc(lab)}</option>`).join("")}</select>`;
    } else if (def.type === "float" && def.min != null && def.max != null) {
      const val = v ?? def.min;
      control = `<div class="range"><input type="range" id="${id}" data-knob="${esc(key)}" min="${def.min}" max="${def.max}" step="${def.step ?? 0.05}" value="${val}" title="${esc(def.help || def.label)}"><output for="${id}">${val}</output></div>`;
    } else {
      const attrs = [def.min != null ? `min="${def.min}"` : "", def.max != null ? `max="${def.max}"` : "", `step="${def.type === "int" ? 1 : def.step ?? "any"}"`].join(" ");
      control = `<input type="number" id="${id}" data-knob="${esc(key)}" ${attrs} value="${v ?? ""}" placeholder="${v == null ? "recipe" : ""}" title="${esc(def.help || def.label)}">`;
    }
    return `<div class="knob${hidden}" data-knob-row="${esc(key)}"><div class="top"><label class="label" for="${id}">${esc(def.label)}</label>${group}</div>${control}${help}</div>`;
  }).join("");
}

// Re-applies `when` after a knob changes without rebuilding the rows.
export function updateKnobVisibility(root, cat, model, values) {
  for (const key of model.knobs || []) {
    const row = root.querySelector(`[data-knob-row="${CSS.escape(key)}"]`);
    if (row) row.hidden = !knobVisible(cat, key, values);
  }
}

// Reads one knob control's value in the knob's type.
export function readKnob(def, el) {
  if (def.type === "bool") return el.checked;
  if (el.value === "") return null;
  return castKnob(def, el.value);
}

const INPUT_LABELS = { first_image: "First frame", last_image: "Last frame", references: "References", audio: "Audio", video: "Video", draft: "Draft clip" };
const INPUT_ACCEPT = { first_image: "image/*", last_image: "image/*", references: "image/*", audio: "audio/*", video: "video/*", draft: "video/*" };

export function inputAccept(name, engine) {
  if (name === "references" && engine === "h3") return "image/*,video/*,audio/*";
  return INPUT_ACCEPT[name] || "*/*";
}

export function inputLabel(name) {
  return INPUT_LABELS[name] || name;
}

// The task's drop zones. `picked`: {name: [{file?, url?, name, mime, thumb?}, …]};
// `clips`: session items with a video_url, offered for video / draft inputs.
export function renderInputs(root, cat, model, task, picked, clips) {
  const spec = taskSpec(cat, task);
  const optional = new Set(spec.optional || []);
  const names = spec.inputs || [];
  root.hidden = !names.length;
  root.innerHTML = names.map((name) => {
    const multi = name === "references";
    const list = picked[name] || [];
    const accept = inputAccept(name, model.engine);
    const what = multi ? (model.engine === "h3" ? "images, video or audio, in order" : "images") : accept.replace("/*", "").replace("image", "an image").replace("audio", "an audio file").replace("video", "a video");
    const sessionPick = (name === "video" || name === "draft") && clips.length
      ? `<span class="pick"><span class="what">or</span><select data-clip-for="${esc(name)}" aria-label="Use a clip from this session"><option value="">Use a clip from this session…</option>${clips.map((c) => `<option value="${esc(c.id)}">${esc(c.prompt.slice(0, 48))} · ${esc(c.model)}</option>`).join("")}</select></span>`
      : "";
    const thumbs = list.map((m, i) => `<span class="thumb">${m.thumb ? `<img src="${esc(m.thumb)}" alt="">` : ""}<span class="file" title="${esc(m.name)}">${esc(m.name)}</span><button type="button" class="x ghost" data-remove="${esc(name)}" data-index="${i}" aria-label="Remove ${esc(m.name)}">×</button></span>`).join("");
    return `<div class="drop" data-input="${esc(name)}" tabindex="0" role="button" aria-label="${esc(inputLabel(name))}: drop a file or pick one">
      <span class="head"><span class="label">${esc(inputLabel(name))}${optional.has(name) ? " · optional" : ""}</span><span class="what">${esc(what)}${multi ? "" : " · drop or click"}</span></span>
      ${thumbs ? `<span class="thumbs">${thumbs}</span>` : ""}
      ${sessionPick}
      <input type="file" data-file-for="${esc(name)}" accept="${esc(accept)}" ${multi ? "multiple" : ""} aria-label="${esc(inputLabel(name))} file">
    </div>`;
  }).join("");
  if (spec.help) root.insertAdjacentHTML("beforeend", `<p class="hint">${esc(spec.help)}</p>`);
}
