// The catalog (catalog/catalog.json through GET /api/catalog) and the small
// lookups the form needs. Pure: no DOM.

export const FAMILY_ORDER = ["h3", "fastwan", "ltx", "wan", "fv-h3"];

export function families(cat) {
  const list = [...cat.families];
  list.sort((a, b) => FAMILY_ORDER.indexOf(a.id) - FAMILY_ORDER.indexOf(b.id));
  return list;
}

export function familyById(cat, id) {
  return cat.families.find((f) => f.id === id) || null;
}

// The family's models in catalog order, hidden ones left out, the
// recommended one first.
export function modelsForFamily(cat, familyId) {
  const list = cat.models.filter((m) => m.family === familyId && !m.hidden);
  return [...list.filter((m) => m.recommended), ...list.filter((m) => !m.recommended)];
}

export function modelById(cat, id, engine) {
  return cat.models.find((m) => m.id === id && (!engine || m.engine === engine)) || null;
}

export function defaultModel(cat, familyId) {
  const fam = familyById(cat, familyId);
  const list = modelsForFamily(cat, familyId);
  return (fam && list.find((m) => m.id === fam.default_model)) || list[0] || null;
}

export function taskLabel(cat, task) {
  return cat.tasks[task]?.label || task;
}

export function taskSpec(cat, task) {
  return cat.tasks[task] || { label: task, inputs: [] };
}

export function knobDef(cat, key) {
  return cat.knobs[key] || null;
}

// A knob's short name as the job's `options` carries it: "h3.weights" → "weights".
export function knobName(key) {
  const i = key.indexOf(".");
  return i < 0 ? key : key.slice(i + 1);
}

export function engineLabel(cat, engine) {
  return cat.engines[engine]?.label || engine;
}

export function inlineLimit(cat) {
  return cat.upload_limits?.inline_b64_bytes ?? 6 * 1024 * 1024;
}

export function uploadLimit(cat) {
  return cat.upload_limits?.r2_upload_bytes ?? 512 * 1024 * 1024;
}
