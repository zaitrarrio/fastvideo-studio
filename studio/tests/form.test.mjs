// Unit tests for the pure helpers in studio/web/js/form.js against the real catalog.
import { test, describe } from "node:test";
import assert from "node:assert/strict";
import catalog from "../../catalog/catalog.json" with { type: "json" };
import { modelById, modelsForFamily, families, knobName } from "../web/js/catalog.js";
import { knobDefaults, knobVisible, buildOptions, buildRequest, cliLine, frameLabel, sizeLabel, aspectLabel, shellQuote, missingInputs, upscaleOf, hintFor } from "../web/js/form.js";

const v2 = modelById(catalog, "fasth3-v2");
const base = modelById(catalog, "h3");
const fw = modelById(catalog, "fastwan-1.3b");
const sf = modelById(catalog, "self-forcing-1.3b");
const wan = modelById(catalog, "wan22-ti2v-5b");
const regen = modelById(catalog, "regen-2k");

function state(model, extra = {}) {
  return { task: model.tasks[0], prompt: "A red fox", negativePrompt: "", size: model.default_size, frames: model.frames.default, seed: 7, inputs: {}, values: knobDefaults(catalog, model), ...extra };
}

describe("catalog lookups", () => {
  test("families come in the Studio order and hide hidden models", () => {
    assert.deepEqual(families(catalog).map((f) => f.id), ["h3", "fastwan", "ltx", "wan", "fv-h3"]);
    const fwModels = modelsForFamily(catalog, "fastwan").map((m) => m.id);
    assert.ok(!fwModels.includes("history-forcing-1.3b"), "hidden models are left out");
    assert.equal(fwModels[0], "fastwan-1.3b", "the recommended model comes first");
    assert.ok(!modelsForFamily(catalog, "wan").map((m) => m.id).includes("longlive-1.3b"));
  });
  test("knobName strips the engine prefix", () => {
    assert.equal(knobName("h3.weights"), "weights");
    assert.equal(knobName("fv.prompt_expansion"), "prompt_expansion");
  });
});

describe("knob defaults and `when`", () => {
  test("defaults come from catalog.knobs, overridden by model.knob_defaults", () => {
    const d = knobDefaults(catalog, v2);
    assert.equal(d["h3.weights"], "mxfp8");
    assert.equal(d["h3.vsa_qk8"], true);
    assert.equal(d["h3.attention"], "auto");
    assert.equal(d["h3.upscale_factor"], "0");
    const b = knobDefaults(catalog, base);
    assert.equal(b["h3.steps"], 25, "model.knob_defaults overrides the knob's 49");
    assert.equal(b["h3.attention"], "dense");
    const hf = knobDefaults(catalog, modelById(catalog, "history-forcing-1.3b"));
    assert.equal(hf["fw.cache_frames"], 21);
    assert.equal(knobDefaults(catalog, wan)["fv.steps"], null, "a null default stays unset");
  });
  test("`when` hides the VSA knobs unless attention is vsa", () => {
    const d = knobDefaults(catalog, v2);
    assert.equal(knobVisible(catalog, "h3.vsa_sparsity", d), false);
    assert.equal(knobVisible(catalog, "h3.dense_first_steps", d), false);
    assert.equal(knobVisible(catalog, "h3.weights", d), true);
    assert.equal(knobVisible(catalog, "h3.vsa_sparsity", { ...d, "h3.attention": "vsa" }), true);
    assert.equal(knobVisible(catalog, "fw.vsa_sparsity", { "fw.attention": "dense" }), false);
    assert.equal(knobVisible(catalog, "fw.vsa_sparsity", { "fw.attention": "vsa" }), true);
  });
  test("options use short names, drop hidden knobs and the upscale knob", () => {
    const d = knobDefaults(catalog, v2);
    const o = buildOptions(catalog, v2, d);
    assert.deepEqual(o, { weights: "mxfp8", decoder_weights: "mxfp8", vsa_qk8: true, pv8: true, attention: "auto", stream_blocks: false, tables: true, preview: false, prompt: "off" });
    const vsa = buildOptions(catalog, v2, { ...d, "h3.attention": "vsa", "h3.vsa_sparsity": "0.85", "h3.dense_first_steps": "2" });
    assert.equal(vsa.vsa_sparsity, 0.85);
    assert.equal(vsa.dense_first_steps, 2);
    assert.equal(upscaleOf(v2, { ...d, "h3.upscale_factor": "2" }), 2);
    assert.equal(upscaleOf(v2, d), 0);
    // `expand.prompt` is the one knob every model lists (short name `prompt`, default off).
    assert.deepEqual(buildOptions(catalog, wan, knobDefaults(catalog, wan)), { prompt: "off" }, "null defaults are not sent");
    assert.equal(o.prompt, "off");
  });
});

describe("labels", () => {
  test("frame labels use the model's fps", () => {
    assert.equal(frameLabel(124, 24), "124 frames · 5.2 s");
    assert.equal(frameLabel(81, 16), "81 frames · 5.1 s");
    assert.equal(frameLabel(48, 24), "48 frames · 2 s");
    assert.equal(frameLabel(969, 16), "969 frames · 60.6 s");
  });
  test("size labels give the aspect family and the tier", () => {
    assert.equal(sizeLabel([1344, 768]), "1344×768 · 16:9 · 768p");
    assert.equal(sizeLabel([768, 1344]), "768×1344 · 9:16 · 768p");
    assert.equal(sizeLabel([768, 768]), "768×768 · 1:1 · 768p");
    assert.equal(sizeLabel([832, 480]), "832×480 · 16:9 · 480p");
    assert.equal(sizeLabel([2560, 1440]), "2560×1440 · 16:9 · 1440p");
    assert.equal(sizeLabel([1280, 704]), "1280×704 · 16:9 · 704p");
    assert.equal(aspectLabel(1000, 300), "10:3");
  });
  test("the hint names steps, tier, fps and audio", () => {
    assert.match(hintFor(v2), /^8 denoising steps · turbo tier · 24 fps with audio\./);
    assert.match(hintFor(fw), /^3 denoising steps · standard tier · 16 fps, silent\./);
  });
  test("shellQuote escapes what a double-quoted shell string needs", () => {
    assert.equal(shellQuote('say "hi" $HOME `x`'), '"say \\"hi\\" \\$HOME \\`x\\`"');
  });
});

describe("the job input", () => {
  test("buildRequest follows jobs.md", () => {
    const req = buildRequest(catalog, v2, state(v2), { studio: "test", job: "j1" });
    assert.deepEqual(req, {
      op: "generate", engine: "h3", model: "fasth3-v2", task: "t2v", prompt: "A red fox", width: 1344, height: 768, frames: 124, seed: 7,
      inputs: {}, options: { weights: "mxfp8", decoder_weights: "mxfp8", vsa_qk8: true, pv8: true, attention: "auto", stream_blocks: false, tables: true, preview: false, prompt: "off" },
      upscale: 0, client: { studio: "test", job: "j1" },
    });
  });
  test("inputs follow the task spec; references stay a list; a blank seed is omitted", () => {
    const plug = modelById(catalog, "h3-plug4");
    const st = state(plug, { task: "fl2v", seed: "", inputs: { first_image: [{ b64: "AAAA", mime: "image/png" }], references: [{ url: "x" }] } });
    const req = buildRequest(catalog, plug, st);
    assert.deepEqual(req.inputs, { first_image: { b64: "AAAA", mime: "image/png" } }, "only the task's inputs are sent");
    assert.equal("seed" in req, false);
    const ref = buildRequest(catalog, plug, { ...st, task: "ref2v" });
    assert.deepEqual(ref.inputs, { references: [{ url: "x" }] });
    assert.deepEqual(missingInputs(catalog, "fl2v", {}), ["first_image"], "last_image is optional");
    assert.deepEqual(missingInputs(catalog, "fl2v", { first_image: [{}] }), []);
    assert.deepEqual(missingInputs(catalog, "regen2k", {}), ["draft"]);
  });
  test("the negative prompt is sent only for models that take one", () => {
    const withNeg = buildRequest(catalog, wan, state(wan, { negativePrompt: "blurry" }));
    assert.equal(withNeg.negative_prompt, "blurry");
    const noNeg = buildRequest(catalog, v2, state(v2, { negativePrompt: "blurry" }));
    assert.equal("negative_prompt" in noNeg, false);
  });
});

describe("the command line", () => {
  test("fasth3-v2 at the recipe defaults", () => {
    const st = state(v2);
    const line = cliLine(catalog, v2, buildRequest(catalog, v2, st), st);
    assert.equal(line, 'h3 generate --backend cuda --variant fasth3-v2 --model-dir /workspace/w/h3-8step --transformer /workspace/w/h3-8step-mxfp8/transformer.safetensors --tables /workspace/w/h3-rs/tables/v2.tables --weights mxfp8 --decoder-weights mxfp8 --vsa-qk8 --pv8 --width 1344 --height 768 --frames 124 --seed 7 --prompt "A red fox"');
  });
  test("knob changes show up: VSA with sparsity, no tables, bf16 weights, upscale", () => {
    const st = state(v2, { values: { ...knobDefaults(catalog, v2), "h3.attention": "vsa", "h3.vsa_sparsity": 0.9, "h3.dense_first_steps": 2, "h3.tables": false, "h3.weights": "native", "h3.upscale_factor": "2", "h3.stream_blocks": true } });
    const line = cliLine(catalog, v2, buildRequest(catalog, v2, st), st);
    assert.ok(!line.includes("--tables"), "tables off");
    assert.match(line, / --weights native /);
    assert.match(line, / --attention vsa --vsa-sparsity 0.9 --dense-first-steps 2 --stream-blocks /);
    assert.match(line, / --upscale realesr-animevideov3.safetensors --upscale-factor 2 /);
  });
  test("the base model: steps from the override, the recipe's dense attention through the knob", () => {
    const st = state(base, { task: "fl2v", inputs: { first_image: [{ b64: "x", mime: "image/png", name: "first frame.png" }] } });
    const line = cliLine(catalog, base, buildRequest(catalog, base, st), st);
    assert.match(line, /^h3 generate --backend cuda --variant h3 --model-dir \/workspace\/w\/h3-base --tables \/workspace\/w\/h3-rs\/tables\/base25.tables --task fl2va /);
    assert.match(line, / --attention dense /);
    assert.match(line, / --steps 25 /);
    assert.equal((line.match(/--attention/g) || []).length, 1, "the recipe flag is not duplicated");
    assert.match(line, / --first-image first_frame.png /);
  });
  test("Ref2VA tables and references for h3-plug4", () => {
    const plug = modelById(catalog, "h3-plug4");
    const st = state(plug, { task: "ref2v", inputs: { references: [{ b64: "x", mime: "image/jpeg", name: "cat.jpg", kind: "image" }, { r2: "uploads/x", mime: "video/mp4", name: "walk.mp4", kind: "video" }] } });
    const line = cliLine(catalog, plug, buildRequest(catalog, plug, st), st);
    assert.match(line, / --variant-dir \/workspace\/w\/h3-rs\/plug4 --tables \/workspace\/w\/h3-rs\/plug4\/plug4-ref.tables --task ref2va /);
    assert.match(line, / --ref image:cat.jpg --ref video:walk.mp4 /);
  });
  test("regen-2k uses h3 regen2k with the draft", () => {
    const st = state(regen, { inputs: { draft: [{ url: "https://r2/jobs/1/video.mp4?sig=1", name: "session clip", kind: "video" }] } });
    const line = cliLine(catalog, regen, buildRequest(catalog, regen, st), st);
    const adaln = regen.recipe.adaln_table ? ` --adaln-table ${regen.recipe.adaln_table}` : "";
    assert.equal(line, `h3 regen2k --backend cuda --regen-dir ${regen.recipe.regen_dir} --model-dir ${regen.recipe.model_dir} --draft video.mp4${adaln} --weights mxfp8 --pv8 --seed 7 --prompt "A red fox"`);
  });
  test("the fastwan recipe", () => {
    const st = state(fw);
    const line = cliLine(catalog, fw, buildRequest(catalog, fw, st), st);
    const r = fw.recipe;
    assert.equal(line, `fastwan generate --backend cuda --transformer ${r.transformer} --tae ${r.tae} --text-encoder ${r.model_dir}/text_encoder --tokenizer ${r.model_dir}/tokenizer/tokenizer.json --weights native --tae-chunk 2 --width 832 --height 480 --frames 81 --seed 7 --prompt "A red fox"`);
    const int8 = state(fw, { values: { ...knobDefaults(catalog, fw), "fw.weights": "int8", "fw.attention": "vsa", "fw.vsa_sparsity": 0.9, "fw.upscale_factor": "4" } });
    const l2 = cliLine(catalog, fw, buildRequest(catalog, fw, int8), int8);
    assert.match(l2, / --weights int8 --attention vsa --vsa-sparsity 0.9 --tae-chunk 2 --upscale realesr-animevideov3.safetensors --upscale-x 4 /);
  });
  test("Self-Forcing keeps the recipe's --causal --attention dense and its own paths", () => {
    const st = state(sf);
    const line = cliLine(catalog, sf, buildRequest(catalog, sf, st), st);
    const r = sf.recipe;
    assert.ok(line.startsWith(`fastwan generate --backend cuda --transformer ${r.transformer} --tae ${r.tae} --text-encoder ${r.text_encoder} --tokenizer ${r.tokenizer} --causal --attention dense --cache-frames 12 --sink-frames 3 --width 640 --height 352 --frames 165 --seed 7 `), line);
  });
  test("fastvideo models give a curl against fv-serve's jobs API", () => {
    const st = state(wan, { negativePrompt: "blurry", values: { ...knobDefaults(catalog, wan), "fv.steps": 40, "fv.guidance": 5 }, inputs: { first_image: [{ b64: "x", mime: "image/png", name: "a.png" }] }, task: "i2v" });
    const line = cliLine(catalog, wan, buildRequest(catalog, wan, st), st);
    assert.match(line, /^curl -X POST http:\/\/127\.0\.0\.1:8080\/fv\/v1\/jobs -H 'Content-Type: application\/json' -d '/);
    const body = JSON.parse(line.slice(line.indexOf("-d '") + 4, -1));
    assert.deepEqual(body, { model: "wan22-ti2v-5b", prompt: "A red fox", negative_prompt: "blurry", width: 1280, height: 704, num_frames: 121, seed: 7, steps: 40, guidance: 5, image_url: "https://…/a.png" });
    const ltx = modelById(catalog, "ltx25-distill-sol");
    const ls = state(ltx);
    const lb = JSON.parse(cliLine(catalog, ltx, buildRequest(catalog, ltx, ls), ls).split("-d '")[1].slice(0, -1));
    assert.equal(lb.fps, 24);
    assert.equal(lb.num_frames, 121);
  });
});
