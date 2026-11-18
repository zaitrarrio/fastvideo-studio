#!/usr/bin/env node
// The Studio page against a mock of the Worker's API, driven by Playwright.
//
//   node studio/tests/screenshot.cjs                 H3 → FastH3 V2 → t2v, generate, wait for the
//                                                    clip, save docs/screenshot-{light,dark,phone}.png
//   node studio/tests/screenshot.cjs --family fastwan   check the form regenerates for FastWan (no screenshots)
//   node studio/tests/screenshot.cjs --family ltx       … and for LTX-2.5
//   node studio/tests/screenshot.cjs --serve [--port N] keep the mock server up for manual use
//
// The mock walks a job through the progress stages over ~3 s and completes
// with a small real MP4 (made with ffmpeg when it is installed, else the
// committed fixture). Token: `studio-test`. No npm deps beyond the preinstalled
// playwright (`require('playwright')`).
const http = require("node:http");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { execFileSync } = require("node:child_process");

const ROOT = path.resolve(__dirname, "..");
const WEB = path.join(ROOT, "web");
const CATALOG = path.join(ROOT, "..", "catalog", "catalog.json");
const DOCS = path.join(ROOT, "docs");
const TOKEN = "studio-test";
const JOB_SECONDS = 3;
const MIME = { ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".json": "application/json", ".mp4": "video/mp4", ".jpg": "image/jpeg", ".png": "image/png", ".wav": "audio/wav" };

const args = process.argv.slice(2);
const opt = (name, dflt) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : dflt; };
const FAMILY = opt("--family", "h3");
const SERVE = args.includes("--serve");
const PORT = Number(opt("--port", 0));

// ---- the mock media ----------------------------------------------------------------
function makeMedia() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "studio-mock-"));
  const mp4 = path.join(dir, "video.mp4"), jpg = path.join(dir, "thumb.jpg"), wav = path.join(dir, "audio.wav");
  try {
    execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=duration=1:size=320x180:rate=24", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
      "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", "-shortest", mp4], { stdio: "pipe" });
    execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-i", mp4, "-frames:v", "1", "-q:v", "6", jpg], { stdio: "pipe" });
    execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-i", mp4, "-vn", "-ac", "2", "-ar", "32000", wav], { stdio: "pipe" });
    return { dir, mp4, jpg, wav, made: "ffmpeg" };
  } catch {
    fs.copyFileSync(path.join(__dirname, "fixtures", "test.mp4"), mp4);
    fs.copyFileSync(path.join(__dirname, "fixtures", "thumb.jpg"), jpg);
    return { dir, mp4, jpg, wav: null, made: "fixture" };
  }
}

// ---- the mock job ----------------------------------------------------------------
// Stage → the fraction of JOB_SECONDS at which it starts.
const TIMELINE = [["staging", 0.1], ["loading", 0.22], ["encoding", 0.36], ["preparing", 0.46], ["denoising", 0.52], ["decoding", 0.76], ["muxing", 0.9], ["uploading", 0.95]];

function jobView(job, now, origin) {
  const t = (now - job.started) / 1000;
  const frac = t / JOB_SECONDS;
  if (job.cancelled) return { id: job.id, status: "CANCELLED", delayTime: 300 };
  if (frac < 0.1) return { id: job.id, status: "IN_QUEUE" };
  if (frac >= 1) {
    const i = job.input, up = i.upscale ? { factor: i.upscale, width: i.width * i.upscale, height: i.height * i.upscale } : null;
    return {
      id: job.id, status: "COMPLETED", delayTime: 300, executionTime: JOB_SECONDS * 1000,
      output: {
        engine: i.engine, model: i.model, task: i.task, seed: i.seed ?? 4242, width: i.width, height: i.height, frames: i.frames, fps: 24, audio: i.engine === "h3",
        video_url: `${origin}/mock/video.mp4?job=${job.id}`, audio_url: i.engine === "h3" ? `${origin}/mock/audio.wav?job=${job.id}` : undefined,
        thumbnail_url: `${origin}/mock/thumb.jpg?job=${job.id}`, upscale: up,
        timings: { stage_s: 0.4, load_s: 0.42, encode_s: 0.3, denoise_s: 0.72, decode_s: 0.42, upscale_s: 0, mux_s: 0.15, total_s: 2.7, steps_s: [0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09] },
        command: `${i.engine === "h3" ? "h3" : i.engine === "fastwan" ? "fastwan" : "fv-serve"} generate --backend cuda --variant ${i.model} --width ${i.width} --height ${i.height} --frames ${i.frames} --seed ${i.seed ?? 4242} --prompt "…"  # mock`,
        worker: { gpu: "NVIDIA RTX PRO 6000 Blackwell Server Edition", driver: "580.65.06", image: "sha-mock" },
        expires_at: new Date(now + 24 * 3600 * 1000).toISOString(),
        client: i.client,
      },
    };
  }
  let stage = "staging";
  for (const [s, at] of TIMELINE) if (frac >= at) stage = s;
  const p = { stage, elapsed_s: Math.round(t * 10) / 10, engine: job.input.engine, model: job.input.model };
  if (stage === "denoising") { p.total_steps = 8; p.step = Math.min(8, Math.floor((frac - 0.52) / 0.24 * 8) + 1); p.detail = `step ${p.step}/8`; }
  if (stage === "decoding") { p.total_frames = job.input.frames; p.frames = Math.min(job.input.frames, Math.floor((frac - 0.76) / 0.14 * job.input.frames)); }
  if (stage === "staging") p.detail = "h3-8step-mxfp8";
  return { id: job.id, status: "IN_PROGRESS", progress: p, delayTime: 300 };
}

// ---- the server ----------------------------------------------------------------
function startServer(media) {
  const catalog = JSON.parse(fs.readFileSync(CATALOG, "utf8"));
  const jobs = new Map();
  let n = 0;
  const send = (res, status, data) => { res.writeHead(status, { "Content-Type": "application/json", "Cache-Control": "no-store" }); res.end(JSON.stringify(data)); };
  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url, "http://x");
    const origin = `http://127.0.0.1:${server.address().port}`;
    if (url.pathname.startsWith("/mock/")) {
      const file = { "/mock/video.mp4": media.mp4, "/mock/thumb.jpg": media.jpg, "/mock/audio.wav": media.wav }[url.pathname];
      if (!file || !fs.existsSync(file)) { res.writeHead(404); res.end(); return; }
      res.writeHead(200, { "Content-Type": MIME[path.extname(file)], "Content-Length": fs.statSync(file).size, "Accept-Ranges": "none" });
      fs.createReadStream(file).pipe(res); return;
    }
    if (url.pathname.startsWith("/api/")) {
      if (url.pathname === "/api/catalog") return send(res, 200, catalog);
      if ((req.headers.authorization || "") !== `Bearer ${TOKEN}`) return send(res, 401, { error: "unlock the studio with its token" });
      const body = await new Promise((r) => { const c = []; req.on("data", (d) => c.push(d)); req.on("end", () => r(Buffer.concat(c))); });
      if (url.pathname === "/api/health") return send(res, 200, { label: "mock endpoint", workers: { idle: 1, ready: 0, running: 0 }, jobs: { inQueue: 0, inProgress: 0 } });
      if (url.pathname === "/api/jobs" && req.method === "POST") {
        const input = JSON.parse(body.toString() || "{}");
        const id = `mock-${++n}-${Date.now().toString(36)}`;
        jobs.set(id, { id, input, started: Date.now(), cancelled: false });
        return send(res, 202, { id, status: "IN_QUEUE" });
      }
      const m = url.pathname.match(/^\/api\/jobs\/([^/]+)(\/cancel)?$/);
      if (m) {
        const job = jobs.get(m[1]);
        if (!job) return send(res, 404, { error: "no such job" });
        if (m[2]) { job.cancelled = true; return send(res, 200, { id: job.id, status: "CANCELLED" }); }
        return send(res, 200, jobView(job, Date.now(), origin));
      }
      if (url.pathname === "/api/uploads") return send(res, 201, { r2: `uploads/2026-11/mock-${decodeURIComponent(req.headers["x-file-name"] || "file")}`, bytes: body.length, mime: req.headers["content-type"] });
      if (url.pathname === "/api/info") return send(res, 200, { gpu: "NVIDIA RTX PRO 6000 Blackwell Server Edition", driver: "580.65.06", image: "sha-mock", catalog_version: catalog.version, versions: { h3: "0.9.0", fastwan: "0.8.1", "fv-serve": "1.4.0" }, sidecars: ["h3 fasth3-v2 (ready)"], staged: ["h3-8step-mxfp8", "h3-8step"], volume: ["h3-base", "h3-ref2va", "h3-8step", "ltx25"] });
      return send(res, 404, { error: "no such route" });
    }
    // static
    let file = path.join(WEB, url.pathname === "/" ? "index.html" : url.pathname);
    if (!file.startsWith(WEB) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404); res.end("not found"); return; }
    res.writeHead(200, { "Content-Type": MIME[path.extname(file)] || "application/octet-stream", "Cache-Control": "no-store" });
    fs.createReadStream(file).pipe(res);
  });
  return new Promise((resolve) => server.listen(PORT, "127.0.0.1", () => resolve(server)));
}

// ---- the browser flow ----------------------------------------------------------------
async function unlock(page, base) {
  await page.goto(base + "/");
  await page.waitForSelector("#unlock:not([hidden])");
  await page.fill("#token", TOKEN);
  await page.click("#unlock-go");
  await page.waitForSelector("#eng-dot.ok", { timeout: 10000 });
}

async function describeForm(page) {
  return page.evaluate(() => ({
    family: document.querySelector('#families [aria-pressed="true"]')?.dataset.family,
    model: document.querySelector('#models [aria-pressed="true"]')?.dataset.model,
    models: [...document.querySelectorAll("#models [data-model]")].map((b) => b.dataset.model),
    task: document.querySelector('#tasks [aria-pressed="true"]')?.dataset.task,
    tasks: [...document.querySelectorAll("#tasks [data-task]")].map((b) => b.dataset.task),
    sizes: [...document.querySelectorAll("#size option")].map((o) => o.textContent),
    lengths: [...document.querySelectorAll("#length option")].map((o) => o.textContent),
    knobs: [...document.querySelectorAll("#knobs [data-knob-row]")].map((r) => r.dataset.knobRow + (r.hidden ? " (hidden)" : "")),
    negative: !document.getElementById("neg-row").hidden,
    inputs: [...document.querySelectorAll("#inputs [data-input]")].map((d) => d.dataset.input),
    hint: document.getElementById("steps-hint").textContent,
    cmd: document.getElementById("cmd").textContent,
  }));
}

async function checkFamily(page, base, family, expectModel) {
  await unlock(page, base);
  await page.click(`#families [data-family="${family}"]`);
  const f = await describeForm(page);
  const problems = [];
  if (f.family !== family) problems.push(`family ${f.family}`);
  if (f.model !== expectModel) problems.push(`default model ${f.model}, expected ${expectModel}`);
  if (!f.knobs.length) problems.push("no knobs rendered");
  if (!f.sizes.length || !f.lengths.length) problems.push("sizes / lengths empty");
  if (!f.cmd) problems.push("no command line");
  // Switch to another model of the family and back: the form must follow.
  const other = f.models.find((m) => m !== f.model);
  if (other) {
    await page.click(`#models [data-model="${other}"]`);
    const g = await describeForm(page);
    if (g.model !== other) problems.push(`picking ${other} did not take`);
    if (g.cmd === f.cmd) problems.push("the command line did not change with the model");
    console.log(`  ${other}: tasks ${g.tasks.join(", ")} · ${g.knobs.length} knobs · ${g.sizes.length} sizes · inputs ${g.inputs.join(", ") || "none"}`);
  }
  console.log(`  ${f.model}: tasks ${f.tasks.join(", ")} · knobs ${f.knobs.join(", ")}`);
  console.log(`  sizes: ${f.sizes.join(" | ")}`);
  console.log(`  lengths: ${f.lengths.join(" | ")}`);
  console.log(`  hint: ${f.hint}`);
  console.log(`  cmd: ${f.cmd.slice(0, 160)}${f.cmd.length > 160 ? "…" : ""}`);
  if (problems.length) throw new Error(`${family}: ${problems.join("; ")}`);
  console.log(`  ${family}: form OK`);
}

async function h3Flow(browser, base) {
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 }, deviceScaleFactor: 1, colorScheme: "light" });
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  await unlock(page, base);
  await page.click('#families [data-family="h3"]');
  await page.click('#models [data-model="fasth3-v2"]');
  await page.click('#tasks [data-task="t2v"]');
  await page.click('#examples .chip >> nth=1');
  await page.fill("#seed", "7");
  const before = await describeForm(page);
  console.log(`  form: ${before.model} / ${before.task} · ${before.knobs.length} knobs · ${before.sizes[0]} · ${before.lengths[0]}`);
  console.log(`  cmd: ${before.cmd}`);
  await page.click("#go");
  await page.waitForSelector(".stage.active", { timeout: 5000 });
  const active = await page.$eval(".stage.active b", (b) => b.textContent);
  console.log(`  running: first active stage "${active}"`);
  await page.waitForFunction(() => document.querySelectorAll(".stage.done").length >= 8, null, { timeout: 15000 });
  await page.waitForSelector("#video:not([hidden])", { timeout: 15000 });
  await page.waitForFunction(() => document.getElementById("video").readyState >= 2, null, { timeout: 15000 });
  await page.waitForFunction(() => document.querySelectorAll("#history .clip").length >= 1);
  const summary = await page.$eval("#summary", (e) => e.textContent);
  const stages = await page.$$eval(".stage", (xs) => xs.map((x) => `${x.querySelector("b").textContent}=${x.querySelector("span").textContent}${x.classList.contains("done") ? "✓" : ""}`));
  console.log(`  summary: ${summary}`);
  console.log(`  stages: ${stages.join(" · ")}`);
  const links = await page.$$eval("#links a", (as) => as.map((a) => a.textContent));
  console.log(`  links: ${links.join(", ")}`);
  await page.evaluate(() => window.scrollTo(0, 0));
  fs.mkdirSync(DOCS, { recursive: true });
  await page.screenshot({ path: path.join(DOCS, "screenshot-light.png") });
  await page.emulateMedia({ colorScheme: "dark" });
  await page.waitForTimeout(150);
  await page.screenshot({ path: path.join(DOCS, "screenshot-dark.png") });
  if (process.env.STUDIO_FULLPAGE) {
    await page.emulateMedia({ colorScheme: "light" });
    await page.screenshot({ path: path.join(process.env.STUDIO_FULLPAGE, "full-light.png"), fullPage: true });
  }
  // The phone: same browser storage (token + session), 390×844.
  const storage = await page.evaluate(() => ({ token: localStorage.getItem("studio.token"), session: localStorage.getItem("studio.session"), form: localStorage.getItem("studio.form") }));
  await ctx.close();
  const phone = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 1, colorScheme: "light", isMobile: true, hasTouch: true });
  await phone.addInitScript((s) => { localStorage.setItem("studio.token", s.token); localStorage.setItem("studio.session", s.session); localStorage.setItem("studio.form", s.form); }, storage);
  const pp = await phone.newPage();
  pp.on("pageerror", (e) => errors.push(String(e)));
  await pp.goto(base + "/");
  await pp.waitForSelector("#eng-dot.ok", { timeout: 10000 });
  await pp.waitForFunction(() => document.querySelectorAll("#history .clip").length >= 1);
  await pp.screenshot({ path: path.join(DOCS, "screenshot-phone.png") });
  if (process.env.STUDIO_FULLPAGE) await pp.screenshot({ path: path.join(process.env.STUDIO_FULLPAGE, "full-phone.png"), fullPage: true });
  const overflow = await pp.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  console.log(`  phone horizontal overflow: ${overflow}px`);
  await phone.close();
  if (overflow > 0) throw new Error(`the phone layout scrolls horizontally by ${overflow}px`);
  if (errors.length) throw new Error(`page errors: ${errors.join(" | ")}`);
  console.log(`  screenshots: ${["light", "dark", "phone"].map((k) => path.relative(ROOT, path.join(DOCS, `screenshot-${k}.png`))).join(", ")}`);
}

async function main() {
  const media = makeMedia();
  const server = await startServer(media);
  const base = `http://127.0.0.1:${server.address().port}`;
  console.log(`mock studio at ${base} (token: ${TOKEN}; media via ${media.made})`);
  if (SERVE) { console.log("serving; Ctrl-C to stop"); return; }
  const { chromium } = require("playwright");
  const browser = await chromium.launch();
  try {
    if (FAMILY === "h3") await h3Flow(browser, base);
    else {
      const expect = { fastwan: "fastwan-1.3b", ltx: "ltx25-distill-sol", wan: "fastwan22-ti2v-5b", "fv-h3": "fasth3-4step-vsa" }[FAMILY];
      const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 }, deviceScaleFactor: 1 });
      const page = await ctx.newPage();
      const errors = [];
      page.on("pageerror", (e) => errors.push(String(e)));
      await checkFamily(page, base, FAMILY, expect);
      if (errors.length) throw new Error(`page errors: ${errors.join(" | ")}`);
      await ctx.close();
    }
  } finally {
    await browser.close();
    server.close();
    fs.rmSync(media.dir, { recursive: true, force: true });
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
