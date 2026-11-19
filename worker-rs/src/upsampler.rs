//! Prompt expansion for every engine through a llama-server sidecar
//! (Qwen3-VL-4B-Instruct, Q8_0 + F16 vision projector): one chat completion
//! per job that rewrites the prompt in the spirit of the model family's prompt
//! guide, cleaned and held under the family's text-encoder budget. A port of
//! fastvideo-rs's batch rewrites (`crates/fastvideo-upsampler/src/rewrite.rs`,
//! `docs/serve/prompt-upsampling.md` "Batch rewrites") and of its launcher's
//! llama-server flags (`deploy/runpod/fv-upsampler.sh`), through the Python
//! worker's `fvstudio/upsampler.py`.
//!
//! The worker wires it in: `plan_sidecar` → `Sidecar(ENGINE, …, llama_probe)`
//! through `SidecarManager.ensure(…, vram_gib=VRAM_GIB)`, then
//! `Upsampler(base).expand(...)` before the engine runs. A rewrite that fails
//! for any reason returns the original prompt and records why
//! (`Upsampler.last_error`): the job never fails because the LLM did.

use crate::catalog::{b64_encode, Job};
use crate::httpc;
use crate::sidecar::Probe;
use crate::Log;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

// ---- the model tree (fastvideo-rs scripts/gpu/weights-manifest.tsv line 98,
// docs/serve/upsampler-local.md §1) --------------------------------------------
pub const TREE_NAME: &str = "qwen3-vl-4b-instruct-gguf";
pub const TREE: &str = "r2:fv-weights/weights/qwen3-vl-4b-instruct-gguf=qwen3-vl-4b-instruct-gguf";
pub const MODEL_FILE: &str = "Qwen3VL-4B-Instruct-Q8_0.gguf";
pub const MMPROJ_FILE: &str = "mmproj-Qwen3VL-4B-Instruct-F16.gguf";
pub const MODEL_BYTES: u64 = 4_280_406_144; // upsampler-local.md §1
pub const MMPROJ_BYTES: u64 = 836_180_256;

/// the first-use fetch: 4.77 GiB (5.12 GB)
pub fn fetch_gib() -> f64 {
    crate::py::round_to((MODEL_BYTES + MMPROJ_BYTES) as f64 / 2f64.powi(30), 2)
}

// ---- the sidecar (fv-upsampler.sh's defaults, except the context) -----------
/// the SidecarManager key; never an engine the catalog lists
pub const ENGINE: &str = "upsampler";
/// FV_UPSAMPLER_MODEL: llama-server's --alias, the `model` of every request
pub const ALIAS: &str = "qwen3-vl-4b-instruct";
/// in the worker image, from the fastvideo-rs serve image
pub const LLAMA_BIN: &str = "/opt/fastvideo-rs/bin/llama-server";
/// FV_UPSAMPLER_PORT
pub const DEFAULT_PORT: u16 = 18080;
/// FV_UPSAMPLER_CTX is 65536 for the director's sessions; one Studio rewrite needs far less
pub const CTX: u64 = 16384;
/// FV_UPSAMPLER_PARALLEL 2: the Studio rewrites one prompt at a time
pub const PARALLEL: u64 = 1;
pub const KV_TYPE: &str = "q8_0";
pub const CACHE_REUSE: u64 = 256;
pub const CACHE_RAM_MIB: u64 = 4096;

// ---- memory -------------------------------------------------------------------
// Qwen3-VL-4B's language tower: 36 layers, 8 KV heads of 128 (Qwen3-4B's
// config). KV per token = 2 (K, V) × 36 × 8 × 128 values; q8_0 stores 32
// values in 34 bytes (int8 + an f16 scale), f16 2 bytes a value. At 65536
// tokens this gives 4.78 GiB, the "~4.8 GB" fv-upsampler.sh states.
pub const KV_LAYERS: u64 = 36;
pub const KV_HEADS: u64 = 8;
pub const KV_HEAD_DIM: u64 = 128;
/// llama.cpp's compute buffers + the projector's image activations
pub const COMPUTE_GIB: f64 = 0.5;

fn kv_bytes_per_value(kv: &str) -> f64 {
    match kv {
        "q8_0" => 34.0 / 32.0,
        "f16" | "bf16" => 2.0,
        "q4_0" => 18.0 / 32.0,
        _ => 2.0,
    }
}

/// The KV cache in bytes for `ctx` tokens at the `kv` cache type.
pub fn kv_bytes(ctx: u64, kv: &str) -> u64 {
    (ctx as f64 * 2.0 * (KV_LAYERS * KV_HEADS * KV_HEAD_DIM) as f64 * kv_bytes_per_value(kv)) as u64
}

/// What `SidecarManager.ensure` reserves: the Q8_0 weights (3.99 GiB) +
/// the F16 projector (0.78 GiB) + the KV cache (1.20 GiB at 16384 tokens
/// q8_0) + compute buffers (0.5 GiB) = 6.5 GiB at the defaults.
pub fn vram_gib(ctx: u64, kv: &str, vision: bool) -> f64 {
    let total = MODEL_BYTES + if vision { MMPROJ_BYTES } else { 0 } + kv_bytes(ctx, kv);
    crate::py::round_to(total as f64 / 2f64.powi(30) + COMPUTE_GIB, 1)
}

pub fn default_vram_gib() -> f64 {
    vram_gib(CTX, KV_TYPE, true) // 6.5
}

// ---- the knob -------------------------------------------------------------------
/// catalog.json: a shared knob every model lists (the worker handles it, no cli / field)
pub const KNOB: &str = "expand.prompt";
pub const MODES: [&str; 3] = ["off", "balanced", "quality"];

/// The job's expansion mode: `off`, `balanced` or `quality` (a `Job`'s
/// effective options carry the knob under its full id; the page sends it
/// as `options.prompt`, which the catalog maps back).
pub fn mode_of(job: &Job) -> &'static str {
    let v = job.options.get(KNOB).filter(|v| crate::py::truthy(v)).map(crate::py::py_str).unwrap_or_else(|| "off".into()).to_lowercase();
    MODES.iter().find(|m| **m == v).copied().unwrap_or("off")
}

// ---- families -------------------------------------------------------------------
pub const FAMILIES: [&str; 4] = ["h3", "ltx", "wan", "other"];

/// The prompt family of a catalog family / engine id: `h3` (h3, fv-h3),
/// `wan` (fastwan, wan), `ltx`, else `other`.
pub fn family_of(name: Option<&str>) -> &'static str {
    match name.unwrap_or("").to_lowercase().as_str() {
        "h3" | "fv-h3" => "h3",
        "fastwan" | "wan" => "wan",
        "ltx" | "ltx2" => "ltx",
        _ => "other",
    }
}

fn family_norm(family: &str) -> &'static str {
    FAMILIES.iter().find(|f| **f == family).copied().unwrap_or("other")
}

// Prompt budgets in characters, fastvideo-rs's defaults without the family's
// tokenizer: Wan's umT5-XXL takes 512 tokens → 1800; LTX's Gemma 3 1024
// tokens → 3600; H3's Qwen3-VL text tower 1024 tokens (refused beyond) →
// 3200. The Studio has no tokenizer, so characters are the limit; the
// catalog caps the input at 2000 anyway.
pub fn max_chars(family: &str) -> usize {
    match family_norm(family) {
        "wan" => 1800,
        "ltx" => 3600,
        "h3" => 3200,
        _ => 1800,
    }
}

/// The answer's token cap: H3's 1024-token layout; Wan's 512.
pub fn max_tokens(family: &str) -> u64 {
    match family_norm(family) {
        "wan" => 512,
        "ltx" => 1024,
        "h3" => 1024,
        _ => 512,
    }
}

/// rewrite.rs `request`
pub const TEMPERATURE: f64 = 0.6;

/// Target length in words per (mode, family): rewrite.rs `Effort::words`.
pub fn words(mode: &str, family: &str) -> (u32, u32) {
    match (mode, family_norm(family)) {
        ("balanced", "wan") => (60, 100),
        ("quality", "wan") => (100, 160),
        ("balanced", "h3") => (60, 110),
        ("quality", "h3") => (120, 220),
        ("balanced", "ltx") => (80, 130),
        ("quality", "ltx") => (140, 200),
        ("balanced", _) => (80, 130),
        _ => (140, 200),
    }
}

// ---- the system prompts: fastvideo-rs crates/fastvideo-upsampler/src/rewrite.rs,
// verbatim through the Python port. ----------------------------------------------
pub const COMMON_RULES: &str = "Rules:
- Keep the user's subjects, intent, quoted text and any style or camera instruction.
- Add concrete visual detail only. One continuous shot, present tense.
- One English paragraph. No title, list, markdown or comments.
- Reply with the prompt only.";

pub const H3_RULES: &str = "Rules:
- Keep the user's subjects, intent, quoted text and any style or camera instruction.
- Add concrete visual detail only. Present tense.
- English, the three labels exactly as shown, each section on its own line. No markdown or comments.
- Reply with the prompt only.";

pub const WAN_GUIDE: &str = "Rewrite the user's request as a video prompt.
Order: subject and action, appearance, setting, style, shot and camera motion, lighting.
Stress motion: what moves, where, how fast. This model makes no sound: no audio, no dialogue.";

pub const LTX_GUIDE: &str = "Rewrite the user's request as a video prompt, the way a cinematographer describes a shot.
Start directly with the main action. Then movements, appearance, background, camera angle and movement, lighting and color.
End with the ambience and sound effects. Spoken words go in double quotes after the speaker.";

pub const H3_GUIDE: &str = "Rewrite the user's request as a video prompt for MiniMax H3 in its structured format, three labelled sections:
integrated_multimodal_description: the setting, each subject with a concrete look, the action through the shot in present tense, every spoken line inline as Name says <d>[English] the line</d>, then camera language (shot size, angle, push in, pan, tracking), lighting and style.
overall_soundscape: ambience and sound effects.
non_diegetic_music: the score, or None.";

pub const GENERIC_GUIDE: &str = "Rewrite the user's request as a detailed video prompt.
Describe the subject, action, setting, camera, lighting and mood.";

pub fn family_guide(family: &str) -> &'static str {
    match family_norm(family) {
        "wan" => WAN_GUIDE,
        "ltx" => LTX_GUIDE,
        "h3" => H3_GUIDE,
        _ => GENERIC_GUIDE,
    }
}

/// rewrite.rs line 115: the text-only image line (the LLM does not see the image).
pub const IMAGE_UNSEEN: &str = "\nThe first frame is an image you cannot see: do not describe its look; describe the motion and camera from it.";
/// Ours: the image is attached as a vision part (the projector is loaded), the
/// way fv-serve's director attaches an opening image.
pub const IMAGE_SEEN: &str = "\nThe first frame is the attached image: keep its subjects and look as they are; describe the motion and camera from it.";

pub fn h3_shots(mode: &str) -> &'static str {
    if mode == "quality" {
        "Be rich and specific. When the request has distinct shots, write the description as [Shot 1], [Shot 2] … blocks, each a few sentences; otherwise one paragraph."
    } else {
        "Keep it short: one paragraph of description."
    }
}

// H3's section labels (Context-IR).
pub const H3_DESCRIPTION: &str = "integrated_multimodal_description:";
pub const H3_SOUNDSCAPE: &str = "overall_soundscape:";
pub const H3_MUSIC: &str = "non_diegetic_music:";

/// The system prompt for one rewrite (rewrite.rs `system_prompt`):
/// the family's guide, the image line when the clip starts from an image
/// (`vision`: it is attached), the H3 shot rule, the rules and the
/// length bound. A negative prompt is not rewritten; when given, the LLM
/// is told what the user excludes so the rewrite does not add it.
pub fn system_prompt(family: &str, mode: &str, image: bool, vision: bool, negative: Option<&str>) -> String {
    let family = family_norm(family);
    let mode = if mode == "balanced" || mode == "quality" { mode } else { "balanced" };
    let (lo, hi) = words(mode, family);
    let max = max_chars(family);
    let image_line = if image { if vision { IMAGE_SEEN } else { IMAGE_UNSEEN } } else { "" };
    let neg = match negative.map(|n| n.split_whitespace().collect::<Vec<_>>().join(" ")).filter(|n| !n.is_empty()) {
        Some(n) => format!("\n- The user excludes: {n}. Do not add any of it."),
        None => String::new(),
    };
    if family == "h3" {
        return format!("{H3_GUIDE}{image_line}\n{}\n\n{H3_RULES}\n- {lo} to {hi} words, at most {max} characters.{neg}", h3_shots(mode));
    }
    format!("{}{image_line}\n\n{COMMON_RULES}\n- {lo} to {hi} words, at most {max} characters.{neg}", family_guide(family))
}

// ---- cleaning (rewrite.rs `strip_wrappers`, `clean`, `cut`, `H3Sections`) -------

/// A leading `<think>…</think>` block removed (chat.rs `strip_think`).
pub fn strip_think(s: &str) -> String {
    let t = s.trim_start();
    if let Some(rest) = t.strip_prefix("<think>") {
        return match rest.find("</think>") {
            None => String::new(),
            Some(i) => rest[i + "</think>".len()..].trim_start().to_string(),
        };
    }
    if t.contains("</think>") && !t.contains("<think>") {
        return t.split_once("</think>").unwrap().1.trim_start().to_string();
    }
    t.to_string()
}

const LABELS: [&str; 4] = ["rewritten prompt:", "enhanced prompt:", "expanded prompt:", "prompt:"];
const QUOTE_PAIRS: [(char, char); 3] = [('"', '"'), ('“', '”'), ('\'', '\'')];

/// Code fences, a leading `Prompt:` label and wrapping quotes removed
/// (line breaks kept).
pub fn strip_wrappers(answer: &str) -> String {
    let mut s = strip_think(answer).trim().to_string();
    if s.starts_with("```") {
        s = s.trim_start_matches('`').to_string();
        if let Some((first, rest)) = s.split_once('\n') {
            if !first.trim().contains(' ') {
                // a language tag on the fence line
                s = rest.to_string();
            }
        }
        s = s.trim_end().trim_end_matches('`').to_string();
    }
    s = s.trim().to_string();
    let low = s.to_lowercase();
    for label in LABELS {
        if low.starts_with(label) {
            s = s[label.len()..].trim_start().to_string();
            break;
        }
    }
    s = s.trim().to_string();
    let chars: Vec<char> = s.chars().collect();
    if chars.len() >= 2 {
        for (a, b) in QUOTE_PAIRS {
            if chars[0] == a && chars[chars.len() - 1] == b {
                let inner: String = chars[1..chars.len() - 1].iter().collect();
                // Only a wrapper: no other quote of that kind inside (dialogue keeps its quotes).
                if !inner.contains(a) && !inner.contains(b) {
                    s = inner;
                }
                break;
            }
        }
    }
    s
}

/// `s` within `max_chars`: whole when it fits, else up to the last
/// sentence end that fits (when that keeps at least half), else the last
/// whole word.
pub fn cut(s: &str, max_chars: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max_chars {
        return s.to_string();
    }
    let head = &chars[..max_chars];
    let rstrip = |v: &[char]| -> String { v.iter().collect::<String>().trim_end().to_string() };
    if let Some(i) = head.iter().rposition(|c| matches!(c, '.' | '!' | '?')) {
        if i + 1 >= head.len() / 2 {
            return rstrip(&head[..i + 1]);
        }
    }
    match head.iter().rposition(|c| *c == ' ') {
        Some(i) if i > 0 => rstrip(&head[..i]),
        _ => head.iter().collect(),
    }
}

/// A one-paragraph prompt: wrappers off, whitespace collapsed, cut to
/// `max_chars`. `None` when nothing is left.
pub fn clean(answer: &str, max_chars: usize) -> Option<String> {
    let joined = strip_wrappers(answer).split_whitespace().collect::<Vec<_>>().join(" ");
    if joined.is_empty() {
        return None;
    }
    Some(cut(&joined, max_chars))
}

fn squash(s: &str) -> String {
    s.lines().map(|l| l.split_whitespace().collect::<Vec<_>>().join(" ")).filter(|l| !l.is_empty()).collect::<Vec<_>>().join("\n")
}

/// The three sections of an H3 prompt (`[Shot N]` blocks stay in the
/// description, each on its own line).
#[derive(Debug, Clone, PartialEq)]
pub struct H3Sections {
    pub description: String,
    pub soundscape: Option<String>,
    pub music: Option<String>,
}

impl H3Sections {
    /// Splits an answer at the labels; text without labels is all description.
    pub fn parse(s: &str) -> H3Sections {
        let pos = |lab: &str| s.find(lab);
        let (pd, ps, pm) = (pos(H3_DESCRIPTION), pos(H3_SOUNDSCAPE), pos(H3_MUSIC));
        let mut found: Vec<usize> = [pd, ps, pm].into_iter().flatten().collect();
        found.sort();
        let next_after = |start: usize| found.iter().copied().find(|i| *i > start).unwrap_or(s.len());
        let desc = match pd {
            Some(d) => &s[d + H3_DESCRIPTION.len()..next_after(d)],
            None => match found.first() {
                Some(f) => &s[..*f],
                None => s,
            },
        };
        let section = |i: Option<usize>, label: &str| -> Option<String> {
            let i = i?;
            let t = squash(&s[i + label.len()..next_after(i)]).replace('\n', " ");
            if t.is_empty() { None } else { Some(t) }
        };
        H3Sections { description: squash(desc), soundscape: section(ps, H3_SOUNDSCAPE), music: section(pm, H3_MUSIC) }
    }

    pub fn render(&self) -> String {
        let mut v = vec![format!("{H3_DESCRIPTION} {}", self.description)];
        if let Some(s) = &self.soundscape {
            v.push(format!("{H3_SOUNDSCAPE} {s}"));
        }
        if let Some(m) = &self.music {
            v.push(format!("{H3_MUSIC} {m}"));
        }
        v.join("\n")
    }

    /// Within `max_chars` by priority: the music becomes `None.`, then
    /// the description is cut at a sentence (else word) boundary, never
    /// inside `<d>…</d>`; the soundscape is kept.
    pub fn fit(&mut self, max_chars: usize) -> String {
        let fits = |p: &str| p.chars().count() <= max_chars;
        let p = self.render();
        if fits(&p) {
            return p;
        }
        if self.music.as_ref().map(|m| m.trim_end_matches('.').to_lowercase() != "none").unwrap_or(false) {
            self.music = Some("None.".into());
            let p = self.render();
            if fits(&p) {
                return p;
            }
        }
        let desc = self.description.clone();
        let (sound, music) = (self.soundscape.clone(), self.music.clone());
        self.description = cut_to_fit(&desc, &|d: &str| fits(&H3Sections { description: d.to_string(), soundscape: sound.clone(), music: music.clone() }.render()));
        let p = self.render();
        if fits(&p) { p } else { cut_to_fit(&p, &fits) }
    }
}

/// The longest prefix of `s` that `fits`, ending at a sentence end
/// (else a word), never inside a `<d>…</d>` dialogue pair (tokens.rs
/// `cut_to_fit`).
pub fn cut_to_fit(s: &str, fits: &dyn Fn(&str) -> bool) -> String {
    if fits(s) {
        return s.to_string();
    }
    let mut inside = vec![false; s.len() + 1];
    let mut i = 0;
    while let Some(o) = s[i..].find("<d>").map(|x| x + i) {
        let end = s[o..].find("</d>").map(|e| o + e + 4).unwrap_or(s.len());
        for x in o + 1..end {
            inside[x] = true;
        }
        i = end;
        if i >= s.len() {
            break;
        }
    }
    let mut word_ends: Vec<usize> = Vec::new();
    let mut sentence_ends: Vec<usize> = Vec::new();
    for (b, c) in s.char_indices() {
        if inside[b] {
            continue;
        }
        if c == ' ' || c == '\n' {
            word_ends.push(b);
            let prev = s[..b].trim_end();
            if prev.ends_with('.') || prev.ends_with('!') || prev.ends_with('?') || prev.ends_with("</d>") {
                sentence_ends.push(prev.len());
            }
        }
    }
    let ok = |b: usize| fits(s[..b].trim_end());
    let best = |cands: &[usize]| -> Option<usize> {
        let (mut lo, mut hi) = (0usize, cands.len());
        while lo < hi {
            // shorter prefixes never exceed a longer one's length
            let mid = (lo + hi) / 2;
            if ok(cands[mid]) { lo = mid + 1 } else { hi = mid }
        }
        if lo > 0 { Some(cands[lo - 1]) } else { None }
    };
    let sent = best(&sentence_ends);
    let word = best(&word_ends);
    if let Some(se) = sent {
        if word.map(|w| se * 2 >= w).unwrap_or(true) {
            return s[..se].trim_end().to_string();
        }
    }
    if let Some(w) = word {
        return s[..w].trim_end().to_string();
    }
    String::new()
}

/// The H3 cleaning: the Context-IR sections kept on their own lines (an
/// answer without labels becomes the description), within `max_chars`.
pub fn clean_h3(answer: &str, max_chars: usize) -> Option<String> {
    let mut sec = H3Sections::parse(&strip_wrappers(answer));
    if sec.description.trim().is_empty() {
        return None;
    }
    Some(sec.fit(max_chars))
}

/// `clean` within the family's budget; H3 keeps its sections.
pub fn clean_for(family: &str, answer: &str, max: Option<usize>) -> Option<String> {
    let family = family_norm(family);
    let limit = max.unwrap_or_else(|| max_chars(family));
    if family == "h3" { clean_h3(answer, limit) } else { clean(answer, limit) }
}

// ---- the sidecar: argv, env, probe ------------------------------------------------

/// fv-upsampler.sh `launch`'s command line: model (+ projector), loopback
/// host, port, alias, every layer on the GPU, the context, the slots, a
/// KV per slot, flash attention with a quantized KV, cached prefix reuse,
/// no auto-fit, no web UI.
#[allow(clippy::too_many_arguments)]
pub fn llama_argv(bin: &str, model_path: &str, mmproj_path: Option<&str>, port: u16, ctx: u64, parallel: u64, kv: &str, alias: &str, cache_reuse: u64, cache_ram_mib: u64) -> Vec<String> {
    let mut argv: Vec<String> = vec![bin.into(), "-m".into(), model_path.into()];
    if let Some(m) = mmproj_path {
        argv.extend(["--mmproj".to_string(), m.to_string()]);
    }
    argv.extend(
        [
            "--host", "127.0.0.1", "--port", &port.to_string(), "--alias", alias, "-ngl", "99", "-c", &ctx.to_string(), "-np", &parallel.to_string(), "--no-kv-unified",
            "-fa", "on", "-ctk", kv, "-ctv", kv, "--cache-reuse", &cache_reuse.to_string(), "--cache-ram", &cache_ram_mib.to_string(), "--fit", "off", "--no-webui",
        ]
        .iter()
        .map(|s| s.to_string()),
    );
    argv
}

pub fn llama_argv_default(bin: &str, model_path: &str, mmproj_path: Option<&str>, port: u16) -> Vec<String> {
    llama_argv(bin, model_path, mmproj_path, port, CTX, PARALLEL, KV_TYPE, ALIAS, CACHE_REUSE, CACHE_RAM_MIB)
}

/// `(argv, env)` for a `Sidecar(ENGINE, key, argv, env, llama_probe(base))`
/// serving the GGUF tree at `tree_dir` (`<link_root>/qwen3-vl-4b-instruct-gguf`
/// after `WeightsResolver.ensure_tree(TREE)`). Pure: the file names are the
/// tree's (`MODEL_FILE`, `MMPROJ_FILE`); `env` is passed through.
pub fn plan_sidecar(bin: &str, tree_dir: &str, port: u16, env: Option<&HashMap<String, String>>, ctx: u64, vision: bool) -> (Vec<String>, HashMap<String, String>) {
    let t = tree_dir.trim_end_matches('/');
    let model = format!("{t}/{MODEL_FILE}");
    let mmproj = if vision { Some(format!("{t}/{MMPROJ_FILE}")) } else { None };
    (llama_argv(bin, &model, mmproj.as_deref(), port, ctx, PARALLEL, KV_TYPE, ALIAS, CACHE_REUSE, CACHE_RAM_MIB), env.cloned().unwrap_or_default())
}

/// What the sidecar was started for (`SidecarManager.ensure`'s `key`).
pub fn sidecar_key(tree_dir: &str, port: u16, ctx: u64, vision: bool) -> String {
    plan_sidecar(LLAMA_BIN, tree_dir, port, None, ctx, vision).0[1..].join(" ")
}

pub fn base_url(port: u16) -> String {
    format!("http://127.0.0.1:{port}")
}

/// A `Sidecar` readiness probe like the launcher's: `/health` 200, then
/// one 1-token completion (the CUDA kernels run, not just the load).
/// Never fatal: llama-server answers 503 on `/health` while loading.
pub fn llama_probe(base: &str, model: &str) -> Probe {
    let base = base.trim_end_matches('/').to_string();
    let model = model.to_string();
    Arc::new(move || {
        let Ok((status, _)) = httpc::get_status(&format!("{base}/health"), 2.0) else { return (false, None) };
        if status != 200 {
            return (false, None);
        }
        let body = json!({"model": model, "messages": [{"role": "user", "content": "ok"}], "max_tokens": 1});
        match httpc::post_json(&format!("{base}/v1/chat/completions"), &body, 30.0) {
            Ok(Some(r)) => (r.get("choices").map(crate::py::truthy).unwrap_or(false), None),
            _ => (false, None),
        }
    })
}

// ---- the client ------------------------------------------------------------------

/// llama-server takes image parts up to 10 MB; the inline media limit is 6 MiB
pub const MAX_IMAGE_DATA_URL: usize = 8 * 1024 * 1024;

/// One llama-server's OpenAI-compatible chat API on `base`
/// (`http://127.0.0.1:18080`). `vision`: the projector is loaded, so an
/// image goes as an `image_url` content part (fv-serve's `OpenAiChat::body`).
pub struct Upsampler {
    pub base: String,
    pub model: String,
    pub vision: bool,
    pub log: Log,
    pub last_error: Option<String>,
    pub calls: u32,
    pub failures: u32,
}

impl Upsampler {
    pub fn new(base: &str) -> Upsampler {
        Upsampler { base: base.into(), model: ALIAS.into(), vision: true, log: crate::null_log(), last_error: None, calls: 0, failures: 0 }
    }

    pub fn url(&self) -> String {
        format!("{}/v1/chat/completions", self.base.trim_end_matches('/'))
    }

    /// The chat request (pure): system prompt per family / mode, the
    /// prompt as the user text, the image as a `data:` part when the
    /// projector is loaded (else the system prompt says an unseen image
    /// sets the first frame), temperature 0.6, the family's token cap.
    pub fn request_body(&self, prompt: &str, family: &str, mode: &str, negative: Option<&str>, image_b64: Option<&str>, image_mime: &str) -> Value {
        let family = if FAMILIES.contains(&family) { family } else { family_of(Some(family)) };
        let has_image = image_b64.map(|s| !s.is_empty()).unwrap_or(false);
        let attach = has_image && self.vision && image_b64.map(|s| s.len() <= MAX_IMAGE_DATA_URL).unwrap_or(false);
        let system = system_prompt(family, mode, has_image, attach, negative);
        let user = if attach {
            json!([{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": format!("data:{image_mime};base64,{}", image_b64.unwrap())}}])
        } else {
            json!(prompt)
        };
        json!({"model": self.model, "temperature": TEMPERATURE, "max_tokens": max_tokens(family),
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
    }

    /// The rewritten prompt, or `prompt` itself when the mode is off, the
    /// prompt is empty, or the call fails (HTTP error, timeout, connection
    /// refused, no content, nothing left after cleaning): `last_error` then
    /// says why, for the job log.
    pub fn expand(&mut self, prompt: &str, family: &str, mode: &str, negative: Option<&str>, image_b64: Option<&str>, image_mime: &str, timeout_s: f64) -> String {
        self.last_error = None;
        if !(mode == "balanced" || mode == "quality") || prompt.trim().is_empty() {
            return prompt.to_string();
        }
        let fam = if FAMILIES.contains(&family) { family } else { family_of(Some(family)) };
        let body = self.request_body(prompt, fam, mode, negative, image_b64, image_mime);
        self.calls += 1;
        match httpc::post_json(&self.url(), &body, timeout_s) {
            Ok(r) => match content_of(r.as_ref()) {
                None => self.fail("no content in the completion".into()),
                Some(answer) => match clean_for(fam, &answer, None) {
                    Some(out) => return out,
                    None => self.fail("nothing left after cleaning the completion".into()),
                },
            },
            Err(httpc::Error::Status(e)) => self.fail(format!("HTTP {}: {}", e.status, e.body.chars().take(200).collect::<String>())),
            Err(httpc::Error::Transport(t)) => {
                let low = t.to_lowercase();
                if low.contains("timed out") || low.contains("timeout") {
                    self.fail(format!("timeout after {timeout_s:.0} s"))
                } else {
                    self.fail(t)
                }
            }
        }
        prompt.to_string()
    }

    fn fail(&mut self, why: String) {
        self.failures += 1;
        (self.log)(&format!("[{ENGINE}] prompt expansion failed ({why}); generating with the original prompt"));
        self.last_error = Some(why);
    }
}

/// The first choice's message text of a chat completion, or None.
pub fn content_of(r: Option<&Value>) -> Option<String> {
    let content = r?.get("choices")?.get(0)?.get("message")?.get("content")?;
    match content {
        Value::String(s) => Some(s.clone()),
        Value::Array(parts) => Some(parts.iter().filter_map(|p| p.as_object()).map(|p| p.get("text").and_then(|t| t.as_str()).unwrap_or("")).collect::<Vec<_>>().join(" ")),
        _ => None,
    }
}

/// A file's bytes as base64 (for `expand(image_b64=…)`).
pub fn image_data_b64(path: &str) -> std::io::Result<String> {
    Ok(b64_encode(&std::fs::read(path)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argv_reproduces_the_launcher_flags() {
        let argv = llama_argv_default("/opt/fastvideo-rs/bin/llama-server", "/w/m.gguf", Some("/w/mmproj.gguf"), 18080);
        assert_eq!(&argv[..3], ["/opt/fastvideo-rs/bin/llama-server", "-m", "/w/m.gguf"]);
        assert_eq!(&argv[3..5], ["--mmproj", "/w/mmproj.gguf"]);
        assert_eq!(argv[5..].join(" "), "--host 127.0.0.1 --port 18080 --alias qwen3-vl-4b-instruct -ngl 99 -c 16384 -np 1 --no-kv-unified -fa on -ctk q8_0 -ctv q8_0 --cache-reuse 256 --cache-ram 4096 --fit off --no-webui");
        assert!(!llama_argv_default("b", "m", None, 1).contains(&"--mmproj".to_string()));
        let a = llama_argv("b", "m", Some("p"), 18080, 65536, 2, KV_TYPE, ALIAS, CACHE_REUSE, CACHE_RAM_MIB);
        let at = |f: &str| a[a.iter().position(|x| x == f).unwrap() + 1].clone();
        assert_eq!((at("-c"), at("-np")), ("65536".into(), "2".into()));
    }

    #[test]
    fn plan_sidecar_is_pure_and_names_the_tree_files() {
        let env: HashMap<String, String> = [("FV_WEIGHTS".to_string(), "/workspace/w".to_string())].into_iter().collect();
        let (argv, e) = plan_sidecar("/opt/fastvideo-rs/bin/llama-server", "/workspace/w/qwen3-vl-4b-instruct-gguf", 18080, Some(&env), CTX, true);
        assert_eq!(argv[2], format!("/workspace/w/qwen3-vl-4b-instruct-gguf/{MODEL_FILE}"));
        assert_eq!(argv[4], format!("/workspace/w/qwen3-vl-4b-instruct-gguf/{MMPROJ_FILE}"));
        assert_eq!(e, env);
        let (argv, e) = plan_sidecar("b", "/t/", 1, None, CTX, false);
        assert!(!argv.contains(&"--mmproj".to_string()) && e.is_empty());
        assert_eq!(TREE, "r2:fv-weights/weights/qwen3-vl-4b-instruct-gguf=qwen3-vl-4b-instruct-gguf");
        let sc = crate::sidecar::Sidecar::new(ENGINE, &sidecar_key("/t", DEFAULT_PORT, CTX, true), argv, e, llama_probe("http://127.0.0.1:1", ALIAS));
        assert!(sc.engine == "upsampler" && !sc.alive());
    }

    #[test]
    fn vram_estimate() {
        assert!((4.7..=4.9).contains(&(kv_bytes(65536, KV_TYPE) as f64 / 2f64.powi(30))));
        assert!((1.15..=1.25).contains(&(kv_bytes(16384, KV_TYPE) as f64 / 2f64.powi(30))));
        assert!((default_vram_gib() - 6.5).abs() <= 0.05);
        assert!(vram_gib(CTX, KV_TYPE, false) < default_vram_gib());
        assert!(vram_gib(65536, KV_TYPE, true) >= vram_gib(16384, KV_TYPE, true) + 3.5);
        assert!((4.7..=4.8).contains(&fetch_gib()));
    }

    #[test]
    fn family_selection_and_system_prompts() {
        assert_eq!((family_of(Some("h3")), family_of(Some("fv-h3")), family_of(Some("fastwan")), family_of(Some("wan")), family_of(Some("ltx")), family_of(Some("something")), family_of(None)), ("h3", "h3", "wan", "wan", "ltx", "other", "other"));
        let wan = system_prompt("wan", "balanced", false, false, None);
        assert!(wan.contains("makes no sound") && wan.contains("60 to 100 words") && wan.contains("1800 characters") && wan.contains("One English paragraph"));
        let ltx = system_prompt("ltx", "quality", true, false, None);
        assert!(ltx.contains("Start directly with the main action") && ltx.contains("140 to 200 words") && ltx.contains("image you cannot see") && ltx.contains("3600 characters"));
        let h3 = system_prompt("h3", "balanced", false, false, None);
        assert!(h3.contains("camera language") && h3.contains("<d>[English]") && !h3.contains("image you cannot see"));
        assert!(h3.contains(H3_DESCRIPTION) && h3.contains(H3_SOUNDSCAPE) && h3.contains(H3_MUSIC));
        assert!(h3.contains("one paragraph") && !h3.contains("[Shot 1]") && h3.contains("60 to 110 words") && h3.contains("3200 characters"));
        assert!(h3.contains("the three labels exactly as shown") && !h3.contains("One English paragraph"));
        let q = system_prompt("h3", "quality", false, false, None);
        assert!(q.contains("[Shot 1]") && q.contains("120 to 220 words"));
        let other = system_prompt("other", "balanced", false, false, None);
        assert!(other.contains("detailed video prompt") && other.contains("80 to 130 words"));
        let seen = system_prompt("wan", "balanced", true, true, None);
        assert!(seen.contains("attached image") && !seen.contains("cannot see"));
        let neg = system_prompt("wan", "balanced", false, false, Some("blurry,  low   quality"));
        assert!(neg.contains("The user excludes: blurry, low quality.") && !wan.contains("excludes"));
    }

    #[test]
    fn clean_strips_wrappers() {
        assert_eq!(clean("  A cat walks.  ", 100).as_deref(), Some("A cat walks."));
        assert_eq!(clean("\"A cat walks.\"", 100).as_deref(), Some("A cat walks."));
        assert_eq!(clean("Prompt: A cat\nwalks.", 100).as_deref(), Some("A cat walks."));
        assert_eq!(clean("Rewritten Prompt: A cat walks.", 100).as_deref(), Some("A cat walks."));
        assert_eq!(clean("```text\nA cat walks.\n```", 100).as_deref(), Some("A cat walks."));
        assert_eq!(clean("```\nA cat walks.\n```", 100).as_deref(), Some("A cat walks."));
        assert_eq!(clean("<think>hmm</think>\n**A cat**  walks.", 100).as_deref(), Some("**A cat** walks."));
        assert_eq!(clean("\"Go,\" she says, \"now.\"", 100).as_deref(), Some("\"Go,\" she says, \"now.\""));
        assert_eq!(clean("   ", 100), None);
        assert_eq!(clean("\"\"", 100), None);
        assert_eq!(clean("A fox\n\n  runs   fast.\n", 100).as_deref(), Some("A fox runs fast."));
    }

    #[test]
    fn cut_prefers_sentences_then_words() {
        let s = "One two three. Four five six. Seven eight nine.";
        assert_eq!(cut(s, 100), s);
        assert_eq!(cut(s, 32), "One two three. Four five six.");
        assert_eq!(cut("Alpha beta gamma delta. Epsilon", 12), "Alpha beta");
        assert_eq!(cut("Supercalifragilistic", 5), "Super");
        assert!(cut(&"word ".repeat(1000), 1800).chars().count() <= 1800);
    }

    #[test]
    fn family_caps() {
        let long = "A fox runs. ".repeat(500);
        for fam in FAMILIES {
            let out = clean_for(fam, &long, None).unwrap();
            assert!(out.chars().count() <= max_chars(fam), "{fam}");
        }
        assert_eq!((max_chars("wan"), max_chars("ltx"), max_chars("h3"), max_chars("other")), (1800, 3600, 3200, 1800));
        let w = clean_for("wan", &long, None).unwrap();
        assert!(w.chars().count() <= 1800 && w.ends_with('.'));
    }

    #[test]
    fn h3_answers_keep_their_sections() {
        let a = "```\nintegrated_multimodal_description: A keeper climbs.\n  Mara says <d>[English] Light it.</d>\noverall_soundscape: Wind.\nnon_diegetic_music: None.\n```";
        assert_eq!(clean_h3(a, 10_000).unwrap(), "integrated_multimodal_description: A keeper climbs.\nMara says <d>[English] Light it.</d>\noverall_soundscape: Wind.\nnon_diegetic_music: None.");
        assert_eq!(clean_h3("A cat walks.", 10_000).unwrap(), "integrated_multimodal_description: A cat walks.");
        let q = "integrated_multimodal_description:\n[Shot 1] A wide shot.\n[Shot 2] A close-up.\noverall_soundscape: Rain.\nnon_diegetic_music: Strings.";
        assert_eq!(clean_h3(q, 10_000).unwrap(), "integrated_multimodal_description: [Shot 1] A wide shot.\n[Shot 2] A close-up.\noverall_soundscape: Rain.\nnon_diegetic_music: Strings.");
        assert_eq!(clean_h3("  ", 10_000), None);
    }

    #[test]
    fn h3_answers_fit_the_limit() {
        let a = "integrated_multimodal_description: A keeper climbs the stairs. Mara says <d>[English] Light the lamp now please.</d> The beam sweeps the sea.\noverall_soundscape: Wind and rain.\nnon_diegetic_music: A low cello drone.";
        let full = clean_h3(a, 10_000).unwrap();
        let n = full.chars().count();
        let p = clean_h3(a, n - 1).unwrap();
        assert!(p.ends_with("non_diegetic_music: None.") && p.contains("The beam sweeps the sea."), "{p}");
        for limit in (121..n - 1).rev() {
            let p = clean_h3(a, limit).unwrap();
            assert!(p.chars().count() <= limit, "{p}");
            assert_eq!(p.matches("<d>").count(), p.matches("</d>").count(), "{p}");
            assert!(p.contains("overall_soundscape: Wind and rain."), "{p}");
            assert!(p.starts_with("integrated_multimodal_description: ") && p.ends_with("non_diegetic_music: None."), "{p}");
        }
    }

    #[test]
    fn mode_of_reads_the_catalog_knob() {
        let c = crate::catalog::tests::catalog();
        let knob = &c.knobs[KNOB];
        assert_eq!((knob.type_.as_str(), knob.default.as_str(), knob.options.as_slice()), ("select", Some("off"), &["off".to_string(), "balanced".into(), "quality".into()][..]));
        assert!(knob.group == "quality" && knob.cli.is_none() && knob.fv_field.is_none());
        let help = c.data["knobs"][KNOB]["help"].as_str().unwrap();
        assert!(help.contains("Qwen3-VL-4B") && help.contains("llama-server"));
        for m in &c.models {
            assert!(m.knobs.contains(&KNOB.to_string()), "{}", m.id);
        }
        let j = c.validate(&json!({"engine": "h3", "model": "fasth3-v2", "prompt": "a fox", "options": {"prompt": "quality"}})).unwrap();
        assert!(j.options[KNOB] == "quality" && j.explicit.contains_key(KNOB) && mode_of(&j) == "quality");
        let j = c.validate(&json!({"engine": "fastvideo", "model": "wan22-ti2v-5b", "prompt": "a fox", "options": {"expand.prompt": "balanced"}})).unwrap();
        assert_eq!(mode_of(&j), "balanced");
        let j = c.validate(&json!({"engine": "fastwan", "model": "fastwan-1.3b", "prompt": "a fox"})).unwrap();
        assert!(mode_of(&j) == "off" && !j.explicit.contains_key(KNOB));
        assert!(c.validate(&json!({"engine": "fastwan", "model": "fastwan-1.3b", "prompt": "a fox", "options": {"prompt": "max"}})).unwrap_err().message.contains("must be one of"));
        // The knob never reaches the engines' flags / fields.
        let s = crate::flags::EngineSettings { backend: "cpu".into(), ..Default::default() };
        let j = c.validate(&json!({"engine": "h3", "model": "fasth3-v2", "prompt": "a fox", "options": {"prompt": "quality"}})).unwrap();
        assert!(!crate::flags::h3_serve_argv(&j, &s, &c.knobs, 1).unwrap().iter().any(|a| a.contains("expand") || a == "quality"));
        let j = c.validate(&json!({"engine": "fastvideo", "model": "wan22-ti2v-5b", "prompt": "a fox", "options": {"prompt": "quality"}})).unwrap();
        let body = crate::flags::fastvideo_body(&j, &c.knobs, &Default::default(), Some(1)).unwrap();
        assert!(!body.contains_key("prompt_expansion") && !body.contains_key("expand") && body["prompt"] == "a fox");
    }

    #[test]
    fn request_body_shape_and_content_of() {
        let u = Upsampler::new("http://127.0.0.1:1");
        let b = u.request_body("a fox", "fastwan", "balanced", None, None, "image/jpeg");
        assert_eq!((b["model"].as_str(), b["temperature"].as_f64(), b["max_tokens"].as_u64()), (Some(ALIAS), Some(0.6), Some(512)));
        assert_eq!(b["messages"][1]["content"], "a fox");
        let png = "A".repeat(32);
        let b = u.request_body("x", "ltx", "balanced", None, Some(&png), "image/png");
        assert_eq!(b["messages"][1]["content"][1]["image_url"]["url"], format!("data:image/png;base64,{png}"));
        let big = "A".repeat(MAX_IMAGE_DATA_URL + 1);
        let b = u.request_body("x", "ltx", "balanced", None, Some(&big), "image/png");
        assert!(b["messages"][1]["content"] == "x" && b["messages"][0]["content"].as_str().unwrap().contains("image you cannot see"));
        assert_eq!(content_of(Some(&json!({"choices": [{"message": {"content": "hi"}}]}))).as_deref(), Some("hi"));
        assert_eq!(content_of(Some(&json!({"choices": [{"message": {"content": [{"type": "text", "text": "a"}, {"text": "b"}]}}]}))).as_deref(), Some("a b"));
        assert_eq!(content_of(Some(&json!({"choices": []}))), None);
        assert_eq!(content_of(None), None);
    }
}
