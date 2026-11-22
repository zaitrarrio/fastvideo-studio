//! Catalog recipe + job → the engines' exact command lines and fv-serve's
//! request body. Every flag here is one of the clap definitions in
//! `h3-rs/crates/h3-cli/src/main.rs` and `fastwan-rs/crates/fastwan-cli/src/main.rs`;
//! the fv-serve fields are `NativeBody` in
//! `fastvideo-rs/crates/fastvideo-serve/src/native.rs` (`deny_unknown_fields`).
//! Pure functions, snapshot-tested against lines from the engines' run logs.

use crate::catalog::{Job, Knob};
use crate::errors::{JobError, JobResult};
use crate::py::{format_g6, number_str, py_str, round_to, shlex_join, truthy};
use indexmap::IndexMap;
use serde_json::{Map, Value};
use std::path::Path;

pub fn h3_task_name(task: &str) -> Option<&'static str> {
    match task {
        "t2v" => Some("t2va"),
        "fl2v" => Some("fl2va"),
        "ref2v" => Some("ref2va"),
        "extend" => Some("ref2va"),
        _ => None,
    }
}

/// Flags whose value is the next token (the rest take none).
pub const H3_VALUE_FLAGS: &[&str] = &[
    "--variant",
    "--task",
    "--model-dir",
    "--variant-dir",
    "--transformer",
    "--adapter",
    "--adapter-strength",
    "--tae",
    "--tables",
    "--steps",
    "--attention",
    "--vsa-sparsity",
    "--dense-first-steps",
    "--weights",
    "--decoder-weights",
    "--ffn-chunk",
    "--backend",
    "--addr",
    "--max-pixels",
    "--max-frames",
    "--tile",
    "--upscale",
    "--upscale-factor",
    "--width",
    "--height",
    "--frames",
    "--seed",
    "--prompt",
    "--out",
    "--first-image",
    "--last-image",
    "--ref",
    "--continue",
    "--aspect",
    "--tier",
    "--regen-dir",
    "--draft",
    "--adaln-table",
    "--start-sigma",
    "--embeds",
    "--gallery",
    "--allow-origin",
    "--runs",
];
pub const H3_BOOL_FLAGS: &[&str] = &[
    "--vsa-qk8",
    "--qk8",
    "--pv8",
    "--stream-blocks",
    "--preview",
    "--synthetic",
    "--ops-only",
];
pub const H3_ALIAS: &[(&str, &str)] = &[("--qk8", "--vsa-qk8")];
pub const FW_VALUE_FLAGS: &[&str] = &[
    "--attention",
    "--vsa-sparsity",
    "--weights",
    "--backend",
    "--addr",
    "--model-dir",
    "--tae",
    "--transformer",
    "--gallery",
    "--max-pixels",
    "--max-frames",
    "--tae-chunk",
    "--allow-origin",
    "--upscaler",
    "--embeds",
    "--prompt",
    "--text-encoder",
    "--tokenizer",
    "--runs",
    "--seed",
    "--out",
    "--ffn-chunk",
    "--upscale",
    "--upscale-x",
    "--cache-frames",
    "--sink-frames",
    "--width",
    "--height",
    "--frames",
];
pub const FW_BOOL_FLAGS: &[&str] = &["--causal", "--synthetic"];

/// The order flags are emitted in (readable, stable for the snapshot tests).
pub const H3_ORDER: &[&str] = &[
    "--variant",
    "--task",
    "--model-dir",
    "--variant-dir",
    "--transformer",
    "--adapter",
    "--adapter-strength",
    "--tables",
    "--tae",
    "--steps",
    "--attention",
    "--vsa-sparsity",
    "--dense-first-steps",
    "--vsa-qk8",
    "--pv8",
    "--weights",
    "--decoder-weights",
    "--stream-blocks",
    "--preview",
];
pub const FW_ORDER: &[&str] = &[
    "--model-dir",
    "--transformer",
    "--tae",
    "--text-encoder",
    "--tokenizer",
    "--weights",
    "--attention",
    "--vsa-sparsity",
    "--tae-chunk",
    "--causal",
    "--cache-frames",
    "--sink-frames",
];

/// fv-serve's `NativeBody` fields (native.rs): a knob whose `field` is not here cannot be sent.
pub const FV_FIELDS: &[&str] = &[
    "model",
    "prompt",
    "negative_prompt",
    "seed",
    "size",
    "width",
    "height",
    "aspect_ratio",
    "short_edge",
    "seconds",
    "num_frames",
    "fps",
    "steps",
    "guidance",
    "image_url",
    "last_image_url",
    "reference_urls",
    "reference_strength",
    "reference_lora_strength",
    "audio_url",
    "video_url",
    "start_s",
    "end_s",
    "retake_mode",
    "extend_s",
    "extend_at",
    "context_s",
    "prompt_expansion",
];
/// h3 `--continue`: 5 / 22 / 39 frames at the canvas
pub const CONTINUE_FRAMES: i64 = 39;

/// Worker-side paths the command lines need.
#[derive(Debug, Clone)]
pub struct EngineSettings {
    pub h3_bin: String,
    pub fastwan_bin: String,
    pub backend: String,
    pub synthetic: bool,
    /// realesr-animevideov3.safetensors
    pub esrgan: Option<String>,
    /// taeh3.safetensors for h3 --preview
    pub taeh3: Option<String>,
    pub h3_ref_transformer: String,
}

impl Default for EngineSettings {
    fn default() -> Self {
        EngineSettings {
            h3_bin: "h3".into(),
            fastwan_bin: "fastwan".into(),
            backend: "cuda".into(),
            synthetic: false,
            esrgan: None,
            taeh3: None,
            h3_ref_transformer: "/workspace/w/h3-ref2va/transformer_ref".into(),
        }
    }
}

/// A flag's value in a flag map.
#[derive(Debug, Clone, PartialEq)]
pub enum FlagVal {
    None,
    Bool(bool),
    Str(String),
    Int(i64),
    Float(f64),
    List(Vec<String>),
}

impl FlagVal {
    fn from_value(v: &Value) -> FlagVal {
        match v {
            Value::Null => FlagVal::None,
            Value::Bool(b) => FlagVal::Bool(*b),
            Value::Number(n) => {
                if n.is_i64() || n.is_u64() {
                    FlagVal::Int(n.as_i64().unwrap_or(0))
                } else {
                    FlagVal::Float(n.as_f64().unwrap_or(0.0))
                }
            }
            Value::String(s) => FlagVal::Str(s.clone()),
            Value::Array(a) => FlagVal::List(a.iter().map(py_str).collect()),
            other => FlagVal::Str(py_str(other)),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            FlagVal::Str(s) => Some(s),
            _ => None,
        }
    }

    fn is_none_or_false(&self) -> bool {
        matches!(self, FlagVal::None | FlagVal::Bool(false))
    }
}

/// `{flag: value}` in insertion order (a Python dict).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FlagMap(pub Vec<(String, FlagVal)>);

impl FlagMap {
    pub fn get(&self, f: &str) -> Option<&FlagVal> {
        self.0.iter().find(|(k, _)| k == f).map(|(_, v)| v)
    }

    pub fn contains(&self, f: &str) -> bool {
        self.get(f).is_some()
    }

    /// `d[f] = v` (an existing key keeps its position).
    pub fn set(&mut self, f: &str, v: FlagVal) {
        if let Some(slot) = self.0.iter_mut().find(|(k, _)| k == f) {
            slot.1 = v;
        } else {
            self.0.push((f.to_string(), v));
        }
    }

    pub fn set_str(&mut self, f: &str, v: &str) {
        self.set(f, FlagVal::Str(v.to_string()));
    }

    pub fn remove(&mut self, f: &str) -> Option<FlagVal> {
        let i = self.0.iter().position(|(k, _)| k == f)?;
        Some(self.0.remove(i).1)
    }

    /// `{**self, **other}`.
    pub fn update(&mut self, other: &FlagMap) {
        for (k, v) in &other.0 {
            self.set(k, v.clone());
        }
    }

    fn put(&mut self, f: &str, v: &str) {
        match self.get(f).cloned() {
            Some(FlagVal::List(mut l)) => {
                l.push(v.to_string());
                self.set(f, FlagVal::List(l));
            }
            Some(FlagVal::Str(s)) => self.set(f, FlagVal::List(vec![s, v.to_string()])),
            Some(other) => self.set(f, FlagVal::List(vec![flag_text(&other), v.to_string()])),
            None => self.set(f, FlagVal::Str(v.to_string())),
        }
    }
}

fn flag_text(v: &FlagVal) -> String {
    match v {
        FlagVal::None => "None".into(),
        FlagVal::Bool(b) => {
            if *b {
                "True".into()
            } else {
                "False".into()
            }
        }
        FlagVal::Str(s) => s.clone(),
        FlagVal::Int(i) => i.to_string(),
        FlagVal::Float(f) => format_g6(*f),
        FlagVal::List(l) => format!("{l:?}"),
    }
}

/// `["--weights", "mxfp8", "--pv8"]` → `{"--weights": "mxfp8", "--pv8": True}`
/// (repeated value flags collect into a list).
pub fn parse_flags(
    tokens: &[String],
    value_flags: &[&str],
    bool_flags: &[&str],
    alias: &[(&str, &str)],
) -> Result<FlagMap, String> {
    let resolve = |t: &str| -> String {
        alias
            .iter()
            .find(|(a, _)| *a == t)
            .map(|(_, b)| b.to_string())
            .unwrap_or_else(|| t.to_string())
    };
    let mut out = FlagMap::default();
    let mut i = 0;
    while i < tokens.len() {
        let t = tokens[i].as_str();
        if !t.starts_with("--") {
            return Err(format!(
                "unexpected token {} in flags",
                crate::py::py_str_repr(t)
            ));
        }
        let f = resolve(t);
        if let Some((k, v)) = t.split_once('=') {
            out.put(&resolve(k), v);
            i += 1;
        } else if bool_flags.contains(&f.as_str()) {
            out.set(&f, FlagVal::Bool(true));
            i += 1;
        } else if value_flags.contains(&f.as_str()) {
            if i + 1 >= tokens.len() {
                return Err(format!("flag {t} needs a value"));
            }
            out.put(&f, &tokens[i + 1]);
            i += 2;
        } else {
            return Err(format!("unknown flag {t}"));
        }
    }
    Ok(out)
}

pub fn emit(flags: &FlagMap, order: &[&str]) -> Vec<String> {
    let mut argv = Vec::new();
    let mut names: Vec<String> = order.iter().map(|s| s.to_string()).collect();
    for (k, _) in &flags.0 {
        if !order.contains(&k.as_str()) {
            names.push(k.clone());
        }
    }
    let mut seen: Vec<String> = Vec::new();
    for f in names {
        if seen.contains(&f) {
            continue;
        }
        let Some(v) = flags.get(&f) else { continue };
        seen.push(f.clone());
        match v {
            FlagVal::None | FlagVal::Bool(false) => {}
            FlagVal::Bool(true) => argv.push(f.clone()),
            FlagVal::List(l) => {
                for x in l {
                    argv.push(f.clone());
                    argv.push(x.clone());
                }
            }
            other => {
                argv.push(f.clone());
                argv.push(flag_text(other));
            }
        }
    }
    argv
}

/// Flags from the knobs the job set explicitly plus the model's
/// `knob_defaults`; the recipe's own flags cover the rest.
pub fn knob_overlay(
    job: &Job,
    knobs: &IndexMap<String, Knob>,
    cli_only: Option<&[&str]>,
) -> FlagMap {
    let mut out = FlagMap::default();
    let auto = Value::String("auto".into());
    for kid in &job.model.knobs {
        let knob = &knobs[kid];
        let Some(cli) = knob.cli.as_deref() else {
            continue;
        };
        if let Some(only) = cli_only {
            if !only.contains(&cli) {
                continue;
            }
        }
        if !job.explicit.contains_key(kid) && !job.model.knob_defaults.contains_key(kid) {
            continue;
        }
        // A conditional knob (`when`) is emitted only when its condition holds.
        let blocked = knob.when.iter().any(|(dep, want)| {
            let have = job.opt(dep).cloned().unwrap_or(Value::Null);
            !crate::py::py_eq(&have, want) && have != auto
        });
        if blocked {
            continue;
        }
        let v = job.options.get(kid).cloned().unwrap_or(Value::Null);
        match knob.type_.as_str() {
            "bool" => out.set(cli, FlagVal::Bool(truthy(&v))),
            "select" => {
                if cli == "--attention" && v == auto {
                    out.set(cli, FlagVal::None);
                } else {
                    out.set(cli, FlagVal::from_value(&v));
                }
            }
            _ => out.set(cli, FlagVal::from_value(&v)),
        }
    }
    out
}

// ---- h3 ------------------------------------------------------------------------

pub fn h3_task(job: &Job) -> JobResult<&'static str> {
    h3_task_name(&job.task)
        .ok_or_else(|| JobError::validate(format!("h3 has no task {}", job.task)))
}

fn recipe_str<'a>(r: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    r.get(key)
        .filter(|v| truthy(v))
        .map(|v| v.as_str().unwrap_or(""))
}

pub fn h3_model_flags(
    job: &Job,
    s: &EngineSettings,
    _knobs: &IndexMap<String, Knob>,
    for_serve: bool,
) -> JobResult<FlagMap> {
    let r = &job.model.recipe;
    let mut flags = FlagMap::default();
    flags.set_str(
        "--variant",
        r.get("variant")
            .and_then(|v| v.as_str())
            .unwrap_or(&job.model.id),
    );
    for (key, flag) in [
        ("model_dir", "--model-dir"),
        ("variant_dir", "--variant-dir"),
        ("transformer", "--transformer"),
        ("adapter", "--adapter"),
    ] {
        if let Some(v) = recipe_str(r, key) {
            flags.set_str(flag, v);
        }
    }
    let is_ref = job.task == "ref2v";
    // Ref2VA on a `transformer` variant (base, plug4, pika) and the Turbo Ref2V
    // LoRAs run over h3-ref2va/transformer_ref, which is under neither
    // --model-dir (h3-base) nor --variant-dir: name it (the 2026-11-15 and
    // 2026-11-17 runs do). Omni's tree carries its own transformer_ref.
    let other_dit = is_ref
        && recipe_str(r, "transformer").is_none()
        && job.model.trees.iter().any(|t| t == "weights/h3-ref2va");
    if other_dit {
        flags.set_str("--transformer", &s.h3_ref_transformer);
    }
    let steps: Option<Value> = if job.model.knobs.iter().any(|k| k == "h3.steps") {
        job.opt("steps").cloned()
    } else {
        None
    };
    let steps = steps.filter(|v| !v.is_null());
    let steps_explicit =
        job.is_explicit("steps") || job.model.knob_defaults.contains_key("h3.steps");
    let steps_int = steps.as_ref().and_then(|v| v.as_f64()).map(|f| f as i64);
    if steps_explicit {
        if let Some(n) = steps_int {
            flags.set("--steps", FlagVal::Int(n));
        }
    }
    // AdaLN tables record their DiT: on the other DiT only a `tables_ref` fits.
    let tables = if other_dit {
        recipe_str(r, "tables_ref")
    } else {
        recipe_str(r, "tables")
    };
    let tables_on = job.opt("tables").map(truthy).unwrap_or(true);
    if let Some(tables) = tables {
        if tables_on && job.model.knobs.iter().any(|k| k == "h3.tables") {
            let ts = r
                .get("tables_steps")
                .and_then(|v| v.as_f64())
                .map(|f| f as i64);
            if ts.is_none() || !steps_explicit || steps_int == ts {
                flags.set_str("--tables", tables);
            }
        }
    }
    if job.opt_truthy("preview") {
        let Some(tae) = &s.taeh3 else {
            return Err(JobError::validate(
                "preview decoding needs the TAEH3 weights (STUDIO_TAEH3)",
            ));
        };
        flags.set_str("--tae", tae);
        flags.set("--preview", FlagVal::Bool(true));
    }
    if job.is_explicit("adapter_strength") {
        let v = job
            .opt("adapter_strength")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        flags.set("--adapter-strength", FlagVal::Float(v));
    }
    // serve: an engine started normally serves t2va and fl2va from the same
    // DiT; Ref2VA needs transformer_ref, which only `--task ref2va` loads.
    if !for_serve || h3_task(job)? == "ref2va" {
        flags.set_str("--task", h3_task(job)?);
    }
    Ok(flags)
}

fn recipe_flags(job: &Job) -> Vec<String> {
    job.model
        .recipe
        .get("flags")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().map(py_str).collect())
        .unwrap_or_default()
}

fn overlay_into(flags: &mut FlagMap, overlay: &FlagMap) {
    for (f, v) in &overlay.0 {
        if v.is_none_or_false() {
            flags.remove(f);
        } else {
            flags.set(f, v.clone());
        }
    }
}

pub fn h3_attn_flags(job: &Job, knobs: &IndexMap<String, Knob>) -> JobResult<FlagMap> {
    let recipe = parse_flags(&recipe_flags(job), H3_VALUE_FLAGS, H3_BOOL_FLAGS, H3_ALIAS)
        .map_err(|e| JobError::validate(format!("recipe flags: {e}")))?;
    let mut overlay = knob_overlay(job, knobs, None);
    overlay.remove("--preview"); // handled with --tae in h3_model_flags
    let mut flags = recipe;
    overlay_into(&mut flags, &overlay);
    if flags.get("--attention").and_then(FlagVal::as_str) == Some("auto") {
        flags.remove("--attention");
    }
    if flags.get("--attention").and_then(FlagVal::as_str) == Some("dense") {
        flags.remove("--vsa-sparsity");
        flags.remove("--dense-first-steps");
    }
    Ok(flags)
}

pub fn h3_serve_argv(
    job: &Job,
    s: &EngineSettings,
    knobs: &IndexMap<String, Knob>,
    port: u16,
) -> JobResult<Vec<String>> {
    let mut argv: Vec<String> = [
        &s.h3_bin,
        "serve",
        "--backend",
        &s.backend,
        "--addr",
        &format!("127.0.0.1:{port}"),
    ]
    .iter()
    .map(|x| x.to_string())
    .collect();
    if s.synthetic {
        argv.push("--synthetic".into());
    } else {
        let mut flags = h3_model_flags(job, s, knobs, true)?;
        flags.update(&h3_attn_flags(job, knobs)?);
        argv.extend(emit(&flags, H3_ORDER));
    }
    argv.extend([
        "--max-pixels".to_string(),
        h3_max_pixels(job).to_string(),
        "--max-frames".to_string(),
        job.model.frames.max.to_string(),
    ]);
    if job.upscale != 0 && job.engine == "h3" {
        if s.esrgan.is_none() && !s.synthetic {
            return Err(JobError::validate(
                "upscaling needs the Real-ESRGAN weights (STUDIO_ESRGAN)",
            ));
        }
        argv.push("--upscale".into());
        argv.push(s.esrgan.clone().unwrap_or_else(|| "synthetic".into()));
    }
    Ok(argv)
}

/// What a resident `h3 serve` was started for: everything but the port.
pub fn h3_serve_key(
    job: &Job,
    s: &EngineSettings,
    knobs: &IndexMap<String, Knob>,
) -> JobResult<String> {
    Ok(h3_serve_argv(job, s, knobs, 0)?.join(" "))
}

pub fn h3_max_pixels(job: &Job) -> i64 {
    let up32 = |v: i64| (v + 31) / 32 * 32; // finish::canvas_edge
    job.model
        .sizes
        .iter()
        .map(|(w, h)| up32(*w) * up32(*h))
        .max()
        .unwrap_or(0)
}

/// `h3 generate`'s media: `first_image` / `last_image`, `references` as
/// `(kind, path, audio)`, `continue` as `(y4m, wav)`, `draft`.
#[derive(Debug, Clone, Default)]
pub struct H3Media {
    pub first_image: Option<String>,
    pub last_image: Option<String>,
    pub references: Vec<(String, String, Option<String>)>,
    pub continue_: Option<(String, Option<String>)>,
    pub draft: Option<String>,
}

pub fn h3_generate_argv(
    job: &Job,
    s: &EngineSettings,
    knobs: &IndexMap<String, Knob>,
    media: &H3Media,
    out: &Path,
    seed: i64,
) -> JobResult<Vec<String>> {
    let mut flags = h3_model_flags(job, s, knobs, false)?;
    flags.update(&h3_attn_flags(job, knobs)?);
    let mut argv: Vec<String> = vec![
        s.h3_bin.clone(),
        "generate".into(),
        "--backend".into(),
        s.backend.clone(),
    ];
    argv.extend(emit(&flags, H3_ORDER));
    argv.extend([
        "--width".into(),
        job.width.to_string(),
        "--height".into(),
        job.height.to_string(),
        "--frames".into(),
        job.frames.to_string(),
        "--seed".into(),
        seed.to_string(),
    ]);
    if let Some(p) = &media.first_image {
        argv.extend(["--first-image".into(), p.clone()]);
    }
    if let Some(p) = &media.last_image {
        argv.extend(["--last-image".into(), p.clone()]);
    }
    for (kind, path, audio) in &media.references {
        let mut spec = format!("{kind}:{path}");
        if kind == "video" {
            if let Some(a) = audio {
                spec.push_str(&format!(",audio={a}"));
            }
        }
        argv.extend(["--ref".into(), spec]);
    }
    if let Some((y4m, wav)) = &media.continue_ {
        argv.push("--continue".into());
        argv.push(match wav {
            Some(w) => format!("{y4m},audio={w}"),
            None => y4m.clone(),
        });
    }
    if job.upscale != 0 {
        argv.extend([
            "--upscale".into(),
            esrgan(s)?,
            "--upscale-factor".into(),
            job.upscale.to_string(),
        ]);
    }
    argv.extend([
        "--prompt".into(),
        job.prompt.clone(),
        "--out".into(),
        out.to_string_lossy().into_owned(),
    ]);
    Ok(argv)
}

pub fn h3_regen2k_argv(
    job: &Job,
    s: &EngineSettings,
    knobs: &IndexMap<String, Knob>,
    draft: &Path,
    out: &Path,
    seed: i64,
) -> JobResult<Vec<String>> {
    let r = &job.model.recipe;
    for key in ["regen_dir", "model_dir", "adaln_table"] {
        if recipe_str(r, key).is_none() {
            return Err(JobError::validate(format!("regen2k recipe lacks {key}")));
        }
    }
    let mut argv: Vec<String> = vec![
        s.h3_bin.clone(),
        "regen2k".into(),
        "--backend".into(),
        s.backend.clone(),
        "--regen-dir".into(),
        recipe_str(r, "regen_dir").unwrap().into(),
        "--model-dir".into(),
        recipe_str(r, "model_dir").unwrap().into(),
        "--draft".into(),
        draft.to_string_lossy().into_owned(),
        "--adaln-table".into(),
        recipe_str(r, "adaln_table").unwrap().into(),
    ];
    argv.extend(emit(&h3_attn_flags(job, knobs)?, H3_ORDER));
    if let Some(steps) = job.model.steps.filter(|n| *n != 0) {
        argv.extend(["--steps".into(), steps.to_string()]);
    }
    argv.extend([
        "--seed".into(),
        seed.to_string(),
        "--prompt".into(),
        job.prompt.clone(),
        "--out".into(),
        out.to_string_lossy().into_owned(),
    ]);
    Ok(argv)
}

// ---- fastwan ------------------------------------------------------------------

pub fn fastwan_flags(
    job: &Job,
    knobs: &IndexMap<String, Knob>,
    cli_only: Option<&[&str]>,
) -> JobResult<FlagMap> {
    let recipe = parse_flags(&recipe_flags(job), FW_VALUE_FLAGS, FW_BOOL_FLAGS, &[])
        .map_err(|e| JobError::validate(format!("recipe flags: {e}")))?;
    let overlay = knob_overlay(job, knobs, cli_only);
    let mut flags = recipe;
    overlay_into(&mut flags, &overlay);
    if flags.get("--attention").and_then(FlagVal::as_str) == Some("auto") {
        flags.remove("--attention");
    }
    if flags.get("--attention").and_then(FlagVal::as_str) == Some("dense") {
        flags.remove("--vsa-sparsity");
    }
    Ok(flags)
}

pub const FW_SERVE_FLAGS: &[&str] = &["--weights", "--attention", "--vsa-sparsity", "--tae-chunk"];

pub fn fastwan_serve_argv(
    job: &Job,
    s: &EngineSettings,
    knobs: &IndexMap<String, Knob>,
    port: u16,
) -> JobResult<Vec<String>> {
    let r = &job.model.recipe;
    let mut argv: Vec<String> = [
        &s.fastwan_bin,
        "serve",
        "--backend",
        &s.backend,
        "--addr",
        &format!("127.0.0.1:{port}"),
    ]
    .iter()
    .map(|x| x.to_string())
    .collect();
    if s.synthetic {
        argv.push("--synthetic".into());
    } else {
        let Some(model_dir) = recipe_str(r, "model_dir") else {
            return Err(JobError::validate(format!(
                "{}: recipe has no model_dir for fastwan serve",
                job.model.id
            )));
        };
        let mut flags = FlagMap::default();
        flags.set_str("--model-dir", model_dir);
        if let Some(t) = recipe_str(r, "transformer") {
            flags.set_str("--transformer", t);
        }
        if let Some(t) = recipe_str(r, "tae") {
            flags.set_str("--tae", t);
        }
        flags.update(&fastwan_flags(job, knobs, Some(FW_SERVE_FLAGS))?);
        argv.extend(emit(&flags, FW_ORDER));
    }
    let max_px = job
        .model
        .sizes
        .iter()
        .map(|(w, h)| w * h)
        .max()
        .unwrap_or(0);
    argv.extend([
        "--max-pixels".into(),
        max_px.to_string(),
        "--max-frames".into(),
        job.model.frames.max.to_string(),
    ]);
    if job.upscale != 0 {
        if s.esrgan.is_none() && !s.synthetic {
            return Err(JobError::validate(
                "upscaling needs the Real-ESRGAN weights (STUDIO_ESRGAN)",
            ));
        }
        argv.push("--upscaler".into());
        argv.push(s.esrgan.clone().unwrap_or_else(|| "synthetic".into()));
    }
    Ok(argv)
}

pub fn fastwan_serve_key(
    job: &Job,
    s: &EngineSettings,
    knobs: &IndexMap<String, Knob>,
) -> JobResult<String> {
    Ok(fastwan_serve_argv(job, s, knobs, 0)?.join(" "))
}

pub fn fastwan_generate_argv(
    job: &Job,
    s: &EngineSettings,
    knobs: &IndexMap<String, Knob>,
    out: &Path,
    seed: i64,
) -> JobResult<Vec<String>> {
    let mut r = job.model.recipe.clone();
    // A serve recipe names the FastWan download (transformer/, text_encoder/,
    // tokenizer/): `generate` wants each part spelled out.
    if let Some(md) = recipe_str(&job.model.recipe, "model_dir").map(str::to_string) {
        for (k, v) in [
            ("transformer", format!("{md}/transformer")),
            ("text_encoder", format!("{md}/text_encoder")),
            ("tokenizer", format!("{md}/tokenizer/tokenizer.json")),
            ("tae", format!("{md}/taew2_1.safetensors")),
        ] {
            if !r.contains_key(k) {
                r.insert(k.into(), Value::String(v));
            }
        }
    }
    for key in ["transformer", "tae", "text_encoder", "tokenizer"] {
        if recipe_str(&r, key).is_none() {
            return Err(JobError::validate(format!(
                "{}: recipe lacks {key} for fastwan generate",
                job.model.id
            )));
        }
    }
    let mut flags = FlagMap::default();
    flags.set_str("--transformer", recipe_str(&r, "transformer").unwrap());
    flags.set_str("--tae", recipe_str(&r, "tae").unwrap());
    flags.set_str("--text-encoder", recipe_str(&r, "text_encoder").unwrap());
    flags.set_str("--tokenizer", recipe_str(&r, "tokenizer").unwrap());
    flags.update(&fastwan_flags(job, knobs, None)?);
    let mut argv: Vec<String> = vec![
        s.fastwan_bin.clone(),
        "generate".into(),
        "--backend".into(),
        s.backend.clone(),
    ];
    argv.extend(emit(&flags, FW_ORDER));
    argv.extend([
        "--width".into(),
        job.width.to_string(),
        "--height".into(),
        job.height.to_string(),
        "--frames".into(),
        job.frames.to_string(),
        "--seed".into(),
        seed.to_string(),
    ]);
    if job.upscale != 0 {
        argv.extend([
            "--upscale".into(),
            esrgan(s)?,
            "--upscale-x".into(),
            job.upscale.to_string(),
        ]);
    }
    argv.extend([
        "--prompt".into(),
        job.prompt.clone(),
        "--out".into(),
        out.to_string_lossy().into_owned(),
    ]);
    Ok(argv)
}

// ---- fastvideo -------------------------------------------------------------------

/// `media_urls`: `first_image` / `last_image` / `audio` / `video` and
/// `references`, as https or data URIs.
#[derive(Debug, Clone, Default)]
pub struct MediaUrls {
    pub first_image: Option<String>,
    pub last_image: Option<String>,
    pub audio: Option<String>,
    pub video: Option<String>,
    pub references: Vec<String>,
}

/// `POST /fv/v1/jobs` body.
pub fn fastvideo_body(
    job: &Job,
    knobs: &IndexMap<String, Knob>,
    media_urls: &MediaUrls,
    seed: Option<i64>,
) -> JobResult<Map<String, Value>> {
    let mut body = Map::new();
    body.insert("model".into(), Value::String(job.model.id.clone()));
    body.insert("prompt".into(), Value::String(job.prompt.clone()));
    if let Some(n) = job.negative_prompt.as_ref().filter(|n| !n.is_empty()) {
        body.insert("negative_prompt".into(), Value::String(n.clone()));
    }
    if let Some(seed) = seed {
        body.insert("seed".into(), Value::from(seed));
    }
    let edit = job.task == "extend" || job.task == "retake";
    if !edit {
        body.insert(
            "size".into(),
            Value::String(format!("{}x{}", job.width, job.height)),
        );
        body.insert("num_frames".into(), Value::from(job.frames));
    }
    for kid in &job.model.knobs {
        let knob = &knobs[kid];
        let Some(field) = knob.fv_field.as_deref() else {
            continue;
        };
        let Some(v) = job.options.get(kid).filter(|v| !v.is_null()).cloned() else {
            continue;
        };
        if !job.explicit.contains_key(kid)
            && !job.model.knob_defaults.contains_key(kid)
            && knob.default.is_null()
        {
            continue;
        }
        if !FV_FIELDS.contains(&field) {
            if job.explicit.contains_key(kid) {
                return Err(JobError::validate(format!(
                    "option {}: fv-serve's jobs API has no `{field}` field",
                    knob.short()
                )));
            }
            continue;
        }
        if field == "prompt_expansion"
            && v == Value::String("disabled".into())
            && !job.explicit.contains_key(kid)
        {
            continue;
        }
        let v = if field == "fps" {
            if edit {
                continue;
            }
            match &v {
                Value::String(s) => Value::from(s.trim().parse::<i64>().map_err(|_| {
                    JobError::validate(format!("option {}: expects an integer", knob.short()))
                })?),
                Value::Number(n) => Value::from(n.as_f64().unwrap_or(0.0) as i64),
                other => other.clone(),
            }
        } else {
            v
        };
        body.insert(field.into(), v);
    }
    if (job.task == "i2v" || job.task == "fl2v") && media_urls.first_image.is_some() {
        body.insert(
            "image_url".into(),
            Value::String(media_urls.first_image.clone().unwrap()),
        );
    }
    if job.task == "fl2v" && media_urls.last_image.is_some() {
        body.insert(
            "last_image_url".into(),
            Value::String(media_urls.last_image.clone().unwrap()),
        );
    }
    if job.task == "ref2v" {
        body.insert(
            "reference_urls".into(),
            Value::Array(
                media_urls
                    .references
                    .iter()
                    .map(|u| Value::String(u.clone()))
                    .collect(),
            ),
        );
    }
    if job.task == "a2v" {
        let audio = media_urls
            .audio
            .clone()
            .ok_or_else(|| JobError::validate("a2v needs input 'audio'"))?;
        body.insert("audio_url".into(), Value::String(audio));
        if let Some(f) = &media_urls.first_image {
            body.insert("image_url".into(), Value::String(f.clone()));
        }
    }
    if edit {
        let video = media_urls
            .video
            .clone()
            .ok_or_else(|| JobError::validate(format!("{} needs input 'video'", job.task)))?;
        body.insert("video_url".into(), Value::String(video));
        let seconds = job.frames as f64 / job.fps_for() as f64;
        if job.task == "extend" {
            body.insert("extend_s".into(), Value::from(round_to(seconds, 3)));
        } else {
            body.insert("start_s".into(), Value::from(0.0));
            body.insert("end_s".into(), Value::from(round_to(seconds, 3)));
        }
    }
    debug_assert!(body.keys().all(|k| FV_FIELDS.contains(&k.as_str())));
    Ok(body)
}

/// The Real-ESRGAN weights, or (synthetic engines) the file's name for the command line.
fn esrgan(s: &EngineSettings) -> JobResult<String> {
    if let Some(e) = &s.esrgan {
        return Ok(e.clone());
    }
    if s.synthetic {
        return Ok("realesr-animevideov3.safetensors".into());
    }
    Err(JobError::validate(
        "upscaling needs the Real-ESRGAN weights (STUDIO_ESRGAN)",
    ))
}

/// The line to reproduce a run on a rented box: the binary by name.
pub fn command_line(argv: &[String]) -> String {
    let mut v: Vec<String> = argv.to_vec();
    if let Some(first) = v.first_mut() {
        *first = first.rsplit('/').next().unwrap_or(first).to_string();
    }
    shlex_join(&v)
}

/// A number's text as Python prints a knob value (`str(50)`, `str(0.5)`).
pub fn value_text(v: &Value) -> String {
    match v {
        Value::Number(n) => number_str(n),
        other => py_str(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::tests::catalog;
    use crate::catalog::Catalog;
    use serde_json::json;

    fn settings() -> EngineSettings {
        EngineSettings {
            h3_bin: "h3".into(),
            fastwan_bin: "fastwan".into(),
            esrgan: Some("/w/realesr-animevideov3.safetensors".into()),
            taeh3: Some("/w/taeh3.safetensors".into()),
            ..Default::default()
        }
    }

    const IGNORE: &[&str] = &[
        "--backend",
        "--seed",
        "--prompt",
        "--out",
        "--width",
        "--height",
        "--frames",
        "--task",
    ];
    // A flag at its clap default says nothing: the logs leave it out, an explicit knob spells it.
    const CLAP_DEFAULTS: &[(&str, &str)] = &[
        ("--weights", "native"),
        ("--decoder-weights", "native"),
        ("--dense-first-steps", "0"),
        ("--adapter-strength", "1"),
        ("--attention", "auto"),
        ("--steps", "3"),
    ];

    // docs/runs/2026-11-14-runpod-6000s-probe-bug, 2026-11-15-runpod-6000s-longlive-plug,
    // 2026-11-12-runpod-pro6000-v2-omni-trim, 2026-11-12-runpod-pro6000-variants.
    fn log_line(name: &str) -> &'static str {
        match name {
            "v2-best" => "/w/h3-bin generate --backend cuda --seed 1 --width 1344 --height 768 --frames 124 --prompt P --out /root/out/v2-best.y4m --variant fasth3-v2 --model-dir /w/h3-8step --transformer /w/h3-8step-mxfp8/transformer.safetensors --weights mxfp8 --decoder-weights mxfp8 --vsa-qk8 --pv8 --tables /root/v2.tables",
            "v2-budget" => "/w/h3-bin generate --backend cuda --seed 1 --width 1344 --height 768 --frames 124 --prompt P --out /root/out/v2-budget.y4m --variant fasth3-v2 --model-dir /w/h3-8step --transformer /w/h3-8step-mxfp8/transformer.safetensors --weights mxfp8 --decoder-weights mxfp8 --vsa-qk8 --pv8 --tables /root/v2.tables --vsa-sparsity 0.9 --dense-first-steps 2",
            "v2-stream" => "/w/h3-bin generate --backend cuda --seed 1 --width 1344 --height 768 --frames 124 --prompt P --out /root/out/v2-best-stream.y4m --variant fasth3-v2 --model-dir /w/h3-8step --transformer /w/h3-8step-mxfp8/transformer.safetensors --weights mxfp8 --decoder-weights mxfp8 --vsa-qk8 --pv8 --tables /root/v2.tables --stream-blocks",
            "plug-fast" => "/w/h3-bin generate --backend cuda --seed 7 --width 1344 --height 768 --frames 124 --prompt P --out /root/out/plug-fast.y4m --variant h3-plug4 --model-dir /w/h3-base --variant-dir /workspace/h3-rs/plug4 --tables /workspace/h3-rs/plug4/plug4.tables --weights mxfp8 --decoder-weights mxfp8 --pv8",
            "plug-fl2va" => "/w/h3-bin generate --backend cuda --seed 7 --width 1344 --height 768 --frames 124 --prompt P --out /root/out/plug-fl2va.y4m --variant h3-plug4 --model-dir /w/h3-base --variant-dir /workspace/h3-rs/plug4 --tables /workspace/h3-rs/plug4/plug4.tables --weights mxfp8 --decoder-weights mxfp8 --pv8 --task fl2va --first-image /workspace/h3-rs/assets/first.png --last-image /workspace/h3-rs/assets/last.png",
            "plug-ref2va" => "/w/h3-bin generate --backend cuda --seed 7 --width 1344 --height 768 --frames 124 --prompt P --out /root/out/plug-ref2va.y4m --variant h3-plug4 --model-dir /w/h3-base --variant-dir /root/plug4 --transformer /w/h3-ref2va/transformer_ref --tables /root/plug4-ref.tables --weights mxfp8 --decoder-weights mxfp8 --pv8 --task ref2va --ref image:/workspace/h3-rs/assets/first.png",
            "base49-fast" => "/w/h3-bin generate --backend cuda --seed 7 --width 1344 --height 768 --frames 124 --prompt P --out /root/out/base49-fast.y4m --variant h3 --model-dir /w/h3-base --weights mxfp8 --decoder-weights mxfp8 --pv8",
            "omni" => "/w/h3-bin generate --backend cuda --variant fasth3-omni --model-dir /w/h3-base --variant-dir /w/fasth3-omni-8step-v1 --tables /workspace/h3-rs/tables/omni.tables --seed 7 --width 1344 --height 768 --frames 124 --prompt P --task ref2va --ref image:/workspace/h3-rs/assets/first.png --out /root/out/omni.y4m",
            "trim" => "/w/h3-bin generate --backend cuda --variant fasth3-trim --model-dir /w/h3-base --transformer /w/fasth3-trim-8step-nvfp4/diffusion_models/fastvideo_fasth3_trim_8step_nvfp4.safetensors --width 1344 --height 768 --frames 124 --seed 7 --prompt P --out /root/out/trim.y4m",
            "pika-fl2va" => "/w/h3-bin generate --backend cuda --variant pika-flex --model-dir /w/h3-base --variant-dir /w/pika-h3-flex --tables /workspace/h3-rs/tables/pika.tables --seed 7 --width 1344 --height 768 --frames 124 --prompt P --task fl2va --first-image /workspace/h3-rs/assets/first.png --last-image /workspace/h3-rs/assets/last.png --out /root/out/pika-fl2va.y4m",
            "regen2k" => "/w/h3-bin regen2k --backend cuda --regen-dir /w/open-h3-regenerate-2k --model-dir /w/h3-base --draft /workspace/h3-rs/assets/fox.y4m --adaln-table /workspace/h3-rs/assets/adaln_t_table.bin --prompt P --out /root/out/regen.y4m",
            _ => panic!("{name}"),
        }
    }

    /// The file / directory name: the logs ran from pod copies (`/root/v2.tables`,
    /// `/w/h3-8step`), the catalog from `/workspace/w/h3-rs/tables/v2.tables`.
    fn tail(path: &str) -> String {
        path.trim_end_matches('/')
            .rsplit('/')
            .next()
            .unwrap()
            .to_string()
    }

    fn flag_map(argv: &[String]) -> (String, Vec<(String, String)>) {
        let sub = argv[1].clone();
        let m = parse_flags(&argv[2..], H3_VALUE_FLAGS, H3_BOOL_FLAGS, H3_ALIAS).unwrap();
        let mut out = Vec::new();
        for (k, v) in &m.0 {
            if IGNORE.contains(&k.as_str()) {
                continue;
            }
            let mut v = flag_text(v);
            if v.starts_with('/') {
                v = tail(&v);
            }
            if v.contains(':') && (k == "--ref" || k == "--continue") {
                let (kind, rest) = v.split_once(':').unwrap();
                v = format!("{kind}:{}", tail(rest));
            }
            if CLAP_DEFAULTS.contains(&(k.as_str(), v.as_str())) {
                continue;
            }
            out.push((k.clone(), v));
        }
        out.sort();
        (sub, out)
    }

    fn built(c: &Catalog, inp: Value, media: Option<H3Media>, fn_: &str) -> Vec<String> {
        let j = c.validate(&inp).unwrap();
        if fn_ == "regen" {
            return h3_regen2k_argv(
                &j,
                &settings(),
                &c.knobs,
                Path::new("/x/fox.y4m"),
                Path::new("out-2k.y4m"),
                7,
            )
            .unwrap();
        }
        h3_generate_argv(
            &j,
            &settings(),
            &c.knobs,
            &media.unwrap_or_default(),
            Path::new("out.y4m"),
            7,
        )
        .unwrap()
    }

    fn ref_media() -> H3Media {
        H3Media {
            references: vec![("image".into(), "/x/assets/first.png".into(), None)],
            ..Default::default()
        }
    }

    fn fl_media() -> H3Media {
        H3Media {
            first_image: Some("/x/assets/first.png".into()),
            last_image: Some("/x/assets/last.png".into()),
            ..Default::default()
        }
    }

    #[test]
    fn h3_lines_match_the_run_logs() {
        let c = catalog();
        let cases: Vec<(&str, Value, Option<H3Media>, &str)> = vec![
            (
                "v2-best",
                json!({"engine": "h3", "model": "fasth3-v2", "prompt": "P"}),
                None,
                "generate",
            ),
            (
                "v2-budget",
                json!({"engine": "h3", "model": "fasth3-v2", "prompt": "P", "options": {"vsa_sparsity": 0.9, "dense_first_steps": 2}}),
                None,
                "generate",
            ),
            (
                "v2-stream",
                json!({"engine": "h3", "model": "fasth3-v2", "prompt": "P", "options": {"stream_blocks": true}}),
                None,
                "generate",
            ),
            (
                "plug-fast",
                json!({"engine": "h3", "model": "h3-plug4", "prompt": "P"}),
                None,
                "generate",
            ),
            (
                "plug-fl2va",
                json!({"engine": "h3", "model": "h3-plug4", "task": "fl2v", "prompt": "P", "inputs": {"first_image": {"url": "https://e/a.png"}, "last_image": {"url": "https://e/b.png"}}}),
                Some(fl_media()),
                "generate",
            ),
            (
                "plug-ref2va",
                json!({"engine": "h3", "model": "h3-plug4", "task": "ref2v", "prompt": "P", "inputs": {"references": [{"url": "https://e/a.png"}]}}),
                Some(ref_media()),
                "generate",
            ),
            (
                "base49-fast",
                json!({"engine": "h3", "model": "h3", "prompt": "P", "options": {"steps": 49, "tables": false, "attention": "dense"}}),
                None,
                "generate",
            ),
            (
                "omni",
                json!({"engine": "h3", "model": "fasth3-omni", "task": "ref2v", "prompt": "P", "inputs": {"references": [{"url": "https://e/a.png"}]}, "options": {"weights": "native", "decoder_weights": "native", "vsa_qk8": false, "pv8": false}}),
                Some(ref_media()),
                "generate",
            ),
            (
                "trim",
                json!({"engine": "h3", "model": "fasth3-trim", "prompt": "P", "options": {"decoder_weights": "native", "vsa_qk8": false, "pv8": false}}),
                None,
                "generate",
            ),
            (
                "pika-fl2va",
                json!({"engine": "h3", "model": "pika-flex", "task": "fl2v", "prompt": "P", "inputs": {"first_image": {"url": "https://e/a.png"}, "last_image": {"url": "https://e/b.png"}}, "options": {"weights": "native", "decoder_weights": "native", "pv8": false}}),
                Some(fl_media()),
                "generate",
            ),
            (
                "regen2k",
                json!({"engine": "h3", "model": "regen-2k", "task": "regen2k", "prompt": "P", "width": 2560, "height": 1440, "inputs": {"draft": {"url": "https://e/d.mp4"}}, "options": {"weights": "native", "pv8": false}}),
                None,
                "regen",
            ),
        ];
        for (name, inp, media, fn_) in cases {
            let line: Vec<String> = log_line(name).split(' ').map(str::to_string).collect();
            let (want_sub, want) = flag_map(&line);
            let (got_sub, mut got) = flag_map(&built(&c, inp, media, fn_));
            if name == "base49-fast" {
                // 49 steps and dense attention are the base's own preset: the log left
                // them out, the model's knobs spell them.
                got.retain(|(k, _)| k != "--steps" && k != "--attention");
            }
            assert_eq!(got_sub, want_sub, "{name}");
            assert_eq!(got, want, "{name}");
        }
    }

    #[test]
    fn h3_serve_line_and_key() {
        let c = catalog();
        let s = settings();
        let j = c
            .validate(&json!({"engine": "h3", "model": "fasth3-v2", "prompt": "P", "upscale": 2}))
            .unwrap();
        let argv = h3_serve_argv(&j, &s, &c.knobs, 7861).unwrap();
        assert_eq!(
            &argv[..6],
            [
                "h3",
                "serve",
                "--backend",
                "cuda",
                "--addr",
                "127.0.0.1:7861"
            ]
        );
        let i = argv.iter().position(|a| a == "--max-frames").unwrap();
        assert_eq!(argv[i + 1], "362");
        let i = argv.iter().position(|a| a == "--upscale").unwrap();
        assert_eq!(Some(&argv[i + 1]), s.esrgan.as_ref());
        assert!(!argv.contains(&"--task".to_string()));
        let j2 = c
            .validate(&json!({"engine": "h3", "model": "fasth3-v2", "prompt": "Q", "upscale": 2}))
            .unwrap();
        assert_eq!(
            h3_serve_key(&j, &s, &c.knobs).unwrap(),
            h3_serve_key(&j2, &s, &c.knobs).unwrap()
        );
        let r = c.validate(&json!({"engine": "h3", "model": "h3-turbo4-ref", "task": "ref2v", "prompt": "P", "inputs": {"references": [{"url": "https://e/a.png"}]}})).unwrap();
        let rargv = h3_serve_argv(&r, &s, &c.knobs, 7861).unwrap();
        let i = rargv.iter().position(|a| a == "--task").unwrap();
        assert_eq!(rargv[i + 1], "ref2va");
        let v1 = c
            .validate(&json!({"engine": "h3", "model": "fasth3-v1", "prompt": "P"}))
            .unwrap();
        assert_ne!(
            h3_serve_key(&j, &s, &c.knobs).unwrap(),
            h3_serve_key(&v1, &s, &c.knobs).unwrap()
        );
        assert_eq!(command_line(&argv), "h3 serve --backend cuda --addr 127.0.0.1:7861 --variant fasth3-v2 --model-dir /workspace/w/h3-8step --transformer /workspace/w/h3-8step-mxfp8/transformer.safetensors --tables /workspace/w/h3-rs/tables/v2.tables --vsa-qk8 --pv8 --weights mxfp8 --decoder-weights mxfp8 --max-pixels 1032192 --max-frames 362 --upscale /w/realesr-animevideov3.safetensors");
    }

    #[test]
    fn h3_synthetic_serve_line() {
        let c = catalog();
        let s = EngineSettings {
            h3_bin: "/b/h3".into(),
            backend: "cpu".into(),
            synthetic: true,
            ..Default::default()
        };
        let j = c.validate(&json!({"engine": "h3", "model": "fasth3-v2", "prompt": "P", "width": 480, "height": 272})).unwrap();
        assert_eq!(
            &h3_serve_argv(&j, &s, &c.knobs, 7871).unwrap()[..7],
            [
                "/b/h3",
                "serve",
                "--backend",
                "cpu",
                "--addr",
                "127.0.0.1:7871",
                "--synthetic"
            ]
        );
    }

    #[test]
    fn h3_tables_follow_steps() {
        let c = catalog();
        let j = c
            .validate(&json!({"engine": "h3", "model": "h3", "prompt": "P"}))
            .unwrap();
        let argv = h3_generate_argv(
            &j,
            &settings(),
            &c.knobs,
            &H3Media::default(),
            Path::new("o.y4m"),
            1,
        )
        .unwrap();
        assert!(argv.contains(&"--tables".to_string()));
        let i = argv.iter().position(|a| a == "--steps").unwrap();
        assert_eq!(argv[i + 1], "25");
        let j = c
            .validate(
                &json!({"engine": "h3", "model": "h3", "prompt": "P", "options": {"steps": 30}}),
            )
            .unwrap();
        assert!(!h3_generate_argv(
            &j,
            &settings(),
            &c.knobs,
            &H3Media::default(),
            Path::new("o.y4m"),
            1
        )
        .unwrap()
        .contains(&"--tables".to_string()));
    }

    #[test]
    fn h3_turbo_ref_uses_transformer_ref() {
        let c = catalog();
        let s = settings();
        let j = c.validate(&json!({"engine": "h3", "model": "h3-turbo4-ref", "task": "ref2v", "prompt": "P", "inputs": {"references": [{"url": "https://e/a.png"}]}})).unwrap();
        let argv = h3_generate_argv(&j, &s, &c.knobs, &ref_media(), Path::new("o.y4m"), 1).unwrap();
        let i = argv.iter().position(|a| a == "--transformer").unwrap();
        assert_eq!(argv[i + 1], s.h3_ref_transformer);
        assert!(!argv.contains(&"--tables".to_string()));
        let i = argv.iter().position(|a| a == "--variant-dir").unwrap();
        assert!(argv[i + 1].ends_with("h3-ref2va/Minimax-h3-Turbo"));
    }

    #[test]
    fn h3_preview_and_upscale_need_their_weights() {
        let c = catalog();
        let bare = EngineSettings::default();
        let s = settings();
        let j = c.validate(&json!({"engine": "h3", "model": "fasth3-v2", "prompt": "P", "options": {"preview": true}})).unwrap();
        assert!(h3_generate_argv(
            &j,
            &bare,
            &c.knobs,
            &H3Media::default(),
            Path::new("o.y4m"),
            1
        )
        .unwrap_err()
        .message
        .contains("TAEH3"));
        let argv =
            h3_generate_argv(&j, &s, &c.knobs, &H3Media::default(), Path::new("o.y4m"), 1).unwrap();
        assert!(argv.contains(&"--preview".to_string()));
        let i = argv.iter().position(|a| a == "--tae").unwrap();
        assert_eq!(Some(&argv[i + 1]), s.taeh3.as_ref());
        let j = c
            .validate(&json!({"engine": "h3", "model": "fasth3-v2", "prompt": "P", "upscale": 3}))
            .unwrap();
        assert!(h3_generate_argv(
            &j,
            &bare,
            &c.knobs,
            &H3Media::default(),
            Path::new("o.y4m"),
            1
        )
        .unwrap_err()
        .message
        .contains("Real-ESRGAN"));
        let argv =
            h3_generate_argv(&j, &s, &c.knobs, &H3Media::default(), Path::new("o.y4m"), 1).unwrap();
        let i = argv.iter().position(|a| a == "--upscale-factor").unwrap();
        assert_eq!(argv[i + 1], "3");
    }

    #[test]
    fn h3_continue_and_video_refs() {
        let c = catalog();
        let j = c.validate(&json!({"engine": "h3", "model": "h3-plug4", "task": "ref2v", "prompt": "P", "inputs": {"references": [{"url": "https://e/a.mp4"}]}})).unwrap();
        let m = H3Media {
            references: vec![("video".into(), "/x/r.y4m".into(), Some("/x/r.wav".into()))],
            ..Default::default()
        };
        let argv = h3_generate_argv(&j, &settings(), &c.knobs, &m, Path::new("o.y4m"), 1).unwrap();
        let i = argv.iter().position(|a| a == "--ref").unwrap();
        assert_eq!(argv[i + 1], "video:/x/r.y4m,audio=/x/r.wav");
        let m = H3Media {
            continue_: Some(("/x/c.y4m".into(), None)),
            ..Default::default()
        };
        let argv = h3_generate_argv(&j, &settings(), &c.knobs, &m, Path::new("o.y4m"), 1).unwrap();
        let i = argv.iter().position(|a| a == "--continue").unwrap();
        assert_eq!(argv[i + 1], "/x/c.y4m");
    }

    #[test]
    fn fastwan_serve_and_generate_lines() {
        let c = catalog();
        let s = settings();
        let j = c.validate(&json!({"engine": "fastwan", "model": "fastwan-1.3b", "prompt": "P", "options": {"weights": "int8", "attention": "vsa", "vsa_sparsity": 0.9, "tae_chunk": 4}, "upscale": 4})).unwrap();
        let argv = fastwan_serve_argv(&j, &s, &c.knobs, 7862).unwrap();
        let m = parse_flags(&argv[2..], FW_VALUE_FLAGS, FW_BOOL_FLAGS, &[]).unwrap();
        let g = |k: &str| flag_text(m.get(k).unwrap());
        assert_eq!(g("--model-dir"), "/workspace/w/fastwan");
        assert!(g("--tae").ends_with("auxiliary/tae/taew2_1.safetensors"));
        assert_eq!(
            (
                g("--weights"),
                g("--attention"),
                g("--vsa-sparsity"),
                g("--tae-chunk")
            ),
            ("int8".into(), "vsa".into(), "0.9".into(), "4".into())
        );
        assert_eq!(Some(g("--upscaler")), s.esrgan.clone());
        assert_eq!(
            (g("--max-frames"), g("--max-pixels")),
            ("81".into(), (832 * 480).to_string())
        );
        let j = c.validate(&json!({"engine": "fastwan", "model": "self-forcing-1.3b", "task": "long", "prompt": "P", "frames": 321, "options": {"cache_frames": 12, "sink_frames": 3}, "upscale": 2})).unwrap();
        let argv = fastwan_generate_argv(&j, &s, &c.knobs, Path::new("out.y4m"), 7).unwrap();
        let m = parse_flags(&argv[2..], FW_VALUE_FLAGS, FW_BOOL_FLAGS, &[]).unwrap();
        let g = |k: &str| flag_text(m.get(k).unwrap());
        assert_eq!(m.get("--causal"), Some(&FlagVal::Bool(true)));
        assert_eq!(
            (g("--attention"), g("--cache-frames"), g("--sink-frames")),
            ("dense".into(), "12".into(), "3".into())
        );
        assert!(
            g("--transformer").ends_with("self-forcing-1.3b-f16.safetensors")
                && g("--text-encoder").ends_with("fastwan/text_encoder")
        );
        assert!(g("--tokenizer").ends_with("tokenizer/tokenizer.json") && g("--frames") == "321");
        assert_eq!(Some(g("--upscale")), s.esrgan.clone());
        assert_eq!(g("--upscale-x"), "2");
        let j = c.validate(&json!({"engine": "fastwan", "model": "history-forcing-1.3b", "task": "long", "prompt": "P"})).unwrap();
        let argv = fastwan_generate_argv(&j, &s, &c.knobs, Path::new("o.y4m"), 7).unwrap();
        let m = parse_flags(&argv[2..], FW_VALUE_FLAGS, FW_BOOL_FLAGS, &[]).unwrap();
        assert_eq!(flag_text(m.get("--cache-frames").unwrap()), "21");
    }

    #[test]
    fn fastwan_synthetic_serve_line() {
        let c = catalog();
        let s = EngineSettings {
            fastwan_bin: "/b/fastwan".into(),
            backend: "cpu".into(),
            synthetic: true,
            ..Default::default()
        };
        let j = c.validate(&json!({"engine": "fastwan", "model": "fastwan-1.3b", "prompt": "P", "width": 480, "height": 272, "frames": 17, "upscale": 2})).unwrap();
        let argv = fastwan_serve_argv(&j, &s, &c.knobs, 7872).unwrap();
        assert_eq!(
            &argv[..7],
            [
                "/b/fastwan",
                "serve",
                "--backend",
                "cpu",
                "--addr",
                "127.0.0.1:7872",
                "--synthetic"
            ]
        );
        assert!(argv.contains(&"--upscaler".to_string()));
    }

    #[test]
    fn fastvideo_bodies() {
        let c = catalog();
        let j = c.validate(&json!({"engine": "fastvideo", "model": "fasth3-4step-vsa", "task": "fl2v", "prompt": "P", "seed": 3, "frames": 124,
                                   "inputs": {"first_image": {"url": "https://e/a.png"}, "last_image": {"url": "https://e/b.png"}}})).unwrap();
        let urls = MediaUrls {
            first_image: Some("https://e/a.png".into()),
            last_image: Some("data:image/png;base64,AA==".into()),
            ..Default::default()
        };
        let b = fastvideo_body(&j, &c.knobs, &urls, Some(3)).unwrap();
        assert_eq!(
            Value::Object(b),
            json!({"model": "fasth3-4step-vsa", "prompt": "P", "seed": 3, "size": "1344x768", "num_frames": 124, "image_url": "https://e/a.png", "last_image_url": "data:image/png;base64,AA=="})
        );
        let j = c.validate(&json!({"engine": "fastvideo", "model": "ltx25-distill-sol", "task": "a2v", "prompt": "", "inputs": {"audio": {"url": "https://e/a.wav"}}, "options": {"fps": "50"}})).unwrap();
        let b = fastvideo_body(
            &j,
            &c.knobs,
            &MediaUrls {
                audio: Some("https://e/a.wav".into()),
                ..Default::default()
            },
            None,
        )
        .unwrap();
        assert_eq!(b["audio_url"], json!("https://e/a.wav"));
        assert_eq!(b["fps"], json!(50));
        assert!(!b.contains_key("seed"));
        let j = c.validate(&json!({"engine": "fastvideo", "model": "ltx25-distill-sol", "task": "retake", "prompt": "P", "frames": 49, "inputs": {"video": {"url": "https://e/v.mp4"}}})).unwrap();
        let b = fastvideo_body(
            &j,
            &c.knobs,
            &MediaUrls {
                video: Some("https://e/v.mp4".into()),
                ..Default::default()
            },
            Some(1),
        )
        .unwrap();
        assert_eq!(b["video_url"], json!("https://e/v.mp4"));
        assert_eq!(b["start_s"], json!(0.0));
        assert!((b["end_s"].as_f64().unwrap() - 49.0 / 24.0).abs() < 1e-3);
        assert!(!b.contains_key("size") && !b.contains_key("num_frames") && !b.contains_key("fps"));
        let j = c.validate(&json!({"engine": "fastvideo", "model": "h3-ref2v-turbo", "task": "ref2v", "prompt": "P", "inputs": {"references": [{"url": "https://e/a.png"}]}, "options": {"reference_strength": 0.5}})).unwrap();
        let b = fastvideo_body(
            &j,
            &c.knobs,
            &MediaUrls {
                references: vec!["https://e/a.png".into()],
                ..Default::default()
            },
            None,
        )
        .unwrap();
        assert_eq!(b["reference_urls"], json!(["https://e/a.png"]));
        assert_eq!(b["reference_strength"], json!(0.5));
        assert!(!b.contains_key("prompt_expansion"));
        assert!(b.keys().all(|k| FV_FIELDS.contains(&k.as_str())));
    }

    #[test]
    fn fastvideo_flow_shift_is_not_a_knob() {
        let c = catalog();
        assert!(!c.knobs.contains_key("fv.flow_shift"));
        let e = c.validate(&json!({"engine": "fastvideo", "model": "wan22-ti2v-5b", "prompt": "P", "options": {"flow_shift": 5.0}})).unwrap_err();
        assert!(e.message.contains("flow_shift"));
    }

    #[test]
    fn command_line_is_shell_safe() {
        let argv: Vec<String> = ["/opt/studio/bin/h3", "generate", "--prompt", "it's a 'fox'"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            command_line(&argv),
            "h3 generate --prompt 'it'\"'\"'s a '\"'\"'fox'\"'\"''"
        );
    }

    #[test]
    fn parse_flags_collects_repeats_and_aliases() {
        let toks: Vec<String> = ["--ref", "a", "--ref", "b", "--qk8", "--weights=mxfp8"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let m = parse_flags(&toks, H3_VALUE_FLAGS, H3_BOOL_FLAGS, H3_ALIAS).unwrap();
        assert_eq!(
            m.get("--ref"),
            Some(&FlagVal::List(vec!["a".into(), "b".into()]))
        );
        assert_eq!(m.get("--vsa-qk8"), Some(&FlagVal::Bool(true)));
        assert_eq!(m.get("--weights"), Some(&FlagVal::Str("mxfp8".into())));
        assert!(parse_flags(
            &["--nope".to_string()],
            H3_VALUE_FLAGS,
            H3_BOOL_FLAGS,
            H3_ALIAS
        )
        .is_err());
        assert!(parse_flags(
            &["--seed".to_string()],
            H3_VALUE_FLAGS,
            H3_BOOL_FLAGS,
            H3_ALIAS
        )
        .is_err());
    }
}
