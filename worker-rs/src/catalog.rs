//! `catalog/catalog.json`: load it and validate a job against it.
//!
//! A knob the catalog does not list is refused, never defaulted (design §2).
//! Everything here is pure: no I/O apart from reading the catalog file.

use crate::errors::{JobError, JobResult};
use crate::py::{is_int, py_eq, py_repr, py_str, truthy};
use indexmap::IndexMap;
use serde_json::{Map, Value};
use std::path::{Path, PathBuf};

pub const OPS: [&str; 3] = ["generate", "info", "warm"];
pub const UPSCALE_VALUES: [i64; 4] = [0, 2, 3, 4];
pub const PROMPT_MAX_CHARS: usize = 2000;
/// Tasks whose prompt fv-serve lets be empty (the source media carries the content).
pub const PROMPT_OPTIONAL_TASKS: [&str; 3] = ["extend", "retake", "a2v"];
pub const MEDIA_INPUT_KEYS: [&str; 6] = ["first_image", "last_image", "references", "audio", "video", "draft"];
pub const MAX_REFERENCES: usize = 12;
/// Knobs any engine's model may list (`expand.prompt`: the worker handles
/// them, no `cli` / `field`). The page sends them by short name like the others.
pub const SHARED_KNOB_PREFIXES: [&str; 1] = ["expand"];

/// Knob-id prefixes per engine: `options` keys may be short (`weights`) or full (`h3.weights`).
pub fn knob_prefix(engine: &str) -> &'static str {
    match engine {
        "h3" => "h3",
        "fastwan" => "fw",
        "fastvideo" => "fv",
        _ => "",
    }
}

pub fn default_catalog_path() -> PathBuf {
    if let Some(p) = std::env::var_os("STUDIO_CATALOG").filter(|v| !v.is_empty()) {
        return PathBuf::from(p);
    }
    let here = Path::new(env!("CARGO_MANIFEST_DIR")).join("../catalog/catalog.json");
    for p in [here.clone(), PathBuf::from("/opt/studio/catalog/catalog.json")] {
        if p.is_file() {
            return p;
        }
    }
    here
}

/// `"17n+5"`: frames = step·n + offset, within [min, max].
#[derive(Debug, Clone, PartialEq)]
pub struct FrameGrid {
    pub step: i64,
    pub offset: i64,
    pub min: i64,
    pub max: i64,
    pub default: i64,
    pub choices: Vec<i64>,
}

impl FrameGrid {
    pub fn parse(spec: &Value) -> anyhow::Result<FrameGrid> {
        let grid = spec["grid"].as_str().unwrap_or("");
        let t = grid.trim();
        let (a, b) = t
            .split_once('n')
            .and_then(|(a, rest)| rest.trim_start().strip_prefix('+').map(|b| (a.trim(), b.trim())))
            .ok_or_else(|| anyhow::anyhow!("bad frame grid {grid:?}"))?;
        let step: i64 = a.parse().map_err(|_| anyhow::anyhow!("bad frame grid {grid:?}"))?;
        let offset: i64 = b.parse().map_err(|_| anyhow::anyhow!("bad frame grid {grid:?}"))?;
        let int = |k: &str| spec[k].as_f64().map(|f| f as i64).ok_or_else(|| anyhow::anyhow!("frames.{k} missing"));
        Ok(FrameGrid {
            step,
            offset,
            min: int("min")?,
            max: int("max")?,
            default: int("default")?,
            choices: spec["choices"].as_array().map(|a| a.iter().filter_map(|c| c.as_f64().map(|f| f as i64)).collect()).unwrap_or_default(),
        })
    }

    /// None when `frames` is on the grid and in range, else why not.
    pub fn check(&self, frames: i64) -> Option<String> {
        if frames < self.min || frames > self.max {
            return Some(format!("frames must be between {} and {}", self.min, self.max));
        }
        if (frames - self.offset).rem_euclid(self.step) != 0 {
            let eg: Vec<String> = self.choices.iter().take(4).map(|c| c.to_string()).collect();
            return Some(format!("frames must be on the {}n+{} grid (e.g. {})", self.step, self.offset, eg.join(", ")));
        }
        None
    }
}

#[derive(Debug, Clone)]
pub struct Knob {
    pub id: String,
    pub label: String,
    /// select | bool | int | float
    pub type_: String,
    pub default: Value,
    pub options: Vec<String>,
    pub min: Option<Value>,
    pub max: Option<Value>,
    pub group: String,
    pub cli: Option<String>,
    pub fv_field: Option<String>,
    /// `{dep_short: want}`
    pub when: Map<String, Value>,
}

impl Knob {
    pub fn short(&self) -> &str {
        self.id.split_once('.').map(|(_, s)| s).unwrap_or(&self.id)
    }

    pub fn parse(kid: &str, spec: &Value) -> Knob {
        let opt_str = |k: &str| spec.get(k).and_then(|v| v.as_str()).map(str::to_string);
        Knob {
            id: kid.to_string(),
            label: opt_str("label").unwrap_or_else(|| kid.to_string()),
            type_: spec["type"].as_str().unwrap_or("").to_string(),
            default: spec.get("default").cloned().unwrap_or(Value::Null),
            options: spec["options"]
                .as_array()
                .map(|a| a.iter().map(|o| py_str(o.get(0).unwrap_or(&Value::Null))).collect())
                .unwrap_or_default(),
            min: spec.get("min").filter(|v| !v.is_null()).cloned(),
            max: spec.get("max").filter(|v| !v.is_null()).cloned(),
            group: opt_str("group").unwrap_or_default(),
            cli: opt_str("cli"),
            fv_field: opt_str("field"),
            when: spec.get("when").and_then(|v| v.as_object()).cloned().unwrap_or_default(),
        }
    }

    /// The value in the knob's type, or Err with a reason.
    pub fn coerce(&self, value: &Value) -> Result<Value, String> {
        match self.type_.as_str() {
            "bool" => {
                if let Value::Bool(b) = value {
                    return Ok(Value::Bool(*b));
                }
                if let Value::String(s) = value {
                    let l = s.to_lowercase();
                    if l == "true" || l == "false" {
                        return Ok(Value::Bool(l == "true"));
                    }
                }
                Err("expects true or false".into())
            }
            "select" => {
                let s = match value {
                    Value::Bool(b) => if *b { "true".to_string() } else { "false".to_string() },
                    v => py_str(v),
                };
                if !self.options.contains(&s) {
                    return Err(format!("must be one of {}", self.options.join(", ")));
                }
                Ok(Value::String(s))
            }
            "int" => {
                let f = match value {
                    Value::Number(n) => n.as_f64().unwrap_or(f64::NAN),
                    _ => return Err("expects an integer".into()),
                };
                if f.fract() != 0.0 || !f.is_finite() {
                    return Err("expects an integer".into());
                }
                self.check_range(f)?;
                Ok(Value::from(f as i64))
            }
            "float" => {
                let f = match value {
                    Value::Number(n) => n.as_f64().unwrap_or(f64::NAN),
                    _ => return Err("expects a number".into()),
                };
                self.check_range(f)?;
                Ok(Value::from(f))
            }
            t => Err(format!("unknown knob type {t}")),
        }
    }

    fn check_range(&self, v: f64) -> Result<(), String> {
        if let Some(min) = &self.min {
            if min.as_f64().map(|m| v < m).unwrap_or(false) {
                return Err(format!("must be at least {}", py_str(min)));
            }
        }
        if let Some(max) = &self.max {
            if max.as_f64().map(|m| v > m).unwrap_or(false) {
                return Err(format!("must be at most {}", py_str(max)));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Task {
    pub id: String,
    pub label: String,
    pub inputs: Vec<String>,
    pub optional: Vec<String>,
}

impl Task {
    pub fn required(&self) -> Vec<&str> {
        self.inputs.iter().filter(|i| !self.optional.contains(i)).map(String::as_str).collect()
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    pub id: String,
    pub engine: String,
    pub family: String,
    pub label: String,
    pub tasks: Vec<String>,
    pub sizes: Vec<(i64, i64)>,
    pub default_size: (i64, i64),
    pub frames: FrameGrid,
    pub knobs: Vec<String>,
    pub recipe: Map<String, Value>,
    pub trees: Vec<String>,
    pub stage: Vec<String>,
    pub vram_gib: f64,
    pub min_sm: i64,
    pub fps: i64,
    pub audio: bool,
    pub steps: Option<i64>,
    pub cli_only: bool,
    pub hidden: bool,
    pub negative_prompt: bool,
    pub aliases: Vec<String>,
    pub knob_defaults: Map<String, Value>,
    pub tier: Option<String>,
    pub verify: Option<String>,
    pub raw: Value,
}

fn str_list(v: &Value) -> Vec<String> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default()
}

impl Model {
    pub fn parse(spec: &Value) -> anyhow::Result<Model> {
        let s = |k: &str| spec[k].as_str().map(str::to_string).ok_or_else(|| anyhow::anyhow!("model: {k} missing"));
        let id = s("id")?;
        let sizes: Vec<(i64, i64)> = spec["sizes"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("model {id}: sizes missing"))?
            .iter()
            .map(|wh| (wh[0].as_f64().unwrap_or(0.0) as i64, wh[1].as_f64().unwrap_or(0.0) as i64))
            .collect();
        let ds = &spec["default_size"];
        Ok(Model {
            engine: s("engine")?,
            family: s("family")?,
            label: spec["label"].as_str().unwrap_or(&id).to_string(),
            tasks: str_list(&spec["tasks"]),
            sizes,
            default_size: (ds[0].as_f64().unwrap_or(0.0) as i64, ds[1].as_f64().unwrap_or(0.0) as i64),
            frames: FrameGrid::parse(&spec["frames"]).map_err(|e| anyhow::anyhow!("model {id}: {e}"))?,
            knobs: str_list(&spec["knobs"]),
            recipe: spec["recipe"].as_object().cloned().unwrap_or_default(),
            trees: str_list(&spec["trees"]),
            stage: str_list(&spec["stage"]),
            vram_gib: spec["vram_gib"].as_f64().unwrap_or(0.0),
            min_sm: spec["min_sm"].as_f64().unwrap_or(0.0) as i64,
            fps: spec["fps"].as_f64().unwrap_or(24.0) as i64,
            audio: truthy(&spec["audio"]),
            steps: spec["steps"].as_f64().map(|f| f as i64),
            cli_only: truthy(&spec["cli_only"]),
            hidden: truthy(&spec["hidden"]),
            negative_prompt: truthy(&spec["negative_prompt"]),
            aliases: str_list(&spec["aliases"]),
            knob_defaults: spec["knob_defaults"].as_object().cloned().unwrap_or_default(),
            tier: spec["tier"].as_str().map(str::to_string),
            verify: spec["verify"].as_str().map(str::to_string),
            raw: spec.clone(),
            id,
        })
    }
}

/// One `<media>`: exactly one of url / b64 / r2.
#[derive(Debug, Clone, PartialEq)]
pub struct Media {
    /// url | b64 | r2
    pub kind: String,
    pub value: String,
    pub mime: Option<String>,
}

impl Media {
    pub fn new(kind: &str, value: &str) -> Media {
        Media { kind: kind.into(), value: value.into(), mime: None }
    }

    pub fn inline_bytes(&self) -> usize {
        if self.kind != "b64" {
            return 0;
        }
        let s = self.value.trim_end_matches('=');
        s.chars().count() * 3 / 4
    }

    /// `base64.b64decode(value, validate=False)`: non-alphabet characters are
    /// skipped, `-_` are not accepted, padding must be present.
    pub fn decode_b64(&self) -> Result<Vec<u8>, String> {
        b64_decode(&self.value)
    }
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub fn b64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let n = chunk.len();
        let b = [chunk[0], if n > 1 { chunk[1] } else { 0 }, if n > 2 { chunk[2] } else { 0 }];
        let v = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(B64[(v >> 18) as usize & 63] as char);
        out.push(B64[(v >> 12) as usize & 63] as char);
        out.push(if n > 1 { B64[(v >> 6) as usize & 63] as char } else { '=' });
        out.push(if n > 2 { B64[v as usize & 63] as char } else { '=' });
    }
    out
}

pub fn b64_decode(s: &str) -> Result<Vec<u8>, String> {
    let mut vals: Vec<u8> = Vec::with_capacity(s.len());
    let mut pad = 0usize;
    for c in s.bytes() {
        if c == b'=' {
            pad += 1;
            continue;
        }
        if let Some(i) = B64.iter().position(|&b| b == c) {
            if pad > 0 {
                // data after padding: binascii stops at the padding
                break;
            }
            vals.push(i as u8);
        }
    }
    if vals.len() % 4 == 1 {
        return Err("Incorrect padding".into());
    }
    if vals.len() % 4 != 0 && pad == 0 {
        return Err("Incorrect padding".into());
    }
    let mut out = Vec::with_capacity(vals.len() * 3 / 4);
    for q in vals.chunks(4) {
        let n = q.len();
        let v = ((q[0] as u32) << 18) | ((q[1] as u32) << 12) | ((if n > 2 { q[2] } else { 0 } as u32) << 6) | (if n > 3 { q[3] } else { 0 } as u32);
        out.push((v >> 16) as u8);
        if n > 2 {
            out.push((v >> 8) as u8);
        }
        if n > 3 {
            out.push(v as u8);
        }
    }
    Ok(out)
}

/// A validated `op: generate` input.
#[derive(Debug, Clone)]
pub struct Job {
    pub engine: String,
    pub model: Model,
    pub task: String,
    pub prompt: String,
    pub width: i64,
    pub height: i64,
    pub frames: i64,
    pub seed: Option<i64>,
    /// effective values, full knob ids, defaults applied
    pub options: Map<String, Value>,
    /// the knobs the request set, full knob ids
    pub explicit: Map<String, Value>,
    pub inputs: JobInputs,
    pub upscale: i64,
    pub negative_prompt: Option<String>,
    pub client: Value,
}

/// Media per input key (a list for `references`), in request order.
#[derive(Debug, Clone, Default)]
pub struct JobInputs {
    pub first_image: Option<Media>,
    pub last_image: Option<Media>,
    pub references: Vec<Media>,
    pub audio: Option<Media>,
    pub video: Option<Media>,
    pub draft: Option<Media>,
    /// The keys in the order the request gave them (for `job.json`).
    pub order: Vec<String>,
}

impl JobInputs {
    pub fn get(&self, key: &str) -> Option<&Media> {
        match key {
            "first_image" => self.first_image.as_ref(),
            "last_image" => self.last_image.as_ref(),
            "audio" => self.audio.as_ref(),
            "video" => self.video.as_ref(),
            "draft" => self.draft.as_ref(),
            _ => None,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }
}

impl Job {
    /// An effective option by its short name (`weights`).
    pub fn opt(&self, short: &str) -> Option<&Value> {
        self.options.get(&format!("{}.{}", knob_prefix(&self.engine), short))
    }

    /// `job.opt(short, default)`: the option (which may be null) or the default.
    pub fn opt_or<'a>(&'a self, short: &str, default: &'a Value) -> &'a Value {
        self.opt(short).unwrap_or(default)
    }

    pub fn opt_truthy(&self, short: &str) -> bool {
        self.opt(short).map(truthy).unwrap_or(false)
    }

    pub fn is_explicit(&self, short: &str) -> bool {
        self.explicit.contains_key(&format!("{}.{}", knob_prefix(&self.engine), short))
    }

    /// The clip's frame rate: the `fps` knob when the model has one, else the model's.
    pub fn fps_for(&self) -> i64 {
        match self.opt("fps") {
            Some(v) if truthy(v) => match v {
                Value::String(s) => s.trim().parse().unwrap_or(self.model.fps),
                Value::Number(n) => n.as_f64().map(|f| f as i64).unwrap_or(self.model.fps),
                _ => self.model.fps,
            },
            _ => self.model.fps,
        }
    }
}

pub struct Catalog {
    pub data: Value,
    pub version: Value,
    pub weights_root: String,
    pub engines: Map<String, Value>,
    pub tasks: IndexMap<String, Task>,
    pub knobs: IndexMap<String, Knob>,
    pub models: Vec<Model>,
    pub trees: Map<String, Value>,
    pub upload_limits: Map<String, Value>,
}

impl Catalog {
    pub fn new(data: Value) -> anyhow::Result<Catalog> {
        let engines = data["engines"].as_object().cloned().ok_or_else(|| anyhow::anyhow!("catalog: engines missing"))?;
        let mut tasks = IndexMap::new();
        for (k, v) in data["tasks"].as_object().ok_or_else(|| anyhow::anyhow!("catalog: tasks missing"))? {
            tasks.insert(
                k.clone(),
                Task {
                    id: k.clone(),
                    label: v["label"].as_str().unwrap_or(k).to_string(),
                    inputs: str_list(&v["inputs"]),
                    optional: str_list(&v["optional"]),
                },
            );
        }
        let mut knobs = IndexMap::new();
        for (k, v) in data["knobs"].as_object().ok_or_else(|| anyhow::anyhow!("catalog: knobs missing"))? {
            knobs.insert(k.clone(), Knob::parse(k, v));
        }
        let models = data["models"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("catalog: models missing"))?
            .iter()
            .map(Model::parse)
            .collect::<anyhow::Result<Vec<_>>>()?;
        let c = Catalog {
            version: data.get("version").cloned().unwrap_or(Value::Null),
            weights_root: data["weights_root"].as_str().unwrap_or("/workspace/w").to_string(),
            engines,
            tasks,
            knobs,
            models,
            trees: data["trees"].as_object().cloned().unwrap_or_default(),
            upload_limits: data["upload_limits"].as_object().cloned().unwrap_or_default(),
            data,
        };
        c.check()?;
        Ok(c)
    }

    pub fn load(path: Option<&Path>) -> anyhow::Result<Catalog> {
        let p = path.map(Path::to_path_buf).unwrap_or_else(default_catalog_path);
        let text = std::fs::read_to_string(&p).map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?;
        let data: Value = serde_json::from_str(&text).map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?;
        Catalog::new(data)
    }

    /// Internal consistency: every model's knobs and tasks exist.
    fn check(&self) -> anyhow::Result<()> {
        for m in &self.models {
            for k in &m.knobs {
                if !self.knobs.contains_key(k) {
                    anyhow::bail!("catalog: model {} lists unknown knob {k}", m.id);
                }
                let head = k.split_once('.').map(|(h, _)| h).unwrap_or(k);
                if !k.starts_with(&format!("{}.", knob_prefix(&m.engine))) && !SHARED_KNOB_PREFIXES.contains(&head) {
                    anyhow::bail!("catalog: model {} lists knob {k} of another engine", m.id);
                }
            }
            for t in &m.tasks {
                if !self.tasks.contains_key(t) {
                    anyhow::bail!("catalog: model {} lists unknown task {t}", m.id);
                }
            }
            if !self.engines.contains_key(&m.engine) {
                anyhow::bail!("catalog: model {} has unknown engine {}", m.id, m.engine);
            }
        }
        Ok(())
    }

    pub fn inline_limit(&self) -> usize {
        self.upload_limits.get("inline_b64_bytes").and_then(|v| v.as_f64()).map(|f| f as usize).unwrap_or(6 * 1024 * 1024)
    }

    pub fn model(&self, engine: &str, model_id: &str) -> Option<&Model> {
        self.models.iter().find(|m| m.engine == engine && (m.id == model_id || m.aliases.iter().any(|a| a == model_id)))
    }

    pub fn models_of(&self, engine: &str) -> Vec<&Model> {
        self.models.iter().filter(|m| m.engine == engine).collect()
    }

    /// Effective defaults: the knob's, then the model's `knob_defaults`.
    pub fn knob_defaults(&self, model: &Model) -> Map<String, Value> {
        let mut out = Map::new();
        for k in &model.knobs {
            out.insert(k.clone(), self.knobs.get(k).map(|kn| kn.default.clone()).unwrap_or(Value::Null));
        }
        for (k, v) in &model.knob_defaults {
            if out.contains_key(k) {
                out.insert(k.clone(), v.clone());
            }
        }
        out
    }

    // ---- validation ----------------------------------------------------

    /// A `Job` from an `op: generate` input, or `JobError(where="validate")`.
    pub fn validate(&self, inp: &Value) -> JobResult<Job> {
        let bad = |m: String| JobError::validate(m);
        let obj = inp.as_object().ok_or_else(|| bad("input must be a JSON object".into()))?;
        let op = obj.get("op").cloned().unwrap_or(Value::String("generate".into()));
        if op != Value::String("generate".into()) {
            return Err(bad(format!("op {} is not generate", py_repr(&op))));
        }
        let engine = obj.get("engine").and_then(|v| v.as_str()).filter(|e| self.engines.contains_key(*e));
        let engine = engine.ok_or_else(|| bad(format!("engine must be one of {}", self.engines.keys().cloned().collect::<Vec<_>>().join(", "))))?;
        let model_id = obj.get("model").and_then(|v| v.as_str()).ok_or_else(|| bad("model is required".into()))?;
        let model = self.model(engine, model_id).ok_or_else(|| bad(format!("unknown {engine} model {}", py_str_repr(model_id))))?;
        let task_id = match obj.get("task") {
            Some(v) => v.clone(),
            None => Value::String(model.tasks[0].clone()),
        };
        let task_str = task_id.as_str().map(str::to_string);
        if task_str.as_ref().map(|t| !model.tasks.contains(t)).unwrap_or(true) {
            return Err(bad(format!("model {} does not do task {} (it does {})", model.id, py_repr(&task_id), model.tasks.join(", "))));
        }
        let task_id = task_str.unwrap();
        let task = &self.tasks[&task_id];

        let prompt = match obj.get("prompt") {
            None => "",
            Some(Value::String(s)) => s.as_str(),
            Some(_) => return Err(bad("prompt must be a string".into())),
        };
        let prompt = prompt.trim().to_string();
        if prompt.chars().count() > PROMPT_MAX_CHARS {
            return Err(bad(format!("prompt is over {PROMPT_MAX_CHARS} characters")));
        }
        if prompt.is_empty() && !PROMPT_OPTIONAL_TASKS.contains(&task_id.as_str()) {
            return Err(bad("prompt is required".into()));
        }
        let negative = match obj.get("negative_prompt") {
            None | Some(Value::Null) => None,
            Some(v) => {
                if !model.negative_prompt {
                    return Err(bad(format!("model {} takes no negative_prompt", model.id)));
                }
                match v.as_str() {
                    Some(s) if s.chars().count() <= PROMPT_MAX_CHARS => Some(s.to_string()),
                    _ => return Err(bad("negative_prompt must be a string of at most 2000 characters".into())),
                }
            }
        };

        let width = obj.get("width").cloned().unwrap_or(Value::from(model.default_size.0));
        let height = obj.get("height").cloned().unwrap_or(Value::from(model.default_size.1));
        if !is_int(&width) || !is_int(&height) {
            return Err(bad("width and height must be integers".into()));
        }
        let (w, h) = (width.as_i64().unwrap_or(-1), height.as_i64().unwrap_or(-1));
        if !model.sizes.contains(&(w, h)) {
            let sizes: Vec<String> = model.sizes.iter().map(|(w, h)| format!("{w}x{h}")).collect();
            return Err(bad(format!("{}x{} is not one of {}'s sizes ({})", py_str(&width), py_str(&height), model.id, sizes.join(", "))));
        }
        let frames = obj.get("frames").cloned().unwrap_or(Value::from(model.frames.default));
        if !is_int(&frames) {
            return Err(bad("frames must be an integer".into()));
        }
        let frames = frames.as_i64().unwrap_or(-1);
        if let Some(why) = model.frames.check(frames) {
            return Err(bad(why));
        }
        let seed = match obj.get("seed") {
            None | Some(Value::Null) => None,
            Some(v) => {
                if !is_int(v) || v.as_i64().map(|s| s < 0).unwrap_or(true) {
                    return Err(bad("seed must be a non-negative integer".into()));
                }
                v.as_i64()
            }
        };

        let empty = Value::Object(Map::new());
        let inputs = self.validate_inputs(obj.get("inputs").filter(|v| !v.is_null()).unwrap_or(&empty), task, model)?;
        let mut explicit = self.validate_options(obj.get("options").filter(|v| !v.is_null()).unwrap_or(&empty), model)?;
        let mut options = self.knob_defaults(model);
        for (k, v) in &explicit {
            options.insert(k.clone(), v.clone());
        }
        // `when`: a conditional knob is accepted only when its condition holds
        // (a value equal to its default is dropped instead of refused: the
        // page may send every control).
        let auto = Value::String("auto".into());
        for kid in explicit.keys().cloned().collect::<Vec<_>>() {
            let knob = &self.knobs[&kid];
            for (dep_short, want) in &knob.when {
                let dep = format!("{}.{}", knob_prefix(&model.engine), dep_short);
                let have = options.get(&dep).cloned().unwrap_or(Value::Null);
                // `auto` leaves the choice to the variant's preset: the knob stays.
                if !py_eq(&have, want) && have != auto {
                    if py_eq(&explicit[&kid], &knob.default) {
                        explicit.shift_remove(&kid);
                        options.insert(kid.clone(), knob.default.clone());
                    } else {
                        return Err(bad(format!("option {} applies only when {dep_short} is {} (it is {})", knob.short(), py_repr(want), py_repr(&have))));
                    }
                }
            }
        }
        let upscale = self.validate_upscale(obj.get("upscale"), &mut options, model, &mut explicit)?;
        Ok(Job {
            engine: engine.to_string(),
            model: model.clone(),
            task: task_id,
            prompt,
            width: w,
            height: h,
            frames,
            seed,
            options,
            explicit,
            inputs,
            upscale,
            negative_prompt: negative,
            client: obj.get("client").cloned().unwrap_or(Value::Null),
        })
    }

    fn validate_inputs(&self, raw: &Value, task: &Task, _model: &Model) -> JobResult<JobInputs> {
        let bad = |m: String| JobError::validate(m);
        let raw = raw.as_object().ok_or_else(|| bad("inputs must be an object".into()))?;
        for k in raw.keys() {
            if !MEDIA_INPUT_KEYS.contains(&k.as_str()) {
                return Err(bad(format!("unknown input {}", py_str_repr(k))));
            }
            if !task.inputs.contains(k) {
                return Err(bad(format!("task {} takes no input {}", task.id, py_str_repr(k))));
            }
        }
        for k in task.required() {
            let missing = match raw.get(k) {
                None | Some(Value::Null) => true,
                Some(Value::Array(a)) => a.is_empty(),
                Some(Value::Object(o)) => o.is_empty(),
                _ => false,
            };
            if missing {
                return Err(bad(format!("task {} needs input {}", task.id, py_str_repr(k))));
            }
        }
        let mut out = JobInputs::default();
        for (k, v) in raw {
            if v.is_null() {
                continue;
            }
            if k == "references" {
                let list = v.as_array().filter(|a| !a.is_empty()).ok_or_else(|| bad("references must be a non-empty list of media".into()))?;
                if list.len() > MAX_REFERENCES {
                    return Err(bad(format!("at most {MAX_REFERENCES} references")));
                }
                let mut refs = Vec::new();
                for (i, x) in list.iter().enumerate() {
                    refs.push(self.validate_media(x, &format!("references[{i}]"))?);
                }
                out.references = refs;
            } else {
                let m = self.validate_media(v, k)?;
                match k.as_str() {
                    "first_image" => out.first_image = Some(m),
                    "last_image" => out.last_image = Some(m),
                    "audio" => out.audio = Some(m),
                    "video" => out.video = Some(m),
                    "draft" => out.draft = Some(m),
                    _ => unreachable!(),
                }
            }
            out.order.push(k.clone());
        }
        Ok(out)
    }

    fn validate_media(&self, v: &Value, what: &str) -> JobResult<Media> {
        let bad = |m: String| JobError::validate(m);
        let o = v.as_object().ok_or_else(|| bad(format!("{what} must be a media object")))?;
        let keys: Vec<&str> = ["url", "b64", "r2"].into_iter().filter(|k| o.get(*k).map(truthy).unwrap_or(false)).collect();
        if keys.len() != 1 {
            return Err(bad(format!("{what} must have exactly one of url, b64, r2")));
        }
        let kind = keys[0];
        let value = o[kind].as_str().ok_or_else(|| bad(format!("{what}.{kind} must be a string")))?;
        if kind == "url" {
            if !value.starts_with("https://") {
                return Err(bad(format!("{what}.url must be https://")));
            }
            return Ok(Media::new("url", value));
        }
        if kind == "r2" {
            if !value.starts_with("uploads/") || value.contains("..") {
                return Err(bad(format!("{what}.r2 must be an uploads/<key> object")));
            }
            return Ok(Media::new("r2", value));
        }
        let mime = match o.get("mime").and_then(|m| m.as_str()) {
            Some(m) if m.contains('/') => m.to_string(),
            _ => return Err(bad(format!("{what}.mime is required with b64"))),
        };
        let m = Media { kind: "b64".into(), value: value.into(), mime: Some(mime) };
        if m.inline_bytes() > self.inline_limit() {
            return Err(bad(format!("{what} inline data is over the {} MiB limit; upload it instead", self.inline_limit() / (1024 * 1024))));
        }
        Ok(m)
    }

    fn validate_options(&self, raw: &Value, model: &Model) -> JobResult<Map<String, Value>> {
        let bad = |m: String| JobError::validate(m);
        let raw = raw.as_object().ok_or_else(|| bad("options must be an object".into()))?;
        let prefix = knob_prefix(&model.engine);
        let mut out = Map::new();
        for (key, value) in raw {
            let mut kid = if key.contains('.') { key.clone() } else { format!("{prefix}.{key}") };
            if !model.knobs.contains(&kid) {
                // A shared knob by its short name (`prompt` → `expand.prompt`).
                let shared: Vec<&String> = model
                    .knobs
                    .iter()
                    .filter(|k| k.split_once('.').map(|(h, s)| SHARED_KNOB_PREFIXES.contains(&h) && s == key).unwrap_or(false))
                    .collect();
                if shared.len() != 1 {
                    return Err(bad(format!("model {} has no option {}", model.id, py_str_repr(key))));
                }
                kid = shared[0].clone();
            }
            let knob = &self.knobs[&kid];
            if value.is_null() {
                continue;
            }
            match knob.coerce(value) {
                Ok(v) => {
                    out.insert(kid, v);
                }
                Err(e) => return Err(bad(format!("option {}: {e}", knob.short()))),
            }
        }
        Ok(out)
    }

    /// Top-level `upscale: 0|2|3|4` (jobs.md). The `*.upscale_factor` knob
    /// is the page's control for it: accepted in `options` and ignored.
    fn validate_upscale(&self, raw: Option<&Value>, options: &mut Map<String, Value>, model: &Model, explicit: &mut Map<String, Value>) -> JobResult<i64> {
        let bad = |m: String| JobError::validate(m);
        let kid = format!("{}.upscale_factor", knob_prefix(&model.engine));
        explicit.shift_remove(&kid);
        let factor = match raw {
            None | Some(Value::Null) | Some(Value::Bool(false)) => 0,
            Some(Value::Bool(true)) => return Err(bad("upscale must be 0, 2, 3 or 4 (not true)".into())),
            Some(v) => {
                if !is_int(v) || !v.as_i64().map(|i| UPSCALE_VALUES.contains(&i)).unwrap_or(false) {
                    return Err(bad("upscale must be 0, 2, 3 or 4".into()));
                }
                v.as_i64().unwrap()
            }
        };
        if factor != 0 {
            if !model.knobs.contains(&kid) {
                return Err(bad(format!("model {} has no upscaler", model.id)));
            }
            let opts = &self.knobs[&kid].options;
            if !opts.contains(&factor.to_string()) {
                let list: Vec<&str> = opts.iter().filter(|o| *o != "0").map(String::as_str).collect();
                return Err(bad(format!("model {} upscales by {} only", model.id, list.join(", "))));
            }
        }
        if model.knobs.contains(&kid) {
            options.insert(kid, Value::String(factor.to_string()));
        }
        Ok(factor)
    }
}

use crate::py::py_str_repr;

#[cfg(test)]
pub mod tests {
    use super::*;
    use serde_json::json;

    pub fn catalog() -> Catalog {
        Catalog::load(Some(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../catalog/catalog.json"))).unwrap()
    }

    fn good() -> Value {
        json!({"engine": "h3", "model": "fasth3-v2", "task": "t2v", "prompt": "a fox", "width": 1344, "height": 768,
               "frames": 124, "seed": 7, "options": {"weights": "mxfp8", "pv8": true}, "upscale": 0, "client": {"job": "x"}})
    }

    fn patched(patch: Value) -> Value {
        let mut g = good();
        for (k, v) in patch.as_object().unwrap() {
            g[k] = v.clone();
        }
        g
    }

    #[test]
    fn every_model_default_job_validates() {
        let c = catalog();
        for m in &c.models {
            let task = &m.tasks[0];
            let mut inputs = Map::new();
            for k in c.tasks[task].required() {
                inputs.insert(k.to_string(), if k == "references" { json!([{"url": "https://e/x.png"}]) } else { json!({"url": "https://e/x.mp4"}) });
            }
            let j = c.validate(&json!({"engine": m.engine, "model": m.id, "task": task, "prompt": "p", "inputs": inputs})).unwrap();
            assert_eq!((j.width, j.height), m.default_size);
            assert_eq!(j.frames, m.frames.default);
            let keys: Vec<&String> = j.options.keys().collect();
            let want: Vec<&String> = m.knobs.iter().collect();
            assert_eq!(keys, want);
        }
    }

    #[test]
    fn good_job() {
        let c = catalog();
        let j = c.validate(&good()).unwrap();
        assert_eq!(j.model.id, "fasth3-v2");
        assert_eq!(j.seed, Some(7));
        assert_eq!(j.upscale, 0);
        assert_eq!(Value::Object(j.explicit.clone()), json!({"h3.weights": "mxfp8", "h3.pv8": true}));
        assert_eq!(j.options["h3.decoder_weights"], json!("mxfp8"));
        assert_eq!(j.client, json!({"job": "x"}));
    }

    #[test]
    fn alias_resolves_and_full_ids() {
        let c = catalog();
        let j = c.validate(&patched(json!({"model": "draft", "width": 832, "height": 480}))).unwrap();
        assert_eq!(j.model.id, "h3-plug4");
        let j = c.validate(&patched(json!({"options": {"h3.weights": "native"}}))).unwrap();
        assert_eq!(j.options["h3.weights"], json!("native"));
    }

    #[test]
    fn bad_jobs_refused() {
        let c = catalog();
        for (patch, needle) in [
            (json!({"frames": 125}), "17n+5"),
            (json!({"frames": 100}), "between"),
            (json!({"frames": 2000}), "between"),
            (json!({"width": 1000, "height": 768}), "sizes"),
            (json!({"options": {"nope": 1}}), "no option"),
            (json!({"options": {"vsa_sparsity": 0.3}}), "at least"),
            (json!({"options": {"dense_first_steps": 11}}), "at most"),
            (json!({"options": {"dense_first_steps": 1.5}}), "integer"),
            (json!({"options": {"weights": "fp4"}}), "one of"),
            (json!({"options": {"pv8": "maybe"}}), "true or false"),
            (json!({"task": "fl2v"}), "does not do task"),
            (json!({"engine": "nope"}), "engine"),
            (json!({"model": "nope"}), "unknown"),
            (json!({"prompt": ""}), "required"),
            (json!({"prompt": "x".repeat(2001)}), "2000"),
            (json!({"negative_prompt": "ugly"}), "negative_prompt"),
            (json!({"upscale": 5}), "upscale"),
            (json!({"upscale": true}), "upscale"),
            (json!({"seed": -1}), "seed"),
            (json!({"op": "warm"}), "generate"),
        ] {
            let e = c.validate(&patched(patch.clone())).err().unwrap_or_else(|| panic!("{patch} accepted"));
            assert_eq!(e.where_, "validate");
            assert!(e.message.contains(needle), "{patch}: {}", e.message);
        }
    }

    #[test]
    fn when_knob_with_default_is_dropped_not_refused() {
        let c = catalog();
        let j = c.validate(&patched(json!({"options": {"attention": "auto", "vsa_sparsity": 0.9, "dense_first_steps": 0}}))).unwrap();
        assert_eq!(j.explicit["h3.vsa_sparsity"], json!(0.9));
        assert!(j.explicit.contains_key("h3.dense_first_steps"));
        let j = c.validate(&patched(json!({"options": {"attention": "dense", "vsa_sparsity": 0.8}}))).unwrap();
        assert!(!j.explicit.contains_key("h3.vsa_sparsity"));
        let e = c.validate(&patched(json!({"options": {"attention": "dense", "vsa_sparsity": 0.9}}))).unwrap_err();
        assert!(e.message.contains("applies only when"), "{}", e.message);
        assert_eq!(e.message, "option vsa_sparsity applies only when attention is 'vsa' (it is 'dense')");
    }

    #[test]
    fn upscale_is_top_level_and_the_knob_is_ignored() {
        let c = catalog();
        let j = c.validate(&patched(json!({"upscale": 2, "options": {"upscale_factor": "4"}}))).unwrap();
        assert_eq!(j.upscale, 2);
        assert!(!j.explicit.contains_key("h3.upscale_factor"));
        let e = c.validate(&json!({"engine": "fastwan", "model": "fastwan-1.3b", "prompt": "p", "upscale": 3})).unwrap_err();
        assert!(e.message.contains("upscales by 2, 4"), "{}", e.message);
        let e = c.validate(&json!({"engine": "fastvideo", "model": "ltx25-distill-sol", "prompt": "p", "upscale": 2})).unwrap_err();
        assert!(e.message.contains("no upscaler"));
    }

    #[test]
    fn media_rules() {
        let c = catalog();
        let base = json!({"engine": "h3", "model": "h3-plug4", "task": "fl2v", "prompt": "p", "width": 832, "height": 480});
        let with = |inputs: Value| {
            let mut b = base.clone();
            b["inputs"] = inputs;
            b
        };
        assert!(c.validate(&base).unwrap_err().message.contains("needs input 'first_image'"));
        let j = c.validate(&with(json!({"first_image": {"url": "https://e/a.png"}}))).unwrap();
        assert_eq!(j.inputs.first_image.as_ref().unwrap().kind, "url");
        assert!(j.inputs.last_image.is_none());
        assert!(c.validate(&with(json!({"first_image": {"url": "http://e/a.png"}}))).unwrap_err().message.contains("https"));
        assert!(c.validate(&with(json!({"first_image": {"url": "https://e/a.png", "r2": "uploads/x"}}))).unwrap_err().message.contains("exactly one"));
        assert!(c.validate(&with(json!({"first_image": {"r2": "jobs/x/video.mp4"}}))).unwrap_err().message.contains("uploads/"));
        assert!(c.validate(&with(json!({"first_image": {"url": "https://e/a.png"}, "audio": {"url": "https://e/a.wav"}}))).unwrap_err().message.contains("takes no input"));
        let small = b64_encode(&[b'x'; 100]);
        let j = c.validate(&with(json!({"first_image": {"b64": small, "mime": "image/png"}}))).unwrap();
        assert_eq!(j.inputs.first_image.as_ref().unwrap().inline_bytes(), 100);
        let big = b64_encode(&vec![b'x'; c.inline_limit() + 1]);
        assert!(c.validate(&with(json!({"first_image": {"b64": big, "mime": "image/png"}}))).unwrap_err().message.contains("MiB limit"));
        assert!(c.validate(&with(json!({"first_image": {"b64": small}}))).unwrap_err().message.contains("mime"));
    }

    #[test]
    fn references_list() {
        let c = catalog();
        let base = json!({"engine": "fastvideo", "model": "h3-ref2v-turbo", "task": "ref2v", "prompt": "p"});
        let with = |inputs: Value| {
            let mut b = base.clone();
            b["inputs"] = inputs;
            b
        };
        let j = c.validate(&with(json!({"references": [{"url": "https://e/a.png"}, {"r2": "uploads/b.mp4"}]}))).unwrap();
        assert_eq!(j.inputs.references.iter().map(|m| m.kind.as_str()).collect::<Vec<_>>(), ["url", "r2"]);
        let e = c.validate(&with(json!({"references": []}))).unwrap_err();
        assert!(e.message.contains("needs input") || e.message.contains("non-empty"));
        let many: Vec<Value> = (0..13).map(|_| json!({"url": "https://e/a.png"})).collect();
        assert!(c.validate(&with(json!({"references": many}))).unwrap_err().message.contains("at most 12"));
    }

    #[test]
    fn fastvideo_null_default_knobs_absent_unless_set() {
        let c = catalog();
        let j = c.validate(&json!({"engine": "fastvideo", "model": "wan22-ti2v-5b", "prompt": "p"})).unwrap();
        assert!(j.options["fv.steps"].is_null() && j.explicit.is_empty());
        let j = c.validate(&json!({"engine": "fastvideo", "model": "wan22-ti2v-5b", "prompt": "p", "options": {"steps": 20}})).unwrap();
        assert_eq!(Value::Object(j.explicit), json!({"fv.steps": 20}));
    }

    #[test]
    fn frame_grid() {
        let g = FrameGrid::parse(&json!({"grid": "12n+9", "min": 57, "max": 969, "default": 165, "choices": [165]})).unwrap();
        assert!(g.check(165).is_none() && g.check(57).is_none());
        assert!(g.check(166).unwrap().contains("12n+9"));
        assert!(g.check(45).unwrap().contains("between"));
        assert!(FrameGrid::parse(&json!({"grid": "x", "min": 1, "max": 2, "default": 1})).is_err());
    }

    #[test]
    fn base64_roundtrip() {
        for n in 0..40 {
            let data: Vec<u8> = (0..n).map(|i| (i * 37 % 251) as u8).collect();
            let s = b64_encode(&data);
            assert_eq!(b64_decode(&s).unwrap(), data);
        }
        assert_eq!(b64_encode(b"\x89PNG"), "iVBORw==");
        assert_eq!(b64_decode("iVBORw0KGgo=").unwrap(), b"\x89PNG\r\n\x1a\n");
        assert!(b64_decode("abcde").is_err());
    }
}
