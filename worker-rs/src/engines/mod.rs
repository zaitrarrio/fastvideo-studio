//! One driver per engine. Each `run(ctx, job)` returns a `RunResult`.

pub mod common;
pub mod fastvideo;
pub mod fastwan;
pub mod h3;

use crate::catalog::Knob;
use crate::flags::EngineSettings;
use crate::progress::{Reporter, Timings};
use crate::r2::ObjectStore;
use crate::sidecar::SidecarManager;
use crate::Log;
use indexmap::IndexMap;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// What a driver produced for the muxer and the uploader.
#[derive(Debug, Clone)]
pub struct RunResult {
    pub seed: i64,
    pub width: i64,
    pub height: i64,
    pub frames: i64,
    pub fps: i64,
    /// h3 / fastwan: the clip (already upscaled when asked)
    pub video_y4m: Option<PathBuf>,
    pub audio_wav: Option<PathBuf>,
    /// fastvideo: fv-serve's own MP4
    pub video_mp4: Option<PathBuf>,
    /// (factor, width, height)
    pub upscaled: Option<(i64, i64, i64)>,
    pub timings: Timings,
    pub command: String,
    pub extra: Map<String, Value>,
}

impl RunResult {
    pub fn new(seed: i64, width: i64, height: i64, frames: i64, fps: i64) -> RunResult {
        RunResult {
            seed,
            width,
            height,
            frames,
            fps,
            video_y4m: None,
            audio_wav: None,
            video_mp4: None,
            upscaled: None,
            timings: Timings::default(),
            command: String::new(),
            extra: Map::new(),
        }
    }
}

pub type EngineEnv = HashMap<String, HashMap<String, String>>;

/// What a driver needs from the worker for one job.
pub struct Ctx<'a> {
    pub settings: &'a EngineSettings,
    pub knobs: &'a IndexMap<String, Knob>,
    pub sidecars: &'a SidecarManager,
    pub ports: &'a HashMap<String, u16>,
    /// per engine
    pub env: &'a EngineEnv,
    pub job_dir: PathBuf,
    pub store: Option<Arc<dyn ObjectStore>>,
    pub bucket: Option<String>,
    pub log: Log,
    pub report: &'a Reporter<'a>,
    pub fv_serve_bin: String,
    pub fv_config: String,
    pub fv_state_dir: String,
    pub fv_weights: String,
    /// tests: a fake fv-serve already listens on the port
    pub fv_fake: bool,
    /// The catalog's models (the fastvideo driver groups its `fv-serve`
    /// process by `recipe.process`); empty in unit tests.
    pub models: &'a [crate::catalog::Model],
}

impl<'a> Ctx<'a> {
    pub fn port(&self, engine: &str) -> u16 {
        *self.ports.get(engine).unwrap_or(&0)
    }

    pub fn env_of(&self, engine: &str) -> HashMap<String, String> {
        self.env.get(engine).cloned().unwrap_or_default()
    }

    pub fn store_ref(&self) -> Option<&dyn ObjectStore> {
        self.store.as_deref()
    }
}

/// `random.randrange(1, 2**31)`.
pub fn random_seed() -> i64 {
    let mut buf = [0u8; 4];
    let ok = std::fs::File::open("/dev/urandom")
        .and_then(|mut f| std::io::Read::read_exact(&mut f, &mut buf))
        .is_ok();
    let v = if ok {
        u32::from_le_bytes(buf)
    } else {
        (crate::now_s() * 1e6) as u32 ^ std::process::id()
    };
    1 + (v % (u32::MAX / 2)) as i64
}
