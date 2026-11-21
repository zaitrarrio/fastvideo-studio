//! The worker: configuration from the environment, the ops (`generate`,
//! `info`, `warm`, `r2`, `check`), the generate pipeline (validate → stage →
//! expand → run → mux → upload) and the jobs.md output / error shapes.

use crate::catalog::{Catalog, Job};
use crate::engines::{fastvideo, fastwan, h3, Ctx, EngineEnv, RunResult};
use crate::errors::{JobError, JobResult};
use crate::flags::{self, EngineSettings};
use crate::media as md;
use crate::progress::{ProgressFn, Reporter};
use crate::py::{dumps, iso_utc, py_str};
use crate::r2::{LocalStore, ObjectStore, R2Config, R2Store};
use crate::sidecar::{http_info_probe, ping_probe, Sidecar, SidecarManager};
use crate::upsampler as up;
use crate::weights::{Roots, WeightsResolver};
use crate::{gpu, mux, now_s, EnvMap, Log, VERSION};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub fn env_flag(e: &EnvMap, name: &str, default: bool) -> bool {
    match e.get(name) {
        None => default,
        Some(v) => matches!(
            v.trim().to_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub catalog_path: PathBuf,
    pub bin_dir: PathBuf,
    pub fv_serve_bin: String,
    pub fv_config: String,
    pub fv_state_dir: String,
    pub jobs_dir: PathBuf,
    pub ports: HashMap<String, u16>,
    pub llama_bin: String,
    /// a separate CUDA runtime for h3 / fastwan; empty since the one-runtime image
    pub cuda12_lib: String,
    pub backend: String,
    pub synthetic: bool,
    pub fv_fake: bool,
    pub image: String,
    pub url_ttl_s: u64,
    pub esrgan: Option<String>,
    pub taeh3: Option<String>,
    pub keep_job_dirs: bool,
    pub load_timeout_s: f64,
}

impl Config {
    pub fn defaults(catalog_path: PathBuf) -> Config {
        Config {
            catalog_path,
            bin_dir: "/opt/studio/bin".into(),
            fv_serve_bin: "/opt/fastvideo-rs/bin/fv-serve".into(),
            fv_config: "/opt/studio/configs/fastvideo.toml".into(),
            fv_state_dir: "/workspace/fv-state".into(),
            jobs_dir: "/workspace/jobs".into(),
            ports: [
                ("h3", 7861u16),
                ("fastwan", 7862),
                ("fastvideo", 8010), // not 8000: `studio-worker serve` listens there on a pod
                ("upsampler", up::DEFAULT_PORT),
            ]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
            llama_bin: up::LLAMA_BIN.into(),
            cuda12_lib: String::new(),
            backend: "cuda".into(),
            synthetic: false,
            fv_fake: false,
            image: String::new(),
            url_ttl_s: 24 * 3600,
            esrgan: None,
            taeh3: None,
            keep_job_dirs: false,
            load_timeout_s: 1800.0,
        }
    }

    pub fn from_env(e: &EnvMap) -> Config {
        let get = |k: &str| e.get(k).filter(|v| !v.is_empty()).cloned();
        let mut c = Config::defaults(
            get("STUDIO_CATALOG")
                .map(PathBuf::from)
                .unwrap_or_else(crate::catalog::default_catalog_path),
        );
        if let Some(v) = get("STUDIO_BIN_DIR") {
            c.bin_dir = v.into();
        }
        c.fv_serve_bin = get("STUDIO_FV_SERVE").unwrap_or(c.fv_serve_bin);
        c.fv_config = get("STUDIO_FV_CONFIG").unwrap_or(c.fv_config);
        c.llama_bin = get("STUDIO_LLAMA_SERVER").unwrap_or(c.llama_bin);
        c.fv_state_dir = get("STUDIO_FV_STATE_DIR").unwrap_or(c.fv_state_dir);
        if let Some(v) = get("STUDIO_JOBS_DIR") {
            c.jobs_dir = v.into();
        }
        for (eng, var) in [
            ("h3", "STUDIO_H3_PORT"),
            ("fastwan", "STUDIO_FASTWAN_PORT"),
            ("fastvideo", "STUDIO_FV_PORT"),
        ] {
            if let Some(p) = get(var).and_then(|v| v.trim().parse::<u16>().ok()) {
                c.ports.insert(eng.into(), p);
            }
        }
        c.cuda12_lib = e.get("STUDIO_CUDA12_LIB").cloned().unwrap_or(c.cuda12_lib);
        c.backend = get("STUDIO_BACKEND").unwrap_or(c.backend);
        c.synthetic = env_flag(e, "STUDIO_SYNTHETIC", false);
        c.fv_fake = env_flag(e, "STUDIO_FV_FAKE", false);
        c.image = e.get("STUDIO_IMAGE").cloned().unwrap_or_default();
        c.url_ttl_s = get("STUDIO_URL_TTL_S")
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(c.url_ttl_s);
        c.esrgan = Some(get("STUDIO_ESRGAN").unwrap_or_else(|| {
            "/workspace/w/h3-rs/assets/realesr-animevideov3.safetensors".into()
        }));
        c.taeh3 = Some(
            get("STUDIO_TAEH3")
                .unwrap_or_else(|| "/workspace/w/auxiliary/tae/taeh3.safetensors".into()),
        );
        c.keep_job_dirs = env_flag(e, "STUDIO_KEEP_JOB_DIRS", false);
        c.load_timeout_s = get("STUDIO_LOAD_TIMEOUT_S")
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(c.load_timeout_s);
        c
    }
}

pub struct Worker {
    pub cfg: Config,
    pub catalog: Catalog,
    pub store: Option<Arc<dyn ObjectStore>>,
    pub bucket: Option<String>,
    pub log: Log,
    pub env: EnvMap,
    pub roots: Roots,
    pub weights: WeightsResolver,
    pub sidecars: SidecarManager,
    pub settings: EngineSettings,
    pub started: f64,
    pub jobs_done: AtomicU64,
}

impl Worker {
    pub fn new(
        cfg: Config,
        catalog: Catalog,
        store: Option<Arc<dyn ObjectStore>>,
        bucket: Option<String>,
        env: EnvMap,
        log: Log,
    ) -> Worker {
        let roots = Roots::from_catalog(&catalog.trees, &env);
        let weights = WeightsResolver::new(
            roots.clone(),
            store.clone(),
            log.clone(),
            env_flag(&env, "STUDIO_STAGE_DRY_RUN", false),
        );
        let sidecars = SidecarManager::new(log.clone(), cfg.load_timeout_s);
        let settings = EngineSettings {
            h3_bin: cfg.bin_dir.join("h3").to_string_lossy().into_owned(),
            fastwan_bin: cfg.bin_dir.join("fastwan").to_string_lossy().into_owned(),
            backend: cfg.backend.clone(),
            synthetic: cfg.synthetic,
            esrgan: cfg.esrgan.clone().filter(|p| Path::new(p).is_file()),
            taeh3: cfg.taeh3.clone().filter(|p| Path::new(p).is_file()),
            h3_ref_transformer: roots
                .link_root
                .join("h3-ref2va")
                .join("transformer_ref")
                .to_string_lossy()
                .into_owned(),
        };
        Worker {
            cfg,
            catalog,
            store,
            bucket,
            log,
            env,
            roots,
            weights,
            sidecars,
            settings,
            started: now_s(),
            jobs_done: AtomicU64::new(0),
        }
    }

    pub fn from_env(log: Log) -> anyhow::Result<Worker> {
        Self::from_env_map(&crate::process_env(), log)
    }

    pub fn from_env_map(env: &EnvMap, log: Log) -> anyhow::Result<Worker> {
        let cfg = Config::from_env(env);
        let catalog = Catalog::load(Some(&cfg.catalog_path))?;
        let mut store: Option<Arc<dyn ObjectStore>> = None;
        let mut bucket: Option<String> = None;
        if let Some(local) = env.get("STUDIO_R2_LOCAL_DIR").filter(|v| !v.is_empty()) {
            store = Some(Arc::new(LocalStore::new(PathBuf::from(local))));
            bucket = Some(
                env.get("STUDIO_R2_BUCKET")
                    .filter(|v| !v.is_empty())
                    .or_else(|| env.get("FV_R2_BUCKET").filter(|v| !v.is_empty()))
                    .cloned()
                    .unwrap_or_else(|| "studio".into()),
            );
        } else if let Some(r2) = R2Config::from_env(env) {
            bucket = Some(r2.bucket.clone());
            store = Some(Arc::new(R2Store::new(r2)));
        }
        Ok(Worker::new(cfg, catalog, store, bucket, env.clone(), log))
    }

    // ---- environment per engine ------------------------------------------------

    pub fn env_for(&self, engine: &str) -> HashMap<String, String> {
        let mut env = self.env.clone();
        if engine == "h3" || engine == "fastwan" {
            if !self.cfg.cuda12_lib.is_empty() {
                // STUDIO_CUDA12_LIB: a separate runtime ahead of the image's
                let prev = env
                    .get("LD_LIBRARY_PATH")
                    .filter(|v| !v.is_empty())
                    .map(|v| format!(":{v}"))
                    .unwrap_or_default();
                env.insert(
                    "LD_LIBRARY_PATH".into(),
                    format!("{}{prev}", self.cfg.cuda12_lib),
                );
            }
        } else {
            env.insert(
                "FV_WEIGHTS".into(),
                self.roots.link_root.to_string_lossy().into_owned(),
            );
            env.insert("FV_STATE_DIR".into(), self.cfg.fv_state_dir.clone());
            // fv-serve's artifact URLs: loopback, never the public base it
            // discovers. On a Runpod pod it builds
            // `https://<pod>-<its port>.proxy.runpod.net/files/...`, which
            // 404s for any port but the one exposed one and sends the
            // worker's own download out over the internet (2026-11-21: the
            // first clip job on a pod with fv-serve off port 8000).
            env.insert(
                "FV_PUBLIC_BASE_URL".into(),
                format!(
                    "http://127.0.0.1:{}",
                    self.cfg.ports.get("fastvideo").copied().unwrap_or(8010)
                ),
            );
            env.entry("RUST_LOG".into())
                .or_insert_with(|| "info".into());
        }
        env
    }

    fn engine_env(&self) -> EngineEnv {
        ["h3", "fastwan", "fastvideo"]
            .iter()
            .map(|e| (e.to_string(), self.env_for(e)))
            .collect()
    }

    // ---- ops -------------------------------------------------------------------------

    pub fn handle(&self, job: &Value, progress: Option<ProgressFn>) -> Value {
        let Some(inp) = job.get("input").filter(|i| i.is_object()) else {
            return JobError::validate("input must be an object").to_output();
        };
        let op = inp
            .get("op")
            .map(py_str)
            .unwrap_or_else(|| "generate".into());
        let engine = inp.get("engine").map(py_str).unwrap_or_default();
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> JobResult<Value> {
            match op.as_str() {
                "info" => Ok(self.info()),
                "warm" => self.warm(inp, progress),
                "generate" => {
                    let id = job
                        .get("id")
                        .filter(|v| crate::py::truthy(v))
                        .map(py_str)
                        .unwrap_or_else(|| format!("local-{}", now_s() as i64));
                    self.generate(&id, inp, progress)
                }
                "r2" => Ok(self.r2check()),
                "check" => Ok(self.check(inp)),
                other => Err(JobError::validate(format!(
                    "unknown op {}",
                    crate::py::py_str_repr(other)
                ))),
            }
        }));
        let err = match res {
            Ok(Ok(v)) => return v,
            Ok(Err(e)) => e,
            Err(p) => {
                let msg = p
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "panic".into());
                JobError {
                    where_: "engine",
                    message: format!("panic: {msg}"),
                    log_tail: vec![],
                    crash: true,
                }
            }
        };
        if err.crash {
            (self.log)(&format!("job crashed: {}", err.message));
            // Only the job's own engine's sidecar log: another engine's tail
            // misled the first smoke (fastwan's crash carried h3's log).
            let tail = self
                .sidecars
                .get(&engine)
                .map(|s| s.tail(50))
                .unwrap_or_default();
            return JobError::engine(err.message, tail).to_output();
        }
        (self.log)(&format!("job failed ({}): {}", err.where_, err.message));
        err.to_output()
    }

    /// Prompt expansion (the `expand.prompt` knob) for every engine through
    /// the llama-server sidecar (Qwen3-VL-4B-Instruct, fetched from R2 on
    /// first use). Returns the rewritten prompt, or None when off or when
    /// anything failed: a failed rewrite never fails the job.
    pub fn expand_prompt(
        &self,
        job: &mut Job,
        job_dir: &Path,
        report: &Reporter,
    ) -> Option<String> {
        let mode = up::mode_of(job);
        if mode == "off" || job.prompt.trim().is_empty() {
            return None;
        }
        let r = (|| -> anyhow::Result<Option<String>> {
            report.detail_force("staging", "prompt expansion model");
            self.weights.ensure_tree(up::TREE)?;
            let tree_dir = self
                .roots
                .link_root
                .join(up::TREE_NAME)
                .to_string_lossy()
                .into_owned();
            let port = *self.cfg.ports.get("upsampler").unwrap_or(&up::DEFAULT_PORT);
            let (argv, env) = up::plan_sidecar(
                &self.cfg.llama_bin,
                &tree_dir,
                port,
                Some(&self.env_for("fastvideo")),
                up::CTX,
                true,
            );
            let key = up::sidecar_key(&tree_dir, port, up::CTX, true);
            let base = up::base_url(port);
            let make = || {
                Sidecar::new(
                    up::ENGINE,
                    &key,
                    argv.clone(),
                    env.clone(),
                    up::llama_probe(&base, up::ALIAS),
                )
            };
            let loading =
                |_: &str, t: f64| report.detail("loading", &format!("prompt expansion {t:.0} s"));
            self.sidecars.ensure(
                up::ENGINE,
                &key,
                &make,
                up::default_vram_gib(),
                Some(&loading),
            )?;
            let mut image_b64: Option<String> = None;
            let mut image_mime: Option<String> = None;
            if let Some(first) = job
                .inputs
                .first_image
                .as_ref()
                .filter(|_| job.task == "i2v" || job.task == "fl2v")
            {
                let lm = md::fetch_media(
                    first,
                    &job_dir.join("in"),
                    "first_image",
                    self.store.as_deref(),
                    self.bucket.as_deref(),
                )?;
                image_b64 = Some(up::image_data_b64(&lm.path.to_string_lossy())?);
                image_mime = Some(lm.mime.clone());
            }
            let mut u = up::Upsampler::new(&base);
            u.log = self.log.clone();
            report.detail_force("encoding", "expanding the prompt");
            let new = u.expand(
                &job.prompt,
                &job.model.family,
                mode,
                job.negative_prompt.as_deref(),
                image_b64.as_deref(),
                image_mime.as_deref().unwrap_or("image/jpeg"),
                90.0,
            );
            if u.last_error.is_none() && !new.trim().is_empty() && new != job.prompt {
                job.prompt = new.clone();
                return Ok(Some(new));
            }
            Ok(None)
        })();
        match r {
            Ok(v) => v,
            Err(e) => {
                (self.log)(&format!("[upsampler] prompt expansion unavailable ({e:#}); generating with the original prompt"));
                None
            }
        }
    }

    /// `op: check`: the engines' own conformance checkers on this card
    /// (`h3 check --backend cuda`, `fastwan check --backend cuda`): every op
    /// against the CPU reference, the way the engines' CLAUDE.md trusts GPU
    /// code. Stops the sidecars first (the checkers want the GPU).
    pub fn check(&self, inp: &Value) -> Value {
        let engines: Vec<String> = inp
            .get("engines")
            .and_then(|e| e.as_array())
            .filter(|a| !a.is_empty())
            .map(|a| a.iter().map(py_str).collect())
            .unwrap_or_else(|| vec!["h3".into(), "fastwan".into()]);
        self.sidecars.stop_all();
        let mut results = Map::new();
        for name in engines {
            if name != "h3" && name != "fastwan" {
                results.insert(name, json!({"error": "no checker for this engine"}));
                continue;
            }
            let bin = self.cfg.bin_dir.join(&name).to_string_lossy().into_owned();
            let argv = vec![
                bin.clone(),
                "check".to_string(),
                "--backend".to_string(),
                self.cfg.backend.clone(),
            ];
            let t0 = now_s();
            let r = Command::new(&bin)
                .args(&argv[1..])
                .env_clear()
                .envs(self.env_for(&name))
                .output();
            match r {
                Ok(o) => {
                    let text = format!(
                        "{}{}",
                        String::from_utf8_lossy(&o.stdout),
                        String::from_utf8_lossy(&o.stderr)
                    );
                    let lines: Vec<&str> = text.trim().lines().collect();
                    let tail: Vec<&str> = lines[lines.len().saturating_sub(40)..].to_vec();
                    results.insert(name, json!({"exit": o.status.code(), "seconds": crate::py::round_to(now_s() - t0, 1), "tail": tail, "command": argv.join(" ")}));
                }
                Err(e) => {
                    results.insert(
                        name,
                        json!({"error": format!("OSError: {e}"), "command": argv.join(" ")}),
                    );
                }
            }
        }
        let g = gpu::query(0);
        json!({"op": "check", "backend": self.cfg.backend, "results": results, "gpu": g.as_ref().map(|g| g.name.clone()), "driver": g.as_ref().map(|g| g.driver.clone())})
    }

    /// `op: r2`: can this worker write the Studio bucket and read the
    /// weights store? Names and outcomes only, never a secret.
    pub fn r2check(&self) -> Value {
        let e = &self.env;
        let get = |a: &str, b: &str| {
            e.get(a)
                .filter(|v| !v.is_empty())
                .or_else(|| e.get(b).filter(|v| !v.is_empty()))
                .cloned()
        };
        let key = get("STUDIO_R2_ACCESS_KEY_ID", "FV_R2_ACCESS_KEY_ID").unwrap_or_default();
        let mut out = json!({
            "op": "r2", "bucket": self.bucket, "endpoint": get("STUDIO_R2_ENDPOINT", "FV_R2_ENDPOINT"),
            "access_key_id_len": key.chars().count(), "access_key_id_unsubstituted": key.starts_with("{{"),
            "weights_bucket": self.weights.weights_bucket,
        });
        let (Some(store), Some(bucket)) = (
            &self.store,
            self.bucket.as_deref().filter(|b| !b.is_empty()),
        ) else {
            out["write"] = json!("no object store configured");
            return out;
        };
        let probe = self.cfg.jobs_dir.join("r2-probe.txt");
        let _ = std::fs::create_dir_all(&self.cfg.jobs_dir);
        let _ = std::fs::write(
            &probe,
            format!("fastvideo-studio r2 probe {:.0}\n", now_s()),
        );
        match store.put_file(bucket, "probe/r2check.txt", &probe, Some("text/plain")) {
            Ok(()) => {
                out["write"] = json!("ok");
                out["presigned_url_host"] = json!(store
                    .presign_get(bucket, "probe/r2check.txt", 60)
                    .split('/')
                    .nth(2)
                    .unwrap_or(""));
            }
            Err(ex) => {
                out["write"] = json!(format!("{ex:#}").chars().take(300).collect::<String>())
            }
        }
        match store.list_keys(&self.weights.weights_bucket, "manifests/") {
            Ok(keys) => out["read_weights"] = json!(format!("ok ({} manifests)", keys.len())),
            Err(ex) => {
                out["read_weights"] = json!(format!("{ex:#}").chars().take(300).collect::<String>())
            }
        }
        out
    }

    pub fn info(&self) -> Value {
        let g = gpu::query(0);
        let mut trees = Map::new();
        for m in &self.catalog.models {
            for t in &m.trees {
                if !trees.contains_key(t) {
                    if let Ok(st) = self.weights.status(t) {
                        trees.insert(
                            t.clone(),
                            json!({"volume": st.volume, "staged": st.staged, "linked": st.linked}),
                        );
                    }
                }
            }
        }
        let names = |dir: &Path, cap: Option<usize>| -> Vec<String> {
            if !dir.is_dir() {
                return Vec::new();
            }
            let mut v: Vec<String> = std::fs::read_dir(dir)
                .map(|rd| {
                    rd.flatten()
                        .map(|e| e.file_name().to_string_lossy().into_owned())
                        .collect()
                })
                .unwrap_or_default();
            v.sort();
            if let Some(c) = cap {
                v.truncate(c);
            }
            v
        };
        let port = *self.cfg.ports.get("upsampler").unwrap_or(&up::DEFAULT_PORT);
        json!({
            "op": "info",
            "worker_version": VERSION,
            "gpu": g.as_ref().map(|g| g.name.clone()),
            "driver": g.as_ref().map(|g| g.driver.clone()),
            "vram_total_gib": g.as_ref().map(|g| crate::py::round_to(g.memory_total_mib as f64 / 1024.0, 1)),
            "vram_free_gib": g.as_ref().map(|g| crate::py::round_to(g.free_gib(), 1)),
            "compute_cap": g.as_ref().and_then(|g| g.compute_cap.clone()),
            "image": if self.cfg.image.is_empty() { self.env.get("RUNPOD_IMAGE").cloned().unwrap_or_default() } else { self.cfg.image.clone() },
            "catalog_version": self.catalog.version,
            "versions": self.versions(),
            "sidecars": self.sidecars.resident(),
            "upsampler": {"model": up::ALIAS, "tree": up::TREE_NAME, "port": port, "vram_gib": up::default_vram_gib(),
                          "resident": self.sidecars.contains(up::ENGINE), "binary": Path::new(&self.cfg.llama_bin).exists()},
            "trees": trees,
            "staged": names(&self.roots.stage_dir, None),
            "volume": {"path": self.roots.volume.to_string_lossy(), "mounted": self.roots.volume.is_dir(), "weights": names(&self.roots.volume_weights, Some(200))},
            "uptime_s": crate::py::round_to(now_s() - self.started, 1),
            "jobs_done": self.jobs_done.load(Ordering::SeqCst),
            "pod": self.env.get("RUNPOD_POD_ID"),
            "endpoint": self.env.get("RUNPOD_ENDPOINT_ID"),
        })
    }

    /// `h3` / `fastwan` have no --version: the image records their commits
    /// in /opt/studio/{h3,fastwan}.sha; fv-serve prints its build identity.
    pub fn versions(&self) -> Map<String, Value> {
        let mut out = Map::new();
        let first_line = |o: &std::process::Output| -> String {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            let s = if s.is_empty() {
                String::from_utf8_lossy(&o.stderr).trim().to_string()
            } else {
                s
            };
            s.lines().next().unwrap_or("").to_string()
        };
        for name in ["h3", "fastwan"] {
            let sha = self
                .cfg
                .bin_dir
                .parent()
                .unwrap_or(Path::new("/"))
                .join(format!("{name}.sha"));
            if sha.is_file() {
                let s = std::fs::read_to_string(&sha).unwrap_or_default();
                out.insert(
                    name.into(),
                    json!(format!(
                        "{name} @ {}",
                        s.trim().chars().take(12).collect::<String>()
                    )),
                );
                continue;
            }
            let path = self.cfg.bin_dir.join(name);
            match Command::new(&path)
                .arg("--help")
                .env_clear()
                .envs(self.env_for(name))
                .output()
            {
                Ok(o) => {
                    let first = first_line(&o);
                    out.insert(
                        name.into(),
                        json!(if o.status.success() {
                            format!("{name} ({})", first.chars().take(60).collect::<String>())
                        } else {
                            format!(
                                "{name}: exit {}",
                                o.status
                                    .code()
                                    .map(|c| c.to_string())
                                    .unwrap_or_else(|| "None".into())
                            )
                        }),
                    );
                }
                Err(e) => {
                    out.insert(
                        name.into(),
                        json!(format!(
                            "{name}: unavailable ({})",
                            if e.kind() == std::io::ErrorKind::NotFound {
                                "FileNotFoundError"
                            } else {
                                "OSError"
                            }
                        )),
                    );
                }
            }
        }
        match Command::new(&self.cfg.fv_serve_bin)
            .arg("--version")
            .env_clear()
            .envs(self.env_for("fastvideo"))
            .output()
        {
            Ok(o) => {
                let txt = first_line(&o);
                out.insert(
                    "fv-serve".into(),
                    json!(if txt.is_empty() {
                        format!(
                            "exit {}",
                            o.status
                                .code()
                                .map(|c| c.to_string())
                                .unwrap_or_else(|| "None".into())
                        )
                    } else {
                        txt
                    }),
                );
            }
            Err(e) => {
                out.insert(
                    "fv-serve".into(),
                    json!(format!(
                        "unavailable ({})",
                        if e.kind() == std::io::ErrorKind::NotFound {
                            "FileNotFoundError"
                        } else {
                            "OSError"
                        }
                    )),
                );
            }
        }
        out
    }

    pub fn warm(&self, inp: &Value, progress: Option<ProgressFn>) -> JobResult<Value> {
        let engine = inp.get("engine").and_then(|e| e.as_str()).unwrap_or("");
        let model = if self.catalog.engines.contains_key(engine) {
            self.catalog
                .model(engine, &inp.get("model").map(py_str).unwrap_or_default())
        } else {
            None
        };
        let Some(model) = model else {
            return Err(JobError::validate("warm needs a catalog engine and model"));
        };
        let report = Reporter::new(progress, engine, &model.id);
        let t0 = now_s();
        report.force("staging");
        let staged = self.weights.ensure_model(
            &model.trees,
            &model.stage,
            Some(&|d: &str| report.detail("staging", d)),
        )?;
        let probe = json!({"engine": engine, "model": model.id, "prompt": "warm", "width": model.default_size.0, "height": model.default_size.1, "frames": model.frames.default});
        let job = self.catalog.validate(&probe)?;
        let engine_env = self.engine_env();
        let ctx = self.ctx(self.cfg.jobs_dir.join("warm"), &engine_env, &report);
        let mut load_s = 0.0;
        let loading = |_: &str, t: f64| report.detail("loading", &format!("{t:.0} s"));
        if engine == "h3" && !model.cli_only {
            let port = *self.cfg.ports.get("h3").unwrap_or(&7861);
            let key = flags::h3_serve_key(&job, &self.settings, &self.catalog.knobs)?;
            let argv = flags::h3_serve_argv(&job, &self.settings, &self.catalog.knobs, port)?;
            let env = ctx.env_of("h3");
            let make = || {
                Sidecar::new(
                    "h3",
                    &key,
                    argv.clone(),
                    env.clone(),
                    http_info_probe(&format!("http://127.0.0.1:{port}/api/info")),
                )
            };
            load_s = self
                .sidecars
                .ensure("h3", &key, &make, model.vram_gib, Some(&loading))?
                .1;
        } else if engine == "fastwan" && !model.cli_only {
            let port = *self.cfg.ports.get("fastwan").unwrap_or(&7862);
            let key = flags::fastwan_serve_key(&job, &self.settings, &self.catalog.knobs)?;
            let argv = flags::fastwan_serve_argv(&job, &self.settings, &self.catalog.knobs, port)?;
            let env = ctx.env_of("fastwan");
            let make = || {
                Sidecar::new(
                    "fastwan",
                    &key,
                    argv.clone(),
                    env.clone(),
                    http_info_probe(&format!("http://127.0.0.1:{port}/api/info")),
                )
            };
            load_s = self
                .sidecars
                .ensure("fastwan", &key, &make, model.vram_gib, Some(&loading))?
                .1;
        } else if engine == "fastvideo" && !self.cfg.fv_fake {
            let port = *self.cfg.ports.get("fastvideo").unwrap_or(&8010);
            let argv = fastvideo::serve_argv(&ctx, None)?;
            let env = ctx.env_of("fastvideo");
            let make = || {
                Sidecar::new(
                    "fastvideo",
                    fastvideo::KEY,
                    argv.clone(),
                    env.clone(),
                    ping_probe(&format!("http://127.0.0.1:{port}/ping")),
                )
            };
            load_s = self
                .sidecars
                .ensure(
                    "fastvideo",
                    fastvideo::KEY,
                    &make,
                    model.vram_gib,
                    Some(&loading),
                )?
                .1;
        }
        report.force("done");
        Ok(
            json!({"op": "warm", "engine": engine, "model": model.id, "stage_s": staged.stage_s, "fetched": staged.fetched, "staged": staged.staged,
                  "load_s": crate::py::round_to(load_s, 1), "total_s": crate::py::round_to(now_s() - t0, 1), "sidecars": self.sidecars.resident()}),
        )
    }

    // ---- generate -------------------------------------------------------------------

    pub fn generate(
        &self,
        job_id: &str,
        inp: &Value,
        progress: Option<ProgressFn>,
    ) -> JobResult<Value> {
        let t_start = now_s();
        let mut job = self.catalog.validate(inp)?;
        let report = Reporter::new(progress, &job.engine, &job.model.id);
        report.force("queued");
        let job_dir = self.cfg.jobs_dir.join(safe(job_id));
        if job_dir.exists() {
            let _ = std::fs::remove_dir_all(&job_dir);
        }
        std::fs::create_dir_all(&job_dir)?;
        let result = (|| -> JobResult<Value> {
            report.force("staging");
            let staged = self.weights.ensure_model(
                &job.model.trees,
                &job.model.stage,
                Some(&|d: &str| report.detail("staging", d)),
            )?;
            let original_prompt = job.prompt.clone();
            let expanded = self.expand_prompt(&mut job, &job_dir, &report);
            let engine_env = self.engine_env();
            let ctx = self.ctx(job_dir.clone(), &engine_env, &report);
            let mut res = match job.engine.as_str() {
                "h3" => h3::run(&ctx, &job)?,
                "fastwan" => fastwan::run(&ctx, &job)?,
                "fastvideo" => fastvideo::run(&ctx, &job)?,
                other => return Err(JobError::validate(format!("no driver for engine {other}"))),
            };
            res.timings.stage_s = staged.stage_s;
            let out = self.finish(
                job_id,
                &job,
                &mut res,
                &job_dir,
                &report,
                t_start,
                &original_prompt,
                expanded,
            )?;
            self.jobs_done.fetch_add(1, Ordering::SeqCst);
            Ok(out)
        })();
        if !self.cfg.keep_job_dirs {
            let _ = std::fs::remove_dir_all(&job_dir);
        }
        result
    }

    /// The resident `fv-serve` for a fastvideo catalog model (started when
    /// it is not up), for the pod's live streams and segments
    /// (docs/pod-mode.md): its base URL and the load it cost.
    pub fn fastvideo_serve(
        &self,
        model_id: &str,
        progress: Option<ProgressFn>,
    ) -> JobResult<(String, f64)> {
        let model = self.catalog.model("fastvideo", model_id).ok_or_else(|| {
            JobError::validate(format!(
                "unknown fastvideo model {}",
                crate::py::py_str_repr(model_id)
            ))
        })?;
        let report = Reporter::new(progress, "fastvideo", &model.id);
        // The weights first, exactly as the job pipeline does: fv-serve loads
        // its resident model at start, and without the trees that load fails
        // and its /ping answers 503 {"status":"failed"} (2026-11-21, the first
        // segment session on a pod).
        if !self.cfg.fv_fake {
            report.force("staging");
            self.weights.ensure_model(
                &model.trees,
                &model.stage,
                Some(&|d: &str| report.detail("staging", d)),
            )?;
        }
        let engine_env: EngineEnv = ["h3", "fastwan", "fastvideo"]
            .iter()
            .map(|e| (e.to_string(), self.env_for(e)))
            .collect();
        let dir = self.cfg.jobs_dir.join("streams");
        std::fs::create_dir_all(&dir)?;
        let ctx = self.ctx(dir, &engine_env, &report);
        let (base, _sc, load_s) = fastvideo::ensure_serve(&ctx, model)?;
        Ok((base, load_s))
    }

    fn ctx<'a>(
        &'a self,
        job_dir: PathBuf,
        env: &'a EngineEnv,
        report: &'a Reporter<'a>,
    ) -> Ctx<'a> {
        Ctx {
            settings: &self.settings,
            knobs: &self.catalog.knobs,
            sidecars: &self.sidecars,
            ports: &self.cfg.ports,
            env,
            job_dir,
            store: self.store.clone(),
            bucket: self.bucket.clone(),
            log: self.log.clone(),
            report,
            fv_serve_bin: self.cfg.fv_serve_bin.clone(),
            fv_config: self.cfg.fv_config.clone(),
            fv_state_dir: self.cfg.fv_state_dir.clone(),
            fv_weights: self.roots.link_root.to_string_lossy().into_owned(),
            fv_fake: self.cfg.fv_fake,
            models: &self.catalog.models,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn finish(
        &self,
        job_id: &str,
        job: &Job,
        res: &mut RunResult,
        job_dir: &Path,
        report: &Reporter,
        t_start: f64,
        original_prompt: &str,
        expanded_prompt: Option<String>,
    ) -> JobResult<Value> {
        // ---- mux
        report.force("muxing");
        let mp4 = job_dir.join("video.mp4");
        let mut audio_wav = res.audio_wav.clone();
        let thumb = (|| -> Result<PathBuf, String> {
            if let Some(src) = &res.video_mp4 {
                std::fs::rename(src, &mp4)
                    .or_else(|_| std::fs::copy(src, &mp4).map(|_| ()))
                    .map_err(|e| e.to_string())?;
                if job.engine == "h3" && job.model.audio {
                    audio_wav = mux::extract_wav(&mp4, &job_dir.join("audio.wav"), None);
                }
            } else {
                let y4m = res
                    .video_y4m
                    .as_ref()
                    .ok_or_else(|| "no video".to_string())?;
                let r = mux::mux(y4m, audio_wav.as_deref(), &mp4, 18, "medium", None)?;
                res.timings.mux_s = r.seconds;
            }
            mux::thumbnail(
                &mp4,
                &job_dir.join("thumb.jpg"),
                res.frames,
                res.fps,
                None,
                3,
            )
        })()
        .map_err(|e| JobError::new("mux", e))?;
        // ---- upload
        report.force("uploading");
        let expires = now_s() as i64 + self.cfg.url_ttl_s as i64;
        res.timings.total_s = now_s() - t_start;
        let g = gpu::query(0);
        let audio_present = audio_wav.as_ref().map(|a| a.is_file()).unwrap_or(false);
        let audio = if job.engine == "h3" {
            audio_present
        } else {
            res.extra
                .get("audio")
                .map(crate::py::truthy)
                .unwrap_or(false)
                || audio_wav.is_some()
        };
        let mut output = json!({
            "engine": job.engine, "model": job.model.id, "task": job.task, "seed": res.seed,
            "width": res.width, "height": res.height, "frames": res.frames, "fps": res.fps,
            "audio": audio,
            "upscale": res.upscaled.map(|(f, w, h)| json!({"factor": f, "width": w, "height": h})),
            "timings": res.timings.to_json(),
            "command": res.command,
            "worker": {"gpu": g.as_ref().map(|g| g.name.clone()), "driver": g.as_ref().map(|g| g.driver.clone()),
                       "image": if self.cfg.image.is_empty() { self.env.get("RUNPOD_IMAGE").cloned().unwrap_or_default() } else { self.cfg.image.clone() }},
            "expires_at": iso_utc(expires),
            "expanded_prompt": expanded_prompt,
            "client": job.client,
        });
        let (Some(store), Some(bucket)) = (
            &self.store,
            self.bucket.as_deref().filter(|b| !b.is_empty()),
        ) else {
            return Err(JobError::new(
                "upload",
                "no object store configured (FV_R2_* / STUDIO_R2_LOCAL_DIR)",
            ));
        };
        let prefix = format!("jobs/{}/", safe(job_id));
        let up_err = |e: anyhow::Error| match e.downcast::<JobError>() {
            Ok(j) => j,
            Err(e) => JobError::new("upload", format!("{e:#}")),
        };
        store
            .put_file(
                bucket,
                &format!("{prefix}video.mp4"),
                &mp4,
                Some("video/mp4"),
            )
            .map_err(up_err)?;
        let mut urls = json!({"video_url": null, "audio_url": null, "thumbnail_url": null});
        urls["video_url"] =
            json!(store.presign_get(bucket, &format!("{prefix}video.mp4"), self.cfg.url_ttl_s));
        if job.engine == "h3" && audio_present {
            let a = audio_wav.as_ref().unwrap();
            store
                .put_file(bucket, &format!("{prefix}audio.wav"), a, Some("audio/wav"))
                .map_err(up_err)?;
            urls["audio_url"] =
                json!(store.presign_get(bucket, &format!("{prefix}audio.wav"), self.cfg.url_ttl_s));
        }
        store
            .put_file(
                bucket,
                &format!("{prefix}thumb.jpg"),
                &thumb,
                Some("image/jpeg"),
            )
            .map_err(up_err)?;
        urls["thumbnail_url"] =
            json!(store.presign_get(bucket, &format!("{prefix}thumb.jpg"), self.cfg.url_ttl_s));
        for (k, v) in urls.as_object().unwrap() {
            output[k] = v.clone();
        }
        let meta = job_dir.join("job.json");
        std::fs::write(&meta, dumps(&json!({"id": job_id, "input": redact_input(job, Some(original_prompt)), "output": output}), Some(1), true)).map_err(|e| JobError::new("upload", format!("OSError: {e}")))?;
        store
            .put_file(
                bucket,
                &format!("{prefix}job.json"),
                &meta,
                Some("application/json"),
            )
            .map_err(up_err)?;
        report.force("done");
        Ok(output)
    }
}

pub fn safe(s: &str) -> String {
    let out: String = s
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || "-_.".contains(c) {
                c
            } else {
                '_'
            }
        })
        .take(120)
        .collect();
    if out.is_empty() {
        "job".into()
    } else {
        out
    }
}

/// The validated request without inline media bytes (the prompt as sent,
/// before any expansion).
pub fn redact_input(job: &Job, original_prompt: Option<&str>) -> Value {
    let one = |m: &crate::catalog::Media| json!({"kind": m.kind, "value": if m.kind != "b64" { m.value.clone() } else { format!("<{} bytes>", m.inline_bytes()) }});
    let mut inputs = Map::new();
    for k in &job.inputs.order {
        if k == "references" {
            inputs.insert(
                k.clone(),
                Value::Array(job.inputs.references.iter().map(one).collect()),
            );
        } else if let Some(m) = job.inputs.get(k) {
            inputs.insert(k.clone(), one(m));
        }
    }
    json!({"engine": job.engine, "model": job.model.id, "task": job.task, "prompt": original_prompt.unwrap_or(&job.prompt),
           "negative_prompt": job.negative_prompt, "width": job.width, "height": job.height, "frames": job.frames,
           "seed": job.seed, "options": job.explicit, "upscale": job.upscale, "inputs": inputs, "client": job.client})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_ids_and_config_defaults() {
        assert_eq!(safe("h3-1"), "h3-1");
        assert_eq!(safe("a/b c"), "a_b_c");
        assert_eq!(safe(""), "job");
        let e: EnvMap = [
            ("STUDIO_H3_PORT", "7881"),
            ("STUDIO_SYNTHETIC", "yes"),
            ("STUDIO_BIN_DIR", "/b"),
            ("STUDIO_CATALOG", "/c.json"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        let c = Config::from_env(&e);
        assert_eq!(
            (
                c.ports["h3"],
                c.ports["fastwan"],
                c.ports["fastvideo"],
                c.ports["upsampler"]
            ),
            (7881, 7862, 8010, 18080)
        );
        assert!(
            c.synthetic
                && !c.fv_fake
                && c.bin_dir == Path::new("/b")
                && c.catalog_path == Path::new("/c.json")
        );
        assert_eq!(
            c.esrgan.as_deref(),
            Some("/workspace/w/h3-rs/assets/realesr-animevideov3.safetensors")
        );
        assert_eq!(c.llama_bin, up::LLAMA_BIN);
    }

    #[test]
    fn fv_serve_gets_a_loopback_public_base_on_its_own_port() {
        let dir = crate::tempdir("worker-fvbase");
        let e: EnvMap = [
            ("STUDIO_R2_LOCAL_DIR", dir.join("r2").to_str().unwrap()),
            ("STUDIO_VOLUME", dir.join("vol").to_str().unwrap()),
            ("STUDIO_JOBS_DIR", dir.join("jobs").to_str().unwrap()),
            ("STUDIO_FV_PORT", "8777"),
            (
                "STUDIO_CATALOG",
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../catalog/catalog.json")
                    .to_str()
                    .unwrap(),
            ),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        let w = Worker::from_env_map(&e, crate::null_log()).unwrap();
        let fv = w.env_for("fastvideo");
        assert_eq!(fv["FV_PUBLIC_BASE_URL"], "http://127.0.0.1:8777");
        // h3 and fastwan serve their own files; the knob is fv-serve's only.
        assert!(!w.env_for("h3").contains_key("FV_PUBLIC_BASE_URL"));
    }

    #[test]
    fn fastvideo_serve_stages_the_models_weights_before_starting_fv_serve() {
        // A stream or a segment session starts fv-serve itself; it must stage
        // the model's trees first, or fv-serve's resident load fails and its
        // /ping answers 503 {"status":"failed"}. Here no tree is reachable
        // (empty local R2 dir, empty volume), so staging must be what fails.
        let dir = crate::tempdir("worker-fvserve");
        let e: EnvMap = [
            ("STUDIO_R2_LOCAL_DIR", dir.join("r2").to_str().unwrap()),
            ("STUDIO_VOLUME", dir.join("vol").to_str().unwrap()),
            ("STUDIO_JOBS_DIR", dir.join("jobs").to_str().unwrap()),
            (
                "STUDIO_CATALOG",
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../catalog/catalog.json")
                    .to_str()
                    .unwrap(),
            ),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        let w = Worker::from_env_map(&e, crate::null_log()).unwrap();
        let err = w
            .fastvideo_serve("fasth3-4step-vsa-480p-taeh3", None)
            .unwrap_err();
        assert_eq!(err.where_, "stage", "{}", err.message);
        assert!(
            w.fastvideo_serve("nope", None).unwrap_err().where_ == "validate",
            "an unknown model is still a validation error"
        );
    }

    #[test]
    fn handle_shapes_validation_errors_and_unknown_ops() {
        let dir = crate::tempdir("worker");
        let e: EnvMap = [
            ("STUDIO_R2_LOCAL_DIR", dir.join("r2").to_str().unwrap()),
            ("STUDIO_VOLUME", dir.join("vol").to_str().unwrap()),
            ("STUDIO_JOBS_DIR", dir.join("jobs").to_str().unwrap()),
            (
                "STUDIO_CATALOG",
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../catalog/catalog.json")
                    .to_str()
                    .unwrap(),
            ),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        let w = Worker::from_env_map(&e, crate::null_log()).unwrap();
        assert_eq!(
            w.handle(&json!({"id": "x", "input": {"op": "nope"}}), None),
            json!({"error": "unknown op 'nope'", "where": "validate"})
        );
        assert_eq!(
            w.handle(&json!({"id": "x", "input": 5}), None),
            json!({"error": "input must be an object", "where": "validate"})
        );
        let out = w.handle(&json!({"id": "bad-1", "input": {"engine": "h3", "model": "fasth3-v2", "prompt": "x", "frames": 100}}), None);
        assert!(out["where"] == "validate" && out["error"].as_str().unwrap().contains("frames"));
        let out = w.handle(
            &json!({"id": "w", "input": {"op": "warm", "engine": "nope"}}),
            None,
        );
        assert_eq!(out["error"], "warm needs a catalog engine and model");
        let info = w.handle(&json!({"id": "i", "input": {"op": "info"}}), None);
        assert_eq!(info["op"], "info");
        assert_eq!(info["catalog_version"], json!(1));
        assert!(info["versions"]["h3"]
            .as_str()
            .unwrap()
            .starts_with("h3: unavailable"));
        assert_eq!(info["upsampler"]["model"], up::ALIAS);
        assert_eq!(info["upsampler"]["vram_gib"], json!(6.5));
        assert!(info["trees"]["weights/h3-8step"]["volume"] == false);
        let r2 = w.handle(&json!({"id": "r", "input": {"op": "r2"}}), None);
        assert!(
            r2["write"] == "ok"
                && r2["read_weights"]
                    .as_str()
                    .unwrap()
                    .starts_with("ok (0 manifests)")
        );
        assert_eq!(r2["access_key_id_len"], json!(0));
        let chk = w.handle(
            &json!({"id": "c", "input": {"op": "check", "engines": ["nope"]}}),
            None,
        );
        assert_eq!(
            chk["results"]["nope"]["error"],
            "no checker for this engine"
        );
        let j = w.catalog.validate(&json!({"engine": "h3", "model": "fasth3-v2", "prompt": "a fox", "inputs": {}, "options": {"pv8": true}, "client": {"job": "c1"}})).unwrap();
        let r = redact_input(&j, Some("orig"));
        assert_eq!(r["prompt"], "orig");
        assert_eq!(r["options"], json!({"h3.pv8": true}));
        assert_eq!(r["client"], json!({"job": "c1"}));
    }
}
