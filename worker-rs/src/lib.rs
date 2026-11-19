//! fastvideo-studio worker: the RunPod handler's library, a module-for-module
//! port of `worker/fvstudio` (Python, the reference until it is removed).
//!
//! `rp` speaks RunPod's serverless protocol (no SDK), `r2` signs R2's S3 API
//! (SigV4, no AWS SDK). The contract is `docs/jobs.md` and
//! `catalog/catalog.json`; `docs/design.md` has the reasoning.

pub mod catalog;
pub mod engines;
pub mod errors;
pub mod flags;
pub mod frames;
pub mod gpu;
pub mod httpc;
pub mod media;
pub mod mux;
pub mod progress;
pub mod py;
pub mod r2;
pub mod rp;
pub mod sidecar;
pub mod upsampler;
pub mod weights;
pub mod worker;

/// `fvstudio.__version__`.
pub const VERSION: &str = "0.1.0";

/// A log sink shared by the worker, the sidecars and the resolver.
pub type Log = std::sync::Arc<dyn Fn(&str) + Send + Sync>;

/// The default log: one line on stderr.
pub fn stderr_log() -> Log {
    std::sync::Arc::new(|s: &str| eprintln!("{s}"))
}

/// A log that drops everything (tests).
pub fn null_log() -> Log {
    std::sync::Arc::new(|_: &str| {})
}

/// Seconds since the Unix epoch, as `time.time()`.
pub fn now_s() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// An environment snapshot (`os.environ`), so the worker can be configured
/// from a map in tests without touching the process environment.
pub type EnvMap = std::collections::HashMap<String, String>;

pub fn process_env() -> EnvMap {
    std::env::vars().collect()
}

/// A fresh scratch directory under the system temp dir (tests and fakes).
pub fn tempdir(tag: &str) -> std::path::PathBuf {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let p = std::env::temp_dir().join(format!(
        "fvstudio-{tag}-{}-{}-{n}",
        std::process::id(),
        now_s().to_bits()
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}
