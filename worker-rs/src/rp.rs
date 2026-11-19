//! RunPod's serverless worker protocol with a blocking HTTP client (no SDK).
//!
//! The protocol is the one runpod-python 1.7.13 speaks (`serverless/modules/
//! rp_job.py`, `rp_http.py`, `rp_ping.py`, `rp_progress.py`), as h3-rs's
//! `h3 worker` does:
//!
//! ```text
//! take      GET  $RUNPOD_WEBHOOK_GET_JOB      ($ID → worker id)  &job_in_progress=0|1
//!             long poll; 204 / 400: no job; 429: wait 5 s; 200: {"id", "input"} or a list
//! progress  POST $RUNPOD_WEBHOOK_POST_OUTPUT  ($RUNPOD_POD_ID → worker id, $ID → job id) &isStream=false
//!             {"status": "IN_PROGRESS", "output": <progress object>}
//! done      POST $RUNPOD_WEBHOOK_POST_OUTPUT  (same URL)
//!             {"output": <handler output>} and / or {"error": <handler's error>}
//!             (no `status`: the platform marks the job COMPLETED or FAILED by the
//!             presence of `error`); 3 attempts, Fibonacci back-off 1, 1, 2 s
//! ping      GET  $RUNPOD_WEBHOOK_PING         ($RUNPOD_POD_ID → worker id) ?job_id=<running job>&runpod_version=…
//!             every $RUNPOD_PING_INTERVAL ms (default 10000)
//! headers   Authorization: $RUNPOD_AI_API_KEY (raw), X-Request-ID: <job id> on POSTs,
//!           Content-Type: application/x-www-form-urlencoded (the body is JSON text)
//! ```
//!
//! The URL templates are opaque: `$ID` / `$RUNPOD_POD_ID` are substituted and
//! query parameters appended with `?` or `&` as needed (they come with
//! `?gpu=…` already). A handler's object output has its `error` key lifted out
//! beside `output` (`run_job`, like the SDK), so a `{"error", "where"}` result
//! is FAILED with that error and keeps `output.where`. Concurrency is 1: one
//! GPU, one job. `RUNPOD_WEBHOOK_GET_JOB` absent means local mode (`run_local`
//! runs one job from a file). SIGTERM / SIGINT stop the loop after the
//! current job.

use crate::py::{dumps, quote};
use crate::{EnvMap, Log};
use serde_json::{json, Map, Value};
use std::io::Read;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::time::Duration;

pub const VERSION: &str = concat!("fvstudio-worker/", "0.1.0");

pub fn stderr(msg: &str) {
    eprintln!("{msg}");
}

pub fn join_query(url: &str, q: &str) -> String {
    format!("{url}{}{q}", if url.contains('?') { '&' } else { '?' })
}

fn qenc(s: &str) -> String {
    quote(s, "")
}

/// The variables RunPod injects into a serverless worker.
#[derive(Debug, Clone, PartialEq)]
pub struct RunpodEnv {
    pub get_job: String,
    pub post_output: String,
    pub ping: Option<String>,
    pub api_key: String,
    pub worker_id: String,
    pub ping_interval_s: f64,
    pub hostname: Option<String>,
}

impl RunpodEnv {
    /// `None` when `RUNPOD_WEBHOOK_GET_JOB` is unset (the SDK's local mode).
    pub fn from_env(e: &EnvMap) -> Result<Option<RunpodEnv>, String> {
        let get = |k: &str| e.get(k).map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
        let Some(get_job) = get("RUNPOD_WEBHOOK_GET_JOB") else { return Ok(None) };
        let post_output = get("RUNPOD_WEBHOOK_POST_OUTPUT").ok_or_else(|| "RUNPOD_WEBHOOK_POST_OUTPUT is not set".to_string())?;
        let ping_ms: i64 = get("RUNPOD_PING_INTERVAL").and_then(|v| v.parse().ok()).unwrap_or(10000);
        Ok(Some(RunpodEnv {
            get_job,
            post_output,
            ping: get("RUNPOD_WEBHOOK_PING"),
            api_key: get("RUNPOD_AI_API_KEY").unwrap_or_default(),
            worker_id: get("RUNPOD_POD_ID").unwrap_or_else(|| format!("local-{}", std::process::id())),
            ping_interval_s: ping_ms.max(100) as f64 / 1000.0,
            hostname: get("RUNPOD_POD_HOSTNAME").or_else(|| get("HOSTNAME")),
        }))
    }

    /// `GET` job-take (`$ID` is the worker here).
    pub fn take_url(&self, in_progress: bool) -> String {
        let base = self.get_job.replace("$RUNPOD_POD_ID", &self.worker_id).replace("$ID", &self.worker_id);
        join_query(&base, &format!("job_in_progress={}", if in_progress { 1 } else { 0 }))
    }

    /// `POST` job-done, for progress and the result alike (`isStream=false`).
    pub fn done_url(&self, job_id: &str) -> String {
        let base = self.post_output.replace("$RUNPOD_POD_ID", &self.worker_id).replace("$ID", job_id);
        join_query(&base, "isStream=false")
    }

    pub fn ping_url(&self, job_id: Option<&str>) -> Option<String> {
        let ping = self.ping.as_ref()?;
        let mut u = ping.replace("$RUNPOD_POD_ID", &self.worker_id);
        if let Some(j) = job_id.filter(|j| !j.is_empty()) {
            u = join_query(&u, &format!("job_id={}", qenc(j)));
        }
        Some(join_query(&u, &format!("runpod_version={}", qenc(VERSION))))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TakeError {
    #[error("too many requests")]
    TooManyRequests,
    /// A transport error or an unexpected status (the Python loop's
    /// `URLError` / `OSError` / `RuntimeError`).
    #[error("{0}")]
    Failed(String),
}

pub type Sleep = Arc<dyn Fn(f64) + Send + Sync>;

/// The HTTP side: take, progress, done, ping. Progress goes out from a
/// sender thread in order (the job is not held up by the platform);
/// `flush_progress` drains it before the result is posted.
pub struct RunpodClient {
    pub env: RunpodEnv,
    pub log: Log,
    pub sleep: Sleep,
    pub take_timeout_s: f64,
    pub post_timeout_s: f64,
    pub done_attempts: usize,
    pub done_backoff_s: Vec<f64>,
    current_job: Mutex<Option<String>>,
    progress_tx: Mutex<Option<mpsc::Sender<(String, Value)>>>,
    pending: Arc<(Mutex<usize>, Condvar)>,
    pub progress_sent: AtomicUsize,
    pub progress_failed: AtomicUsize,
    agent: ureq::Agent,
}

impl RunpodClient {
    pub fn new(env: RunpodEnv, log: Log, sleep: Sleep) -> Arc<RunpodClient> {
        Arc::new(RunpodClient {
            env,
            log,
            sleep,
            take_timeout_s: 90.0,  // the SDK's job-take long poll
            post_timeout_s: 60.0,
            done_attempts: 3,
            done_backoff_s: vec![1.0, 1.0, 2.0], // Fibonacci, like the SDK's aiohttp_retry
            current_job: Mutex::new(None),
            progress_tx: Mutex::new(None),
            pending: Arc::new((Mutex::new(0), Condvar::new())),
            progress_sent: AtomicUsize::new(0),
            progress_failed: AtomicUsize::new(0),
            agent: ureq::AgentBuilder::new().timeout_connect(Duration::from_secs(15)).build(),
        })
    }

    pub fn with_defaults(env: RunpodEnv) -> Arc<RunpodClient> {
        Self::new(env, crate::stderr_log(), Arc::new(|s| std::thread::sleep(Duration::from_secs_f64(s))))
    }

    pub fn current_job(&self) -> Option<String> {
        self.current_job.lock().unwrap().clone()
    }

    pub fn set_current_job(&self, id: Option<String>) {
        *self.current_job.lock().unwrap() = id;
    }

    // ---- HTTP -------------------------------------------------------------------

    fn request(&self, method: &str, url: &str, body: Option<&Value>, job_id: Option<&str>, timeout: Option<f64>) -> Result<(u16, Vec<u8>), String> {
        let mut req = self.agent.request(method, url).set("Authorization", &self.env.api_key).timeout(Duration::from_secs_f64(timeout.unwrap_or(self.post_timeout_s)));
        if let Some(j) = job_id {
            req = req.set("X-Request-ID", j);
        }
        let r = match body {
            Some(b) => {
                let data = dumps(b, None, false);
                req.set("Content-Type", "application/x-www-form-urlencoded").set("charset", "utf-8").send_string(&data)
            }
            None => req.call(),
        };
        match r {
            Ok(resp) => {
                let status = resp.status();
                let mut v = Vec::new();
                resp.into_reader().take(64 << 20).read_to_end(&mut v).map_err(|e| e.to_string())?;
                Ok((status, v))
            }
            Err(ureq::Error::Status(code, resp)) => {
                let mut v = Vec::new();
                let _ = resp.into_reader().take(1 << 20).read_to_end(&mut v);
                Ok((code, v))
            }
            Err(ureq::Error::Transport(t)) => Err(t.to_string()),
        }
    }

    /// One job-take: the jobs it returned (`[]` on 204 / 400);
    /// `TooManyRequests` on 429, `Failed` on any other status or a transport error.
    pub fn take(&self) -> Result<Vec<Value>, TakeError> {
        let url = self.env.take_url(self.current_job().is_some());
        let (status, body) = self.request("GET", &url, None, None, Some(self.take_timeout_s)).map_err(TakeError::Failed)?;
        if status == 204 || status == 400 {
            // 400 is expected under FlashBoot
            return Ok(Vec::new());
        }
        if status == 429 {
            return Err(TakeError::TooManyRequests);
        }
        if !(200..300).contains(&status) {
            return Err(TakeError::Failed(format!("job-take: HTTP {status}: {}", crate::py::py_str_repr(&String::from_utf8_lossy(&body[..body.len().min(200)])))));
        }
        if body.iter().all(|b| b.is_ascii_whitespace()) {
            return Ok(Vec::new());
        }
        let v: Value = serde_json::from_slice(&body).map_err(|e| TakeError::Failed(format!("job-take: not JSON: {e}")))?;
        let items = match v {
            Value::Array(a) => a,
            other => vec![other],
        };
        let mut jobs = Vec::new();
        for j in items {
            if j.is_object() && j.get("id").is_some() && j.get("input").is_some() {
                jobs.push(j);
            } else {
                let s = dumps(&j, None, false);
                (self.log)(&format!("worker: job-take returned a job without id / input: {}", s.chars().take(200).collect::<String>()));
            }
        }
        Ok(jobs)
    }

    /// The result, with retries; false when it was never accepted.
    pub fn post_done(&self, job_id: &str, body: &Value) -> bool {
        let url = self.env.done_url(job_id);
        for attempt in 0..self.done_attempts {
            let err = match self.request("POST", &url, Some(body), Some(job_id), None) {
                Ok((status, _)) if (200..300).contains(&status) => return true,
                Ok((status, text)) => format!("HTTP {status}: {}", crate::py::py_str_repr(&String::from_utf8_lossy(&text[..text.len().min(200)]))),
                Err(e) => e,
            };
            (self.log)(&format!("worker: job {job_id}: posting the result failed (attempt {}/{}): {err}", attempt + 1, self.done_attempts));
            if attempt + 1 < self.done_attempts {
                (self.sleep)(self.done_backoff_s[attempt.min(self.done_backoff_s.len() - 1)]);
            }
        }
        false
    }

    pub fn ping(&self) {
        let Some(url) = self.env.ping_url(self.current_job().as_deref()) else { return };
        match self.request("GET", &url, None, None, Some(self.env.ping_interval_s * 2.0)) {
            Ok((status, _)) if (200..300).contains(&status) => {}
            Ok((status, _)) => (self.log)(&format!("worker: ping: HTTP {status}")),
            Err(e) => (self.log)(&format!("worker: ping failed: {e}")),
        }
    }

    // ---- progress ---------------------------------------------------------------

    /// `runpod.serverless.progress_update`'s shape: the object goes out
    /// verbatim as `{"status": "IN_PROGRESS", "output": data}`.
    pub fn progress_update(self: &Arc<Self>, job_id: &str, data: &Value) {
        let mut tx = self.progress_tx.lock().unwrap();
        if tx.is_none() {
            let (s, r) = mpsc::channel::<(String, Value)>();
            let me = Arc::clone(self);
            std::thread::Builder::new().name("rp-progress".into()).spawn(move || me.send_progress(r)).expect("progress thread");
            *tx = Some(s);
        }
        {
            let (count, _) = &*self.pending;
            *count.lock().unwrap() += 1;
        }
        let _ = tx.as_ref().unwrap().send((job_id.to_string(), data.clone()));
    }

    fn send_progress(&self, rx: mpsc::Receiver<(String, Value)>) {
        while let Ok((job_id, data)) = rx.recv() {
            let body = json!({"status": "IN_PROGRESS", "output": data});
            match self.request("POST", &self.env.done_url(&job_id), Some(&body), Some(&job_id), None) {
                Ok((status, _)) if (200..300).contains(&status) => {
                    self.progress_sent.fetch_add(1, Ordering::SeqCst);
                }
                Ok((status, text)) => {
                    self.progress_failed.fetch_add(1, Ordering::SeqCst);
                    (self.log)(&format!("worker: job {job_id}: progress update: HTTP {status}: {}", crate::py::py_str_repr(&String::from_utf8_lossy(&text[..text.len().min(200)]))));
                }
                Err(e) => {
                    self.progress_failed.fetch_add(1, Ordering::SeqCst);
                    (self.log)(&format!("worker: job {job_id}: progress update failed: {e}"));
                }
            }
            let (count, cv) = &*self.pending;
            let mut n = count.lock().unwrap();
            *n -= 1;
            cv.notify_all();
        }
    }

    /// Wait until every queued progress update has been posted.
    pub fn flush_progress(&self) {
        let (count, cv) = &*self.pending;
        let mut n = count.lock().unwrap();
        while *n > 0 {
            n = cv.wait(n).unwrap();
        }
    }
}

// ---- jobs -------------------------------------------------------------------------

pub type Handler<'a> = &'a (dyn Fn(&Value) -> Value + Sync);

/// Runs `handler(job)` and shapes the job-done body like the SDK's
/// `rp_job.run_job`: an object output has `error` (and `refresh_worker`) lifted
/// out, an empty output is dropped, a panic is `{"error": <json report>}`.
pub fn run_job(handler: Handler, job: &Value, env: Option<&RunpodEnv>) -> Value {
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handler(job)));
    match res {
        Ok(out) => {
            let mut result = Map::new();
            if let Value::Object(mut o) = out {
                let error = o.shift_remove("error");
                let refresh = o.shift_remove("refresh_worker");
                result.insert("output".into(), Value::Object(o));
                if let Some(e) = error.filter(crate::py::truthy) {
                    result.insert("error".into(), e);
                }
                if refresh.map(|r| crate::py::truthy(&r)).unwrap_or(false) {
                    result.insert("stopPod".into(), Value::Bool(true));
                }
            } else {
                result.insert("output".into(), out);
            }
            if result.get("output").map(|o| o.as_object().map(|m| m.is_empty()).unwrap_or(false)).unwrap_or(false) {
                result.shift_remove("output");
            }
            Value::Object(result)
        }
        Err(p) => {
            let msg = p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_else(|| "panic".into());
            let bt = std::backtrace::Backtrace::force_capture();
            let report = json!({
                "error_type": "<class 'panic'>",
                "error_message": msg,
                "error_traceback": format!("Traceback (most recent call last):\n{bt}"),
                "hostname": env.and_then(|e| e.hostname.clone()).or_else(|| std::env::var("RUNPOD_POD_HOSTNAME").ok()).unwrap_or_else(|| "unknown".into()),
                "worker_id": env.map(|e| e.worker_id.clone()).or_else(|| std::env::var("RUNPOD_POD_ID").ok()).unwrap_or_else(|| "unknown".into()),
                "runpod_version": VERSION,
            });
            json!({"error": dumps(&report, None, true)})
        }
    }
}

static SIGNAL_SEEN: AtomicI32 = AtomicI32::new(0);

extern "C" fn on_signal(signum: i32) {
    SIGNAL_SEEN.store(signum, Ordering::SeqCst);
}

extern "C" {
    fn signal(signum: i32, handler: extern "C" fn(i32)) -> usize;
}

const SIGINT: i32 = 2;
const SIGTERM: i32 = 15;

/// SIGTERM / SIGINT set `stop`: the loop ends after the current job. A
/// watcher thread turns the flag into the log line (a handler may not log).
pub fn install_signal_handlers(stop: Arc<AtomicBool>, log: Log) {
    // SAFETY: `on_signal` only stores an atomic; `signal` is the C library's.
    unsafe {
        signal(SIGTERM, on_signal);
        signal(SIGINT, on_signal);
    }
    std::thread::Builder::new()
        .name("rp-signals".into())
        .spawn(move || loop {
            let s = SIGNAL_SEEN.swap(0, Ordering::SeqCst);
            if s != 0 {
                log(&format!("worker: {} received: stopping after the current job", if s == SIGTERM { "SIGTERM" } else { "SIGINT" }));
                stop.store(true, Ordering::SeqCst);
            }
            std::thread::sleep(Duration::from_millis(50));
        })
        .expect("signal thread");
}

/// The worker loop: take, run, post, until `stop` is set (or `max_jobs`
/// ran). Returns the number of jobs run.
pub fn serve(handler: Handler, client: &Arc<RunpodClient>, stop: Arc<AtomicBool>, max_jobs: Option<usize>, idle_s: f64) -> usize {
    let log = client.log.clone();
    let env = client.env.clone();
    if env.ping.is_some() {
        let (c, stop) = (Arc::clone(client), Arc::clone(&stop));
        std::thread::Builder::new()
            .name("rp-ping".into())
            .spawn(move || {
                while !stop.load(Ordering::SeqCst) {
                    c.ping();
                    let mut waited = 0.0;
                    while waited < c.env.ping_interval_s && !stop.load(Ordering::SeqCst) {
                        std::thread::sleep(Duration::from_millis(20));
                        waited += 0.02;
                    }
                }
            })
            .expect("ping thread");
    }
    log(&format!("worker: {VERSION} as {}: waiting for jobs", env.worker_id));
    let mut done = 0;
    let mut backoff = 1.0f64;
    while !stop.load(Ordering::SeqCst) {
        let jobs = match client.take() {
            Ok(j) => {
                backoff = 1.0;
                j
            }
            Err(TakeError::TooManyRequests) => {
                log("worker: job-take: 429, waiting 5 s");
                (client.sleep)(5.0);
                continue;
            }
            Err(TakeError::Failed(e)) => {
                log(&format!("worker: job-take failed: {e}; retrying in {backoff:.0} s"));
                (client.sleep)(backoff);
                backoff = (backoff * 2.0).min(30.0);
                continue;
            }
        };
        if jobs.is_empty() {
            if idle_s > 0.0 {
                (client.sleep)(idle_s);
            }
            continue;
        }
        for job in jobs {
            let job_id = crate::py::py_str(&job["id"]);
            client.set_current_job(Some(job_id.clone()));
            log(&format!("worker: job {job_id} started"));
            let t0 = std::time::Instant::now();
            let result = run_job(handler, &job, Some(&env));
            client.flush_progress();
            let sent = client.post_done(&job_id, &result);
            client.set_current_job(None);
            done += 1;
            log(&format!(
                "worker: job {job_id} {} in {:.1} s ({})",
                if result.get("error").is_some() { "failed" } else { "done" },
                t0.elapsed().as_secs_f64(),
                if sent { "result sent" } else { "result NOT accepted: the platform will time the job out" }
            ));
            if max_jobs.map(|m| done >= m).unwrap_or(false) {
                stop.store(true, Ordering::SeqCst);
                break;
            }
        }
    }
    log(&format!("worker: stopped after {done} job(s)"));
    done
}

/// Local mode (no `RUNPOD_WEBHOOK_GET_JOB`): one job from `test_input`
/// (a file holding `{"input": …}` or the input itself; the SDK's
/// `test_input.json` in the working directory when none is given), the
/// result printed as JSON to `out`. Exit code 1 when it failed, 2 without a file.
pub fn run_local(handler: Handler, test_input: Option<&str>, log: &Log, out: &mut dyn std::io::Write) -> i32 {
    let path = test_input.unwrap_or("test_input.json");
    if !std::path::Path::new(path).is_file() {
        log(&format!("worker: local mode: {path} not found (RUNPOD_WEBHOOK_GET_JOB is unset; pass --test-input FILE)"));
        return 2;
    }
    let doc: Value = match std::fs::read_to_string(path).map_err(|e| e.to_string()).and_then(|s| serde_json::from_str(&s).map_err(|e| e.to_string())) {
        Ok(d) => d,
        Err(e) => {
            log(&format!("worker: local mode: {path}: {e}"));
            return 2;
        }
    };
    let mut job = json!({"id": "local_test", "input": doc.get("input").cloned().unwrap_or_else(|| doc.clone())});
    if let Some(id) = doc.get("id") {
        job["id"] = Value::String(crate::py::py_str(id));
    }
    log(&format!("worker: local mode: running {path} as job {}", crate::py::py_str(&job["id"])));
    let result = run_job(handler, &job, None);
    let _ = writeln!(out, "{}", dumps(&result, Some(1), false));
    let _ = out.flush();
    if result.get("error").is_some() { 1 } else { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(map: &[(&str, &str)]) -> Result<Option<RunpodEnv>, String> {
        RunpodEnv::from_env(&map.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect())
    }

    #[test]
    fn url_templates() {
        let base = "http://127.0.0.1:1/v2/ep1";
        let e = env(&[
            ("RUNPOD_WEBHOOK_GET_JOB", &format!("{base}/job-take/$ID?gpu=NVIDIA+H200")),
            ("RUNPOD_WEBHOOK_POST_OUTPUT", &format!("{base}/job-done/$RUNPOD_POD_ID/$ID?gpu=NVIDIA+H200")),
            ("RUNPOD_WEBHOOK_PING", &format!("{base}/ping/$RUNPOD_POD_ID")),
            ("RUNPOD_PING_INTERVAL", "100"),
            ("RUNPOD_AI_API_KEY", "secret-key"),
            ("RUNPOD_POD_ID", "w42"),
            ("RUNPOD_POD_HOSTNAME", "host-1"),
        ])
        .unwrap()
        .unwrap();
        assert_eq!(e.take_url(false), format!("{base}/job-take/w42?gpu=NVIDIA+H200&job_in_progress=0"));
        assert_eq!(e.take_url(true), format!("{base}/job-take/w42?gpu=NVIDIA+H200&job_in_progress=1"));
        assert_eq!(e.done_url("j1"), format!("{base}/job-done/w42/j1?gpu=NVIDIA+H200&isStream=false"));
        assert_eq!(e.ping_url(Some("a/b")), Some(format!("{base}/ping/w42?job_id=a%2Fb&runpod_version={}", VERSION.replace('/', "%2F"))));
        assert_eq!(e.ping_url(None), Some(format!("{base}/ping/w42?runpod_version={}", VERSION.replace('/', "%2F"))));
        assert_eq!((e.ping_interval_s, e.api_key.as_str(), e.hostname.as_deref()), (0.1, "secret-key", Some("host-1")));
    }

    #[test]
    fn env_without_query_and_defaults() {
        let e = env(&[("RUNPOD_WEBHOOK_GET_JOB", "https://api.runpod.ai/v2/ep/job-take/$ID"), ("RUNPOD_WEBHOOK_POST_OUTPUT", "https://api.runpod.ai/v2/ep/job-done/$RUNPOD_POD_ID/$ID")]).unwrap().unwrap();
        assert_eq!(e.take_url(false), format!("https://api.runpod.ai/v2/ep/job-take/{}?job_in_progress=0", e.worker_id));
        assert!(e.worker_id.starts_with("local-") && e.ping.is_none() && e.ping_url(Some("x")).is_none());
        assert_eq!(e.done_url("j"), format!("https://api.runpod.ai/v2/ep/job-done/{}/j?isStream=false", e.worker_id));
        assert_eq!((e.ping_interval_s, e.api_key.as_str()), (10.0, ""));
    }

    #[test]
    fn local_mode_when_get_job_is_unset() {
        assert_eq!(env(&[]).unwrap(), None);
        assert_eq!(env(&[("RUNPOD_WEBHOOK_GET_JOB", "  ")]).unwrap(), None);
        assert!(env(&[("RUNPOD_WEBHOOK_GET_JOB", "https://x/job-take/$ID")]).unwrap_err().contains("RUNPOD_WEBHOOK_POST_OUTPUT"));
    }

    #[test]
    fn run_job_shapes() {
        let e = env(&[("RUNPOD_WEBHOOK_GET_JOB", "u/$ID"), ("RUNPOD_WEBHOOK_POST_OUTPUT", "u/$ID")]).unwrap().unwrap();
        let job = json!({"id": "j", "input": {}});
        let out = json!({"error": "x", "where": "validate", "refresh_worker": true});
        assert_eq!(run_job(&|_| out.clone(), &job, Some(&e)), json!({"output": {"where": "validate"}, "error": "x", "stopPod": true}));
        assert_eq!(run_job(&|_| json!({}), &job, Some(&e)), json!({}));
        assert_eq!(run_job(&|_| json!({"error": ""}), &job, Some(&e)), json!({}));
        assert_eq!(run_job(&|_| json!([1, 2]), &job, Some(&e)), json!({"output": [1, 2]}));
        assert_eq!(run_job(&|_| json!(true), &job, Some(&e)), json!({"output": true}));
    }

    #[test]
    fn a_panicking_handler_is_the_sdk_report() {
        let e = env(&[("RUNPOD_WEBHOOK_GET_JOB", "u/$ID"), ("RUNPOD_WEBHOOK_POST_OUTPUT", "u/$ID"), ("RUNPOD_POD_ID", "w42"), ("RUNPOD_POD_HOSTNAME", "host-1")]).unwrap().unwrap();
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let r = run_job(&|_| panic!("engine died"), &json!({"id": "boom-1", "input": {}}), Some(&e));
        std::panic::set_hook(prev);
        assert_eq!(r.as_object().unwrap().keys().collect::<Vec<_>>(), ["error"]);
        let report: Value = serde_json::from_str(r["error"].as_str().unwrap()).unwrap();
        assert_eq!(report["error_message"], "engine died");
        assert!(report["error_traceback"].as_str().unwrap().contains("Traceback"));
        assert_eq!((report["worker_id"].as_str(), report["hostname"].as_str(), report["runpod_version"].as_str()), (Some("w42"), Some("host-1"), Some(VERSION)));
    }

    #[test]
    fn run_local_prints_the_done_body() {
        let dir = crate::tempdir("runlocal");
        let f = dir.join("job.json");
        std::fs::write(&f, r#"{"input": {"engine": "h3", "prompt": "a fox"}}"#).unwrap();
        let seen = Mutex::new(Vec::new());
        let handler = |j: &Value| {
            seen.lock().unwrap().push(j.clone());
            json!({"frames": 124})
        };
        let log = crate::null_log();
        let mut out = Vec::new();
        assert_eq!(run_local(&handler, Some(f.to_str().unwrap()), &log, &mut out), 0);
        assert_eq!(*seen.lock().unwrap(), vec![json!({"id": "local_test", "input": {"engine": "h3", "prompt": "a fox"}})]);
        assert_eq!(String::from_utf8(out).unwrap(), "{\n \"output\": {\n  \"frames\": 124\n }\n}\n");
        std::fs::write(&f, r#"{"id": "j7", "input": {"op": "x"}}"#).unwrap();
        let mut out = Vec::new();
        assert_eq!(run_local(&|_| json!({"error": "unknown op", "where": "validate"}), Some(f.to_str().unwrap()), &log, &mut out), 1);
        assert_eq!(serde_json::from_slice::<Value>(&out).unwrap(), json!({"output": {"where": "validate"}, "error": "unknown op"}));
        assert_eq!(run_local(&handler, Some(dir.join("missing.json").to_str().unwrap()), &log, &mut Vec::new()), 2);
    }

    #[test]
    fn signal_handlers_set_stop() {
        let stop = Arc::new(AtomicBool::new(false));
        let logs = Arc::new(Mutex::new(Vec::new()));
        let l = logs.clone();
        install_signal_handlers(stop.clone(), Arc::new(move |s: &str| l.lock().unwrap().push(s.to_string())));
        on_signal(SIGTERM); // what the kernel would call
        let t0 = std::time::Instant::now();
        while !stop.load(Ordering::SeqCst) && t0.elapsed() < Duration::from_secs(2) {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(stop.load(Ordering::SeqCst));
        assert_eq!(*logs.lock().unwrap(), vec!["worker: SIGTERM received: stopping after the current job".to_string()]);
    }
}
