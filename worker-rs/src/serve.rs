//! `studio-worker serve`: the pod mode (docs/pod-mode.md §2). The job
//! pipeline of the serverless worker over HTTP on a persistent pod, every
//! sidecar kept resident, jobs one at a time on the pod's one GPU. Routes:
//!
//! | route | behaviour |
//! |---|---|
//! | `GET /health` | `{ok, uptime_s, busy, queued, jobs_done, pod}` |
//! | `GET /info` | `op: info` |
//! | `POST /jobs` | the job input (`{"input": …}` or the input itself) → 202 `{id, status}` |
//! | `GET /jobs/{id}` | RunPod's status shape: `output` is the last progress record while `IN_PROGRESS`, the result once `COMPLETED` |
//! | `POST /jobs/{id}/cancel` | queued → `CANCELLED`; running → the sidecars are stopped and the job ends `CANCELLED` |
//! | `GET /jobs` | the recent jobs |
//!
//! Bearer `STUDIO_SERVE_TOKEN` on every route (`STUDIO_SERVE_NO_AUTH=1` for
//! tests). `STUDIO_POD_HOURS` (default 4) is the wall-clock cap: at it the
//! pod terminates itself through RunPod's REST API v2 (`RUNPOD_API_KEY` +
//! `RUNPOD_POD_ID`) and the process exits, whatever the Studio does.

use crate::now_s;
use crate::py::py_str;
use crate::worker::Worker;
use crate::{EnvMap, Log};
use serde_json::{json, Map, Value};
use std::collections::{HashMap, VecDeque};
use std::io::Read;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;
use tiny_http::{Header, Method, Request, Response, Server};

pub const DEFAULT_BIND: &str = "0.0.0.0:8000";
const KEEP_JOBS: usize = 64;
const BODY_LIMIT: usize = 64 << 20;

#[derive(Clone, Debug)]
pub struct ServeConfig {
    pub bind: String,
    /// `None`: no auth (`STUDIO_SERVE_NO_AUTH=1`).
    pub token: Option<String>,
    pub pod_hours: f64,
    pub runpod_api_key: Option<String>,
    pub runpod_pod_id: Option<String>,
}

impl ServeConfig {
    pub fn from_env(e: &EnvMap) -> Result<ServeConfig, String> {
        let get = |k: &str| e.get(k).map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
        let no_auth = get("STUDIO_SERVE_NO_AUTH").map(|v| v == "1").unwrap_or(false);
        let token = get("STUDIO_SERVE_TOKEN");
        if token.is_none() && !no_auth {
            return Err("serve: STUDIO_SERVE_TOKEN is not set (STUDIO_SERVE_NO_AUTH=1 only for tests)".into());
        }
        let pod_hours = match get("STUDIO_POD_HOURS") {
            Some(v) => v.parse::<f64>().map_err(|_| format!("serve: STUDIO_POD_HOURS={v:?} is not a number"))?,
            None => 4.0,
        };
        Ok(ServeConfig {
            bind: get("STUDIO_SERVE_BIND").unwrap_or_else(|| DEFAULT_BIND.into()),
            token: if no_auth { None } else { token },
            pod_hours,
            runpod_api_key: get("RUNPOD_API_KEY"),
            runpod_pod_id: get("RUNPOD_POD_ID"),
        })
    }
}

#[derive(Debug, Clone)]
struct JobRec {
    id: String,
    status: String,
    input: Value,
    progress: Option<Value>,
    output: Option<Value>,
    error: Option<Value>,
    submitted: f64,
    started: Option<f64>,
    finished: Option<f64>,
    cancel: bool,
}

impl JobRec {
    /// RunPod's `/status/{id}` shape, so the Studio Worker's normaliser is shared.
    fn status_json(&self) -> Value {
        let mut o = Map::new();
        o.insert("id".into(), json!(self.id));
        o.insert("status".into(), json!(self.status));
        match self.status.as_str() {
            "IN_PROGRESS" | "IN_QUEUE" => {
                if let Some(p) = &self.progress {
                    o.insert("output".into(), p.clone());
                }
            }
            _ => {
                if let Some(out) = &self.output {
                    o.insert("output".into(), out.clone());
                }
            }
        }
        if let Some(e) = &self.error {
            o.insert("error".into(), e.clone());
        }
        if let Some(s) = self.started {
            o.insert("delayTime".into(), json!(((s - self.submitted) * 1000.0).round() as i64));
            if let Some(f) = self.finished {
                o.insert("executionTime".into(), json!(((f - s) * 1000.0).round() as i64));
            }
        }
        Value::Object(o)
    }
}

struct State {
    worker: Arc<Worker>,
    jobs: Mutex<HashMap<String, JobRec>>,
    order: Mutex<VecDeque<String>>,
    queue: Mutex<VecDeque<String>>,
    wake: Condvar,
    running: Mutex<Option<String>>,
    stop: AtomicBool,
    jobs_done: AtomicU64,
    started: f64,
    log: Log,
    cfg: ServeConfig,
}

impl State {
    fn job(&self, id: &str) -> Option<JobRec> {
        self.jobs.lock().unwrap().get(id).cloned()
    }

    fn patch(&self, id: &str, f: impl FnOnce(&mut JobRec)) {
        if let Some(j) = self.jobs.lock().unwrap().get_mut(id) {
            f(j);
        }
    }

    fn submit(&self, input: Value) -> String {
        let id = format!("{}-{}", random_hex(6), (now_s() * 1000.0) as u64);
        let rec = JobRec {
            id: id.clone(),
            status: "IN_QUEUE".into(),
            input,
            progress: None,
            output: None,
            error: None,
            submitted: now_s(),
            started: None,
            finished: None,
            cancel: false,
        };
        {
            let mut jobs = self.jobs.lock().unwrap();
            let mut order = self.order.lock().unwrap();
            jobs.insert(id.clone(), rec);
            order.push_back(id.clone());
            while order.len() > KEEP_JOBS {
                if let Some(old) = order.pop_front() {
                    let drop_it = jobs.get(&old).map(|j| j.finished.is_some()).unwrap_or(true);
                    if drop_it {
                        jobs.remove(&old);
                    } else {
                        order.push_front(old);
                        break;
                    }
                }
            }
        }
        self.queue.lock().unwrap().push_back(id.clone());
        self.wake.notify_one();
        id
    }

    /// The job thread: one job at a time, FIFO.
    fn run_jobs(self: &Arc<Self>) {
        loop {
            let next = {
                let mut q = self.queue.lock().unwrap();
                loop {
                    if self.stop.load(Ordering::SeqCst) {
                        return;
                    }
                    if let Some(id) = q.pop_front() {
                        break id;
                    }
                    q = self.wake.wait_timeout(q, Duration::from_millis(250)).unwrap().0;
                }
            };
            let Some(rec) = self.job(&next) else { continue };
            if rec.status != "IN_QUEUE" {
                continue; // cancelled while queued
            }
            let t0 = now_s();
            self.patch(&next, |j| {
                j.status = "IN_PROGRESS".into();
                j.started = Some(t0);
            });
            *self.running.lock().unwrap() = Some(next.clone());
            let job = json!({"id": next, "input": rec.input});
            let me = Arc::clone(self);
            let id = next.clone();
            let progress = move |p: &Value| {
                me.patch(&id, |j| j.progress = Some(p.clone()));
            };
            let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.worker.handle(&job, Some(&progress))));
            let now = now_s();
            self.jobs_done.fetch_add(1, Ordering::SeqCst);
            *self.running.lock().unwrap() = None;
            self.patch(&next, |j| {
                j.finished = Some(now);
                match out {
                    Ok(Value::Object(o)) => {
                        let err = o.get("error").filter(|e| crate::py::truthy(e)).cloned();
                        if j.cancel {
                            j.status = "CANCELLED".into();
                        } else if let Some(e) = err {
                            j.status = "FAILED".into();
                            j.error = Some(match o.get("where").and_then(|w| w.as_str()) {
                                Some(w) => json!(format!("{} ({w})", py_str(&e))),
                                None => e,
                            });
                        } else {
                            j.status = "COMPLETED".into();
                        }
                        j.output = Some(Value::Object(o));
                    }
                    Ok(v) => {
                        j.status = if j.cancel { "CANCELLED".into() } else { "COMPLETED".into() };
                        j.output = Some(v);
                    }
                    Err(_) => {
                        j.status = if j.cancel { "CANCELLED".into() } else { "FAILED".into() };
                        j.error = Some(json!("the worker panicked on this job"));
                    }
                }
            });
            if let Some(r) = self.job(&next) {
                (self.log)(&format!("serve: job {next} {} in {:.1} s", r.status, now - t0));
            }
        }
    }

    fn cancel(&self, id: &str) -> Option<Value> {
        let rec = self.job(id)?;
        match rec.status.as_str() {
            "IN_QUEUE" => {
                self.queue.lock().unwrap().retain(|q| q != id);
                self.patch(id, |j| {
                    j.status = "CANCELLED".into();
                    j.finished = Some(now_s());
                    j.cancel = true;
                });
            }
            "IN_PROGRESS" => {
                self.patch(id, |j| j.cancel = true);
                // The running job dies with its engine; the job thread marks it CANCELLED.
                self.worker.sidecars.stop_all();
            }
            _ => {}
        }
        Some(json!({"id": id, "status": self.job(id).map(|j| j.status).unwrap_or_else(|| "CANCELLED".into())}))
    }

    fn health(&self) -> Value {
        json!({
            "ok": true,
            "mode": "pod",
            "uptime_s": crate::py::round_to(now_s() - self.started, 1),
            "busy": self.running.lock().unwrap().is_some(),
            "running": self.running.lock().unwrap().clone(),
            "queued": self.queue.lock().unwrap().len(),
            "jobs_done": self.jobs_done.load(Ordering::SeqCst),
            "pod": self.cfg.runpod_pod_id,
            "pod_hours": self.cfg.pod_hours,
            "image": self.worker.cfg.image,
            "sidecars": self.worker.sidecars.resident(),
        })
    }

    fn recent(&self) -> Value {
        let jobs = self.jobs.lock().unwrap();
        let order = self.order.lock().unwrap();
        Value::Array(order.iter().rev().filter_map(|id| jobs.get(id)).map(|j| {
            json!({"id": j.id, "status": j.status, "submitted": j.submitted, "started": j.started, "finished": j.finished,
                   "model": j.input.get("model"), "engine": j.input.get("engine"), "op": j.input.get("op")})
        }).collect())
    }
}

fn random_hex(n: usize) -> String {
    let mut buf = vec![0u8; n];
    let mut seed = (now_s().to_bits()) ^ (std::process::id() as u64).rotate_left(32);
    for b in buf.iter_mut() {
        // xorshift64*: cheap, not secret (job ids are not secrets; the pod token is).
        seed ^= seed >> 12;
        seed ^= seed << 25;
        seed ^= seed >> 27;
        *b = (seed.wrapping_mul(0x2545F4914F6CDD1D) >> 56) as u8;
    }
    hex::encode(buf)
}

fn constant_time_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let mut diff = a.len() ^ b.len();
    let n = a.len().max(b.len()).max(1);
    for i in 0..n {
        diff |= (*a.get(i % a.len().max(1)).unwrap_or(&0) ^ *b.get(i % b.len().max(1)).unwrap_or(&0)) as usize;
    }
    diff == 0
}

fn header(name: &str, value: &str) -> Header {
    Header::from_bytes(name.as_bytes(), value.as_bytes()).expect("header")
}

fn reply(req: Request, status: u16, body: &Value) {
    let text = serde_json::to_string(body).unwrap_or_default();
    let r = Response::from_string(text)
        .with_status_code(status)
        .with_header(header("Content-Type", "application/json; charset=utf-8"))
        .with_header(header("Cache-Control", "no-store"));
    let _ = req.respond(r);
}

fn authorized(cfg: &ServeConfig, req: &Request) -> bool {
    let Some(want) = &cfg.token else { return true };
    let given = req
        .headers()
        .iter()
        .find(|h| h.field.equiv("Authorization"))
        .map(|h| h.value.as_str().to_string())
        .unwrap_or_default();
    let given = given.strip_prefix("Bearer ").or_else(|| given.strip_prefix("bearer ")).unwrap_or("").trim();
    !given.is_empty() && constant_time_eq(given, want)
}

fn read_body(req: &mut Request) -> Result<Value, String> {
    let len = req.body_length().unwrap_or(0);
    if len > BODY_LIMIT {
        return Err(format!("the body is over {BODY_LIMIT} bytes"));
    }
    let mut buf = Vec::with_capacity(len.min(BODY_LIMIT));
    req.as_reader().take(BODY_LIMIT as u64 + 1).read_to_end(&mut buf).map_err(|e| format!("read: {e}"))?;
    if buf.len() > BODY_LIMIT {
        return Err(format!("the body is over {BODY_LIMIT} bytes"));
    }
    if buf.is_empty() {
        return Ok(Value::Null);
    }
    serde_json::from_slice(&buf).map_err(|e| format!("the body is not valid JSON: {e}"))
}

/// One request. `POST /jobs` takes `{"input": …}` (RunPod's envelope) or
/// the input itself.
fn handle(state: &Arc<State>, mut req: Request) {
    let url = req.url().to_string();
    let path = url.split('?').next().unwrap_or("").trim_end_matches('/').to_string();
    let method = req.method().clone();
    if !authorized(&state.cfg, &req) {
        return reply(req, 401, &json!({"error": "unauthorized"}));
    }
    match (method, path.as_str()) {
        (Method::Get, "/health") => reply(req, 200, &state.health()),
        (Method::Get, "/info") => {
            let mut info = state.worker.info();
            info["serve"] = state.health();
            reply(req, 200, &info)
        }
        (Method::Get, "/jobs") => reply(req, 200, &json!({"jobs": state.recent()})),
        (Method::Post, "/jobs") => {
            let body = match read_body(&mut req) {
                Ok(b) => b,
                Err(e) => return reply(req, 400, &json!({"error": e})),
            };
            let input = match body.get("input") {
                Some(i) if i.is_object() => i.clone(),
                _ if body.is_object() => body,
                _ => return reply(req, 400, &json!({"error": "the body must be a JSON object (the job input)"})),
            };
            if let Some(op) = input.get("op").and_then(|o| o.as_str()) {
                if matches!(op, "serve" | "worker") {
                    return reply(req, 400, &json!({"error": format!("op {op} is not a job")}));
                }
            }
            let id = state.submit(input);
            let status = state.job(&id).map(|j| j.status).unwrap_or_else(|| "IN_QUEUE".into());
            reply(req, 202, &json!({"id": id, "status": status}))
        }
        (m, p) if p.starts_with("/jobs/") => {
            let rest = &p["/jobs/".len()..];
            let (id, action) = match rest.split_once('/') {
                Some((id, a)) => (id, Some(a)),
                None => (rest, None),
            };
            if id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
                return reply(req, 404, &json!({"error": "no such job"}));
            }
            match (m, action) {
                (Method::Get, None) => match state.job(id) {
                    Some(j) => reply(req, 200, &j.status_json()),
                    None => reply(req, 404, &json!({"error": "no such job"})),
                },
                (Method::Post, Some("cancel")) => match state.cancel(id) {
                    Some(v) => reply(req, 200, &v),
                    None => reply(req, 404, &json!({"error": "no such job"})),
                },
                _ => reply(req, 405, &json!({"error": "method not allowed"})),
            }
        }
        _ => reply(req, 404, &json!({"error": "no such route"})),
    }
}

/// Terminate this pod through RunPod's REST API v2 (`DELETE /v2/pods/{id}`),
/// if the pod knows itself; errors are logged, never fatal.
pub fn terminate_pod(cfg: &ServeConfig, log: &Log) {
    let (Some(key), Some(pod)) = (&cfg.runpod_api_key, &cfg.runpod_pod_id) else {
        log("serve: no RUNPOD_API_KEY / RUNPOD_POD_ID: the pod stays up (stop it from the Studio)");
        return;
    };
    let url = format!("https://api.runpod.io/v2/pods/{pod}");
    for attempt in 1..=3 {
        match ureq::delete(&url).set("Authorization", &format!("Bearer {key}")).timeout(Duration::from_secs(30)).call() {
            Ok(r) => {
                log(&format!("serve: pod {pod} terminated ({})", r.status()));
                return;
            }
            Err(ureq::Error::Status(code, _)) => log(&format!("serve: terminate {pod}: HTTP {code} (attempt {attempt})")),
            Err(e) => log(&format!("serve: terminate {pod}: {e} (attempt {attempt})")),
        }
        std::thread::sleep(Duration::from_secs(5));
    }
}

/// Serve until the pod-hours cap (then terminate the pod and return) or
/// `stop` is set. `ready` receives the bound address once listening.
pub fn serve(worker: Arc<Worker>, cfg: ServeConfig, log: Log, stop: Arc<AtomicBool>, ready: Option<&dyn Fn(String)>) -> Result<(), String> {
    let server = Server::http(&cfg.bind).map_err(|e| format!("serve: bind {}: {e}", cfg.bind))?;
    let addr = server.server_addr().to_string();
    let state = Arc::new(State {
        worker,
        jobs: Mutex::new(HashMap::new()),
        order: Mutex::new(VecDeque::new()),
        queue: Mutex::new(VecDeque::new()),
        wake: Condvar::new(),
        running: Mutex::new(None),
        stop: AtomicBool::new(false),
        jobs_done: AtomicU64::new(0),
        started: now_s(),
        log: log.clone(),
        cfg: cfg.clone(),
    });
    log(&format!("serve: listening on {addr} ({} auth), pod cap {:.2} h", if cfg.token.is_some() { "token" } else { "no" }, cfg.pod_hours));
    if let Some(r) = ready {
        r(addr.clone());
    }
    let jobs_thread = {
        let st = Arc::clone(&state);
        std::thread::Builder::new().name("serve-jobs".into()).spawn(move || st.run_jobs()).map_err(|e| format!("serve: jobs thread: {e}"))?
    };
    let deadline = state.started + cfg.pod_hours * 3600.0;
    let mut capped = false;
    while !stop.load(Ordering::SeqCst) {
        if now_s() >= deadline {
            log(&format!("serve: the {:.2} h pod cap is up: stopping", cfg.pod_hours));
            capped = true;
            break;
        }
        match server.recv_timeout(Duration::from_millis(250)) {
            Ok(Some(req)) => handle(&state, req),
            Ok(None) => {}
            Err(e) => {
                log(&format!("serve: recv: {e}"));
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }
    state.stop.store(true, Ordering::SeqCst);
    state.wake.notify_all();
    state.worker.sidecars.stop_all();
    let _ = jobs_thread.join();
    if capped {
        terminate_pod(&cfg, &log);
    }
    Ok(())
}
