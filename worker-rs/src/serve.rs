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
//! | `POST /streams` | `{model, prompt, max_seconds?, seed?, resolution?, aspect_ratio?, output?}`: fv-serve's `POST /fv/v1/streams` on the model's process group (started when it is not up), `output` defaulting to Cloudflare Stream → the stream object (`playback.whep_url`) |
//! | `GET /streams`, `GET /streams/{id}`, `POST /streams/{id}/commands`, `DELETE /streams/{id}` | forwarded to that fv-serve |
//!
//! | `POST /segments` | `{stream_id, segment_index, model, prompt, duration_sec, …}`: fv-serve's `POST /fv/v1/segments` on the model's process group → 202 segment object |
//! | `GET /segments/{id}` | the segment; once `succeeded` its MP4 is mirrored from fv-serve to R2 and `video_url` is the presigned copy (`fv_video_url` keeps fv-serve's) |
//! | `GET /segments/streams/{stream_id}`, `DELETE …` | the segment stream's session, forwarded |
//!
//! A live stream or a segment stream holds the GPU: jobs wait in the queue
//! ("waiting for the live stream") until every one of the pod has ended.
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
use std::collections::{HashMap, HashSet, VecDeque};
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
        let get = |k: &str| {
            e.get(k)
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        };
        let no_auth = get("STUDIO_SERVE_NO_AUTH")
            .map(|v| v == "1")
            .unwrap_or(false);
        let token = get("STUDIO_SERVE_TOKEN");
        if token.is_none() && !no_auth {
            return Err(
                "serve: STUDIO_SERVE_TOKEN is not set (STUDIO_SERVE_NO_AUTH=1 only for tests)"
                    .into(),
            );
        }
        let pod_hours = match get("STUDIO_POD_HOURS") {
            Some(v) => v
                .parse::<f64>()
                .map_err(|_| format!("serve: STUDIO_POD_HOURS={v:?} is not a number"))?,
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
            o.insert(
                "delayTime".into(),
                json!(((s - self.submitted) * 1000.0).round() as i64),
            );
            if let Some(f) = self.finished {
                o.insert(
                    "executionTime".into(),
                    json!(((f - s) * 1000.0).round() as i64),
                );
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
    /// Live streams: id → (the fv-serve base they run on, their model id).
    live: Mutex<HashMap<String, (String, String)>>,
    /// Segment streams: stream_id → (the fv-serve base, the model id); and
    /// the mirrored segments: segment id → the presigned R2 URL of its MP4.
    seg_streams: Mutex<HashMap<String, (String, String)>>,
    seg_urls: Mutex<HashMap<String, Value>>,
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
                    q = self
                        .wake
                        .wait_timeout(q, Duration::from_millis(250))
                        .unwrap()
                        .0;
                }
            };
            let Some(rec) = self.job(&next) else { continue };
            if rec.status != "IN_QUEUE" {
                continue; // cancelled while queued
            }
            // A live stream holds the GPU: wait for it to end (or the job to be cancelled).
            let mut waited = false;
            let mut ticks: u64 = 0;
            while {
                // A session fv-serve no longer knows (its segment failed at
                // the open, or it closed on idle) must not hold the queue:
                // a failed segment probe blocked every job on a pod
                // (2026-11-21). Checked on entry and then every ~5 s.
                if ticks % 10 == 0 {
                    self.prune_sessions();
                }
                ticks += 1;
                !self.live.lock().unwrap().is_empty()
                    || !self.seg_streams.lock().unwrap().is_empty()
            } && !self.stop.load(Ordering::SeqCst)
            {
                if self
                    .job(&next)
                    .map(|j| j.status != "IN_QUEUE")
                    .unwrap_or(true)
                {
                    break;
                }
                if !waited {
                    self.patch(&next, |j| j.progress = Some(json!({"stage": "queued", "detail": "waiting for the live stream", "elapsed_s": 0})));
                    waited = true;
                }
                std::thread::sleep(Duration::from_millis(500));
            }
            if self
                .job(&next)
                .map(|j| j.status != "IN_QUEUE")
                .unwrap_or(true)
            {
                continue;
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
            let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.worker.handle(&job, Some(&progress))
            }));
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
                        j.status = if j.cancel {
                            "CANCELLED".into()
                        } else {
                            "COMPLETED".into()
                        };
                        j.output = Some(v);
                    }
                    Err(_) => {
                        j.status = if j.cancel {
                            "CANCELLED".into()
                        } else {
                            "FAILED".into()
                        };
                        j.error = Some(json!("the worker panicked on this job"));
                    }
                }
            });
            if let Some(r) = self.job(&next) {
                (self.log)(&format!(
                    "serve: job {next} {} in {:.1} s",
                    r.status,
                    now - t0
                ));
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
        Some(
            json!({"id": id, "status": self.job(id).map(|j| j.status).unwrap_or_else(|| "CANCELLED".into())}),
        )
    }

    /// Forgets the live and segment sessions fv-serve no longer has (404 on
    /// its own route): they would otherwise hold the job queue for the rest
    /// of the pod's life. Only sessions whose fv-serve answers are dropped;
    /// a transport error leaves the session alone.
    fn prune_sessions(&self) {
        let live: Vec<(String, String)> = self
            .live
            .lock()
            .unwrap()
            .iter()
            .map(|(id, (b, _))| (id.clone(), b.clone()))
            .collect();
        for (id, base) in live {
            if forward("GET", &format!("{base}/fv/v1/streams/{id}"), None).0 == 404 {
                self.live.lock().unwrap().remove(&id);
                (self.log)(&format!("serve: live stream {id} is gone from fv-serve"));
            }
        }
        let segs: Vec<(String, String)> = self
            .seg_streams
            .lock()
            .unwrap()
            .iter()
            .map(|(id, (b, _))| (id.clone(), b.clone()))
            .collect();
        for (id, base) in segs {
            if forward("GET", &format!("{base}/fv/v1/segments/streams/{id}"), None).0 == 404 {
                self.seg_streams.lock().unwrap().remove(&id);
                (self.log)(&format!("serve: segment stream {id} is gone from fv-serve"));
            }
        }
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
            "streams": self.live.lock().unwrap().keys().cloned().collect::<Vec<_>>(),
            "segment_streams": self.seg_streams.lock().unwrap().keys().cloned().collect::<Vec<_>>(),
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
        diff |= (*a.get(i % a.len().max(1)).unwrap_or(&0)
            ^ *b.get(i % b.len().max(1)).unwrap_or(&0)) as usize;
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
    let given = given
        .strip_prefix("Bearer ")
        .or_else(|| given.strip_prefix("bearer "))
        .unwrap_or("")
        .trim();
    !given.is_empty() && constant_time_eq(given, want)
}

fn read_body(req: &mut Request) -> Result<Value, String> {
    let len = req.body_length().unwrap_or(0);
    if len > BODY_LIMIT {
        return Err(format!("the body is over {BODY_LIMIT} bytes"));
    }
    let mut buf = Vec::with_capacity(len.min(BODY_LIMIT));
    req.as_reader()
        .take(BODY_LIMIT as u64 + 1)
        .read_to_end(&mut buf)
        .map_err(|e| format!("read: {e}"))?;
    if buf.len() > BODY_LIMIT {
        return Err(format!("the body is over {BODY_LIMIT} bytes"));
    }
    if buf.is_empty() {
        return Ok(Value::Null);
    }
    serde_json::from_slice(&buf).map_err(|e| format!("the body is not valid JSON: {e}"))
}

/// A forwarded fv-serve call: its status and JSON (an unreachable fv-serve
/// is a 502 with the reason).
fn forward(method: &str, url: &str, body: Option<&Value>) -> (u16, Value) {
    let mut r = ureq::request(method, url).timeout(Duration::from_secs(120));
    if body.is_some() {
        r = r.set("Content-Type", "application/json");
    }
    let resp = match body {
        Some(b) => r.send_string(&b.to_string()),
        None => r.call(),
    };
    let parse = |resp: ureq::Response| -> (u16, Value) {
        let code = resp.status();
        let text = resp.into_string().unwrap_or_default();
        let v = serde_json::from_str(&text).unwrap_or_else(|_| {
            if text.is_empty() {
                json!({})
            } else {
                json!({"raw": text})
            }
        });
        (code, v)
    };
    match resp {
        Ok(ok) => parse(ok),
        Err(ureq::Error::Status(_, resp)) => parse(resp),
        Err(e) => (502, json!({"error": format!("fv-serve: {e}")})),
    }
}

const TERMINAL_STREAM: [&str; 4] = ["closed", "ended", "failed", "stopped"];

/// The live streams (docs/pod-mode.md §2, phase 2): `POST /streams` opens
/// one on the model's fv-serve; the rest is forwarded by id.
fn handle_streams(state: &Arc<State>, mut req: Request, method: Method, path: &str) {
    if method == Method::Post && path == "/streams" {
        let body = match read_body(&mut req) {
            Ok(b) if b.is_object() => b,
            Ok(_) => {
                return reply(
                    req,
                    400,
                    &json!({"error": "the body must be a JSON object"}),
                )
            }
            Err(e) => return reply(req, 400, &json!({"error": e})),
        };
        let model_id = body.get("model").map(py_str).unwrap_or_default();
        let prompt = body.get("prompt").map(py_str).unwrap_or_default();
        if model_id.is_empty() || prompt.trim().is_empty() {
            return reply(req, 400, &json!({"error": "model and prompt are required"}));
        }
        let Some(model) = state.worker.catalog.model("fastvideo", &model_id).cloned() else {
            return reply(
                req,
                400,
                &json!({"error": format!("unknown fastvideo model {}", crate::py::py_str_repr(&model_id))}),
            );
        };
        let variant = model.recipe.get("variant").map(py_str).unwrap_or_default();
        if let Some(c) = session_conflict(state, &model) {
            return reply(req, 409, &c);
        }
        let (base, load_s) = match state.worker.fastvideo_serve(&model.id, None) {
            Ok(v) => v,
            Err(e) => return reply(req, 503, &e.to_output()),
        };
        let mut out = Map::new();
        out.insert("model".into(), json!(variant));
        out.insert("prompt".into(), json!(prompt));
        out.insert(
            "output".into(),
            body.get("output")
                .cloned()
                .unwrap_or_else(|| json!({"kind": "cloudflare"})),
        );
        // fv-serve's StreamBody, which denies unknown fields: a canvas is
        // `width` / `height` here, not the `resolution` / `aspect_ratio` of
        // jobs and segments (2026-11-21: the page sent those and every
        // stream answered 400 `unknown field`). Anything else the caller
        // sends is dropped, so an older page still opens a stream.
        for k in [
            "width",
            "height",
            "fps",
            "seed",
            "max_seconds",
            "audio",
            "continuity",
            "autoplay",
            "clips",
        ] {
            if let Some(v) = body.get(k) {
                out.insert(k.into(), v.clone());
            }
        }
        let (code, mut v) = forward(
            "POST",
            &format!("{base}/fv/v1/streams"),
            Some(&Value::Object(out)),
        );
        if (200..300).contains(&code) {
            if let Some(id) = v.get("id").and_then(|i| i.as_str()) {
                state
                    .live
                    .lock()
                    .unwrap()
                    .insert(id.to_string(), (base.clone(), model.id.clone()));
                (state.log)(&format!("serve: stream {id} on {} ({model_id})", model.id));
            }
            if let Value::Object(o) = &mut v {
                o.insert("studio_model".into(), json!(model.id));
                o.insert("load_s".into(), json!(crate::py::round_to(load_s, 1)));
            }
        }
        return reply(req, code, &v);
    }
    if method == Method::Get && path == "/streams" {
        let live = state.live.lock().unwrap().clone();
        let bases: HashSet<String> = live.values().map(|(b, _)| b.clone()).collect();
        let mut all = Vec::new();
        for b in bases {
            let (code, v) = forward("GET", &format!("{b}/fv/v1/streams"), None);
            if (200..300).contains(&code) {
                if let Some(a) = v.get("data").and_then(|d| d.as_array()) {
                    all.extend(a.iter().cloned());
                } else if let Some(a) = v.as_array() {
                    all.extend(a.iter().cloned());
                }
            }
        }
        return reply(
            req,
            200,
            &json!({"data": all, "live": live.keys().cloned().collect::<Vec<_>>()}),
        );
    }
    let rest = path.strip_prefix("/streams/").unwrap_or("");
    let (id, action) = match rest.split_once('/') {
        Some((id, a)) => (id, Some(a)),
        None => (rest, None),
    };
    if id.is_empty()
        || !id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return reply(req, 404, &json!({"error": "no such stream"}));
    }
    let Some(base) = state.live.lock().unwrap().get(id).map(|(b, _)| b.clone()) else {
        return reply(req, 404, &json!({"error": "no such stream on this pod"}));
    };
    match (method, action) {
        (Method::Get, None) => {
            let (code, v) = forward("GET", &format!("{base}/fv/v1/streams/{id}"), None);
            let st = v
                .get("status")
                .and_then(|s| s.get("state"))
                .and_then(|s| s.as_str())
                .unwrap_or("");
            if code == 404 || TERMINAL_STREAM.contains(&st) {
                state.live.lock().unwrap().remove(id);
            }
            reply(req, code, &v)
        }
        (Method::Post, Some("commands")) => {
            let body = match read_body(&mut req) {
                Ok(b) => b,
                Err(e) => return reply(req, 400, &json!({"error": e})),
            };
            let (code, v) = forward(
                "POST",
                &format!("{base}/fv/v1/streams/{id}/commands"),
                Some(&body),
            );
            reply(req, code, &v)
        }
        (Method::Delete, None) => {
            let (code, v) = forward("DELETE", &format!("{base}/fv/v1/streams/{id}"), None);
            if (200..300).contains(&code) || code == 404 {
                state.live.lock().unwrap().remove(id);
            }
            reply(req, code, &v)
        }
        _ => reply(req, 405, &json!({"error": "method not allowed"})),
    }
}

/// The segments (docs/pod-mode.md §2, phase 3): fv-serve's `/fv/v1/segments`
/// on the model's fv-serve; a finished segment's MP4 is mirrored to R2 so the
/// page plays a presigned URL (fv-serve's signed local URL is not reachable
/// from outside the pod).
fn handle_segments(state: &Arc<State>, mut req: Request, method: Method, path: &str) {
    let seg_id_ok = |id: &str| {
        !id.is_empty()
            && id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
    };
    if method == Method::Post && path == "/segments" {
        let body = match read_body(&mut req) {
            Ok(b) if b.is_object() => b,
            Ok(_) => {
                return reply(
                    req,
                    400,
                    &json!({"error": "the body must be a JSON object"}),
                )
            }
            Err(e) => return reply(req, 400, &json!({"error": e})),
        };
        let stream_id = body.get("stream_id").map(py_str).unwrap_or_default();
        let model_id = body.get("model").map(py_str).unwrap_or_default();
        let prompt = body.get("prompt").map(py_str).unwrap_or_default();
        if stream_id.is_empty()
            || model_id.is_empty()
            || prompt.trim().is_empty()
            || body.get("segment_index").and_then(|i| i.as_u64()).is_none()
        {
            return reply(
                req,
                400,
                &json!({"error": "stream_id, segment_index, model and prompt are required"}),
            );
        }
        if !seg_id_ok(&stream_id) {
            return reply(
                req,
                400,
                &json!({"error": "stream_id: letters, digits, - _ . only"}),
            );
        }
        let Some(model) = state.worker.catalog.model("fastvideo", &model_id).cloned() else {
            return reply(
                req,
                400,
                &json!({"error": format!("unknown fastvideo model {}", crate::py::py_str_repr(&model_id))}),
            );
        };
        let variant = model.recipe.get("variant").map(py_str).unwrap_or_default();
        // A stream keeps the fv-serve it opened on.
        let known = state
            .seg_streams
            .lock()
            .unwrap()
            .get(&stream_id)
            .map(|(b, _)| b.clone());
        let (base, load_s) = match known {
            Some(b) => (b, 0.0),
            None => {
                if let Some(c) = session_conflict(state, &model) {
                    return reply(req, 409, &c);
                }
                match state.worker.fastvideo_serve(&model.id, None) {
                    Ok(v) => v,
                    Err(e) => return reply(req, 503, &e.to_output()),
                }
            }
        };
        // fv-serve's SegmentRequest, which denies unknown fields too (this
        // path used to forward the body as it came).
        let mut out = Map::new();
        out.insert("model".into(), json!(variant));
        for k in [
            "stream_id",
            "segment_index",
            "prompt",
            "duration_sec",
            "resolution",
            "aspect_ratio",
            "seed",
            "recache",
            "reset",
            "reference_image_urls",
            "end_image_url",
            "context",
            "callback_url",
            "extra",
        ] {
            if let Some(v) = body.get(k) {
                out.insert(k.into(), v.clone());
            }
        }
        let (code, mut v) = forward(
            "POST",
            &format!("{base}/fv/v1/segments"),
            Some(&Value::Object(out)),
        );
        if (200..300).contains(&code) {
            state
                .seg_streams
                .lock()
                .unwrap()
                .insert(stream_id.clone(), (base.clone(), model.id.clone()));
            if let Value::Object(o) = &mut v {
                o.insert("studio_model".into(), json!(model.id));
                o.insert("load_s".into(), json!(crate::py::round_to(load_s, 1)));
            }
        }
        return reply(req, code, &v);
    }
    if let Some(rest) = path.strip_prefix("/segments/streams/") {
        let sid = rest.trim_end_matches('/');
        if !seg_id_ok(sid) {
            return reply(req, 404, &json!({"error": "no such stream"}));
        }
        let Some(base) = state
            .seg_streams
            .lock()
            .unwrap()
            .get(sid)
            .map(|(b, _)| b.clone())
        else {
            return reply(
                req,
                404,
                &json!({"error": "no such segment stream on this pod"}),
            );
        };
        return match method {
            Method::Get => {
                let (code, v) =
                    forward("GET", &format!("{base}/fv/v1/segments/streams/{sid}"), None);
                if code == 404 {
                    state.seg_streams.lock().unwrap().remove(sid);
                }
                reply(req, code, &v)
            }
            Method::Delete => {
                let (code, v) = forward(
                    "DELETE",
                    &format!("{base}/fv/v1/segments/streams/{sid}"),
                    None,
                );
                if (200..300).contains(&code) || code == 404 {
                    state.seg_streams.lock().unwrap().remove(sid);
                }
                reply(req, code, &v)
            }
            _ => reply(req, 405, &json!({"error": "method not allowed"})),
        };
    }
    let id = path
        .strip_prefix("/segments/")
        .unwrap_or("")
        .trim_end_matches('/');
    if method != Method::Get || !seg_id_ok(id) {
        return reply(
            req,
            if method == Method::Get { 404 } else { 405 },
            &json!({"error": "no such segment"}),
        );
    }
    // A segment is looked up on every fv-serve a segment stream runs on (one in practice).
    let bases: HashSet<String> = state
        .seg_streams
        .lock()
        .unwrap()
        .values()
        .map(|(b, _)| b.clone())
        .collect();
    let mut found: Option<(u16, Value)> = None;
    for b in bases {
        let (code, v) = forward("GET", &format!("{b}/fv/v1/segments/{id}"), None);
        if code != 404 {
            found = Some((code, v));
            break;
        }
    }
    let Some((code, mut v)) = found else {
        return reply(req, 404, &json!({"error": "no such segment on this pod"}));
    };
    if code == 200 && v.get("status").and_then(|s| s.as_str()) == Some("succeeded") {
        // The guard must not live into the branch that locks again.
        let mirrored = state.seg_urls.lock().unwrap().get(id).cloned();
        let fv_url = v
            .get("video_url")
            .and_then(|u| u.as_str())
            .map(str::to_string);
        match (mirrored, fv_url) {
            (Some(m), _) => {
                v["fv_video_url"] = v["video_url"].clone();
                v["video_url"] = m;
            }
            (None, Some(url)) => match mirror_segment(state, id, &v, &url) {
                Ok(presigned) => {
                    state
                        .seg_urls
                        .lock()
                        .unwrap()
                        .insert(id.to_string(), presigned.clone());
                    v["fv_video_url"] = json!(url);
                    v["video_url"] = presigned;
                }
                Err(e) => {
                    v["mirror_error"] = json!(e);
                }
            },
            (None, None) => {}
        }
    }
    reply(req, code, &v)
}

/// Download a finished segment from fv-serve and put it in the bucket under
/// `segments/<stream>/<index>-<id>.mp4`; the presigned URL back.
fn mirror_segment(state: &Arc<State>, id: &str, seg: &Value, url: &str) -> Result<Value, String> {
    let (Some(store), Some(bucket)) = (state.worker.store.as_ref(), state.worker.bucket.as_ref())
    else {
        return Err("no object store on this pod: the MP4 stays on fv-serve".into());
    };
    let dir = state.worker.cfg.jobs_dir.join("segments");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let local = dir.join(format!("{id}.mp4"));
    crate::httpc::download(url, &local, 300.0, None).map_err(|e| format!("download: {e}"))?;
    let stream = seg
        .get("stream_id")
        .map(py_str)
        .unwrap_or_else(|| "stream".into());
    let index = seg
        .get("segment_index")
        .and_then(|i| i.as_u64())
        .unwrap_or(0);
    let key = format!("segments/{stream}/{index:04}-{id}.mp4");
    store
        .put_file(bucket, &key, &local, Some("video/mp4"))
        .map_err(|e| format!("upload: {e}"))?;
    let _ = std::fs::remove_file(&local);
    Ok(json!(store.presign_get(
        bucket,
        &key,
        state.worker.cfg.url_ttl_s
    )))
}

/// One request. `POST /jobs` takes `{"input": …}` (RunPod's envelope) or
/// the input itself.
fn handle(state: &Arc<State>, mut req: Request) {
    let url = req.url().to_string();
    let path = url
        .split('?')
        .next()
        .unwrap_or("")
        .trim_end_matches('/')
        .to_string();
    let method = req.method().clone();
    if !authorized(&state.cfg, &req) {
        return reply(req, 401, &json!({"error": "unauthorized"}));
    }
    if path == "/streams" || path.starts_with("/streams/") {
        return handle_streams(state, req, method, &path);
    }
    if path == "/segments" || path.starts_with("/segments/") {
        return handle_segments(state, req, method, &path);
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
                _ => {
                    return reply(
                        req,
                        400,
                        &json!({"error": "the body must be a JSON object (the job input)"}),
                    )
                }
            };
            if let Some(op) = input.get("op").and_then(|o| o.as_str()) {
                if matches!(op, "serve" | "worker") {
                    return reply(req, 400, &json!({"error": format!("op {op} is not a job")}));
                }
            }
            let id = state.submit(input);
            // RunPod's answer to /run: the job is queued, whatever the thread does next.
            reply(req, 202, &json!({"id": id, "status": "IN_QUEUE"}))
        }
        (m, p) if p.starts_with("/jobs/") => {
            let rest = &p["/jobs/".len()..];
            let (id, action) = match rest.split_once('/') {
                Some((id, a)) => (id, Some(a)),
                None => (rest, None),
            };
            if id.is_empty()
                || !id
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            {
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
        match ureq::delete(&url)
            .set("Authorization", &format!("Bearer {key}"))
            .timeout(Duration::from_secs(30))
            .call()
        {
            Ok(r) => {
                log(&format!("serve: pod {pod} terminated ({})", r.status()));
                return;
            }
            Err(ureq::Error::Status(code, _)) => log(&format!(
                "serve: terminate {pod}: HTTP {code} (attempt {attempt})"
            )),
            Err(e) => log(&format!("serve: terminate {pod}: {e} (attempt {attempt})")),
        }
        std::thread::sleep(Duration::from_secs(5));
    }
}

/// A session for another model restarts fv-serve (it keeps one model
/// resident), which would kill the sessions already on it: refuse instead,
/// naming them.
fn session_conflict(state: &Arc<State>, model: &crate::catalog::Model) -> Option<Value> {
    let mut others: Vec<String> = Vec::new();
    for (id, (_, m)) in state.live.lock().unwrap().iter() {
        if *m != model.id {
            others.push(format!("live stream {id} on {m}"));
        }
    }
    for (id, (_, m)) in state.seg_streams.lock().unwrap().iter() {
        if *m != model.id {
            others.push(format!("segment stream {id} on {m}"));
        }
    }
    if others.is_empty() {
        return None;
    }
    others.sort();
    Some(json!({
        "error": format!(
            "fv-serve keeps one model resident and {} is open: stop it before starting a session on {}",
            others.join(", "),
            model.id
        ),
        "where": "engine",
    }))
}

/// Serve until the pod-hours cap (then terminate the pod and return) or
/// `stop` is set. `ready` receives the bound address once listening.
pub fn serve(
    worker: Arc<Worker>,
    cfg: ServeConfig,
    log: Log,
    stop: Arc<AtomicBool>,
    ready: Option<&dyn Fn(String)>,
) -> Result<(), String> {
    // fv-serve binds 127.0.0.1:<its port>; the same port here would let the
    // sidecar die at start with "Address already in use" on every stream.
    let fv_port = worker.cfg.ports.get("fastvideo").copied().unwrap_or(0);
    let bind_port = cfg
        .bind
        .rsplit(':')
        .next()
        .and_then(|p| p.parse::<u16>().ok());
    if bind_port == Some(fv_port) && fv_port != 0 {
        return Err(format!(
            "serve: STUDIO_SERVE_BIND {} uses fv-serve's port {fv_port}; set STUDIO_FV_PORT to another port",
            cfg.bind
        ));
    }
    let server = Server::http(&cfg.bind).map_err(|e| format!("serve: bind {}: {e}", cfg.bind))?;
    let addr = server.server_addr().to_string();
    let state = Arc::new(State {
        worker,
        jobs: Mutex::new(HashMap::new()),
        order: Mutex::new(VecDeque::new()),
        queue: Mutex::new(VecDeque::new()),
        wake: Condvar::new(),
        running: Mutex::new(None),
        live: Mutex::new(HashMap::new()),
        seg_streams: Mutex::new(HashMap::new()),
        seg_urls: Mutex::new(HashMap::new()),
        stop: AtomicBool::new(false),
        jobs_done: AtomicU64::new(0),
        started: now_s(),
        log: log.clone(),
        cfg: cfg.clone(),
    });
    log(&format!(
        "serve: listening on {addr} ({} auth), pod cap {:.2} h",
        if cfg.token.is_some() { "token" } else { "no" },
        cfg.pod_hours
    ));
    if let Some(r) = ready {
        r(addr.clone());
    }
    let jobs_thread = {
        let st = Arc::clone(&state);
        std::thread::Builder::new()
            .name("serve-jobs".into())
            .spawn(move || st.run_jobs())
            .map_err(|e| format!("serve: jobs thread: {e}"))?
    };
    let deadline = state.started + cfg.pod_hours * 3600.0;
    let mut capped = false;
    while !stop.load(Ordering::SeqCst) {
        if now_s() >= deadline {
            log(&format!(
                "serve: the {:.2} h pod cap is up: stopping",
                cfg.pod_hours
            ));
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
