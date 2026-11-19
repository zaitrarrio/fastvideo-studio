//! `studio-worker serve` (docs/pod-mode.md §2): the job pipeline over HTTP
//! on a port, driven with ureq. No engines: `op: info` / `op: r2` jobs and a
//! job that fails validation exercise the queue, the status shapes, auth
//! and cancel.

use fvstudio::serve::{serve, ServeConfig};
use fvstudio::worker::Worker;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

fn env_map(dir: &std::path::Path) -> HashMap<String, String> {
    let mut e = HashMap::new();
    e.insert(
        "STUDIO_CATALOG".into(),
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../catalog/catalog.json").to_string_lossy().into_owned(),
    );
    e.insert("STUDIO_R2_LOCAL_DIR".into(), dir.join("r2").to_string_lossy().into_owned());
    e.insert("STUDIO_R2_BUCKET".into(), "studio-test".into());
    e.insert("STUDIO_VOLUME".into(), dir.join("vol").to_string_lossy().into_owned());
    e.insert("STUDIO_JOBS_DIR".into(), dir.join("jobs").to_string_lossy().into_owned());
    e.insert("STUDIO_SERVE_BIND".into(), "127.0.0.1:0".into());
    e.insert("STUDIO_SERVE_TOKEN".into(), "pod-token".into());
    e.insert("STUDIO_POD_HOURS".into(), "1".into());
    e
}

struct Pod {
    base: String,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Pod {
    fn start(env: HashMap<String, String>) -> Pod {
        let worker = Arc::new(Worker::from_env_map(&env, fvstudio::stderr_log()).unwrap());
        let cfg = ServeConfig::from_env(&env).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let addr: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let (a, s) = (addr.clone(), stop.clone());
        let thread = std::thread::spawn(move || {
            let ready = |bound: String| *a.lock().unwrap() = Some(bound);
            serve(worker, cfg, fvstudio::stderr_log(), s, Some(&ready)).unwrap();
        });
        let t0 = Instant::now();
        let base = loop {
            if let Some(b) = addr.lock().unwrap().clone() {
                break format!("http://{b}");
            }
            assert!(t0.elapsed() < Duration::from_secs(10), "serve did not bind");
            std::thread::sleep(Duration::from_millis(20));
        };
        Pod { base, stop, thread: Some(thread) }
    }

    fn call(&self, method: &str, path: &str, body: Option<Value>, auth: bool) -> (u16, Value) {
        let url = format!("{}{}", self.base, path);
        let mut r = ureq::request(method, &url).timeout(Duration::from_secs(30));
        if auth {
            r = r.set("Authorization", "Bearer pod-token");
        }
        let resp = match body {
            Some(b) => r.set("Content-Type", "application/json").send_string(&b.to_string()),
            None => r.call(),
        };
        let parse = |resp: ureq::Response| -> (u16, Value) {
            let code = resp.status();
            let text = resp.into_string().unwrap_or_default();
            (code, serde_json::from_str(&text).unwrap_or(Value::Null))
        };
        match resp {
            Ok(ok) => parse(ok),
            Err(ureq::Error::Status(_, resp)) => parse(resp),
            Err(e) => panic!("{method} {path}: {e}"),
        }
    }

    fn wait(&self, id: &str) -> Value {
        let t0 = Instant::now();
        loop {
            let (code, s) = self.call("GET", &format!("/jobs/{id}"), None, true);
            assert_eq!(code, 200, "{s}");
            if matches!(s["status"].as_str(), Some("COMPLETED" | "FAILED" | "CANCELLED")) {
                return s;
            }
            assert!(t0.elapsed() < Duration::from_secs(120), "job {id} did not finish: {s}");
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

impl Drop for Pod {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

#[test]
fn serve_runs_jobs_over_http_with_runpod_status_shapes() {
    let dir = fvstudio::tempdir("serve");
    let pod = Pod::start(env_map(&dir));

    // Auth on every route; health and info without a job.
    assert_eq!(pod.call("GET", "/health", None, false).0, 401);
    let (code, h) = pod.call("GET", "/health", None, true);
    assert_eq!(code, 200);
    assert_eq!(h["ok"], true);
    assert_eq!(h["mode"], "pod");
    assert_eq!(h["busy"], false);
    assert_eq!(h["pod_hours"], 1.0);
    let (code, info) = pod.call("GET", "/info", None, true);
    assert_eq!(code, 200, "{info}");
    assert_eq!(info["op"], "info");
    assert_eq!(info["serve"]["mode"], "pod");

    // A job through the queue: RunPod's envelope or the bare input.
    let (code, sub) = pod.call("POST", "/jobs", Some(json!({"input": {"op": "info"}})), true);
    assert_eq!(code, 202, "{sub}");
    assert_eq!(sub["status"], "IN_QUEUE");
    let id = sub["id"].as_str().unwrap().to_string();
    let done = pod.wait(&id);
    assert_eq!(done["status"], "COMPLETED", "{done}");
    assert_eq!(done["output"]["op"], "info");
    assert!(done["delayTime"].is_i64() && done["executionTime"].is_i64(), "{done}");

    let (code, sub) = pod.call("POST", "/jobs", Some(json!({"op": "r2"})), true);
    assert_eq!(code, 202);
    let done = pod.wait(sub["id"].as_str().unwrap());
    assert_eq!(done["status"], "COMPLETED", "{done}");
    assert_eq!(done["output"]["op"], "r2");

    // A refused job is COMPLETED with an error output on RunPod; here it is FAILED with the error.
    let (code, sub) = pod.call("POST", "/jobs", Some(json!({"input": {"op": "nope"}})), true);
    assert_eq!(code, 202);
    let done = pod.wait(sub["id"].as_str().unwrap());
    assert_eq!(done["status"], "FAILED", "{done}");
    assert!(done["error"].as_str().unwrap().contains("nope"), "{done}");
    assert_eq!(done["output"]["where"], "validate");

    // Bad bodies, unknown jobs, the serve / worker ops.
    assert_eq!(pod.call("POST", "/jobs", Some(json!([1, 2])), true).0, 400);
    assert_eq!(pod.call("POST", "/jobs", Some(json!({"op": "serve"})), true).0, 400);
    assert_eq!(pod.call("GET", "/jobs/nope", None, true).0, 404);
    assert_eq!(pod.call("POST", "/jobs/nope/cancel", None, true).0, 404);
    assert_eq!(pod.call("GET", "/nothing", None, true).0, 404);
    assert_eq!(pod.call("GET", "/jobs/../x", None, true).0, 404);

    // The recent list holds every job, newest first; jobs_done counts them.
    let (code, list) = pod.call("GET", "/jobs", None, true);
    assert_eq!(code, 200);
    let jobs = list["jobs"].as_array().unwrap();
    assert_eq!(jobs.len(), 3);
    assert_eq!(jobs[2]["id"], id);
    let (_, h) = pod.call("GET", "/health", None, true);
    assert_eq!(h["jobs_done"], 3);

    // Cancel of a finished job answers its status; of a queued one, CANCELLED.
    let (code, c) = pod.call("POST", &format!("/jobs/{id}/cancel"), None, true);
    assert_eq!(code, 200);
    assert_eq!(c["status"], "COMPLETED");
    let mut queued = Vec::new();
    for _ in 0..6 {
        let (_, s) = pod.call("POST", "/jobs", Some(json!({"op": "r2"})), true);
        queued.push(s["id"].as_str().unwrap().to_string());
    }
    let last = queued.last().unwrap().clone();
    let (code, c) = pod.call("POST", &format!("/jobs/{last}/cancel"), None, true);
    assert_eq!(code, 200);
    let final_ = pod.wait(&last);
    assert!(matches!(final_["status"].as_str(), Some("CANCELLED" | "COMPLETED")), "{final_} after {c}");
    for q in &queued[..queued.len() - 1] {
        assert_eq!(pod.wait(q)["status"], "COMPLETED");
    }
}

#[test]
fn serve_config_needs_a_token_unless_told_not_to() {
    let mut e = HashMap::new();
    assert!(ServeConfig::from_env(&e).unwrap_err().contains("STUDIO_SERVE_TOKEN"));
    e.insert("STUDIO_SERVE_NO_AUTH".into(), "1".into());
    let c = ServeConfig::from_env(&e).unwrap();
    assert!(c.token.is_none());
    assert_eq!(c.bind, fvstudio::serve::DEFAULT_BIND);
    assert_eq!(c.pod_hours, 4.0);
    e.insert("STUDIO_POD_HOURS".into(), "x".into());
    assert!(ServeConfig::from_env(&e).unwrap_err().contains("STUDIO_POD_HOURS"));
}
