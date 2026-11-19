//! `fvstudio::rp`: RunPod's serverless protocol against a fake RunPod that
//! hands out jobs, records the takes, the progress and done posts with their
//! headers, and the pings (the port of `worker/tests/test_rp.py`).

mod common;

use common::FakeRunpod;
use fvstudio::rp::{self, RunpodClient, VERSION};
use serde_json::{json, Value};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

type Logs = Arc<Mutex<Vec<String>>>;

fn client(f: &FakeRunpod, logs: Option<Logs>, sleeps: Option<Arc<Mutex<Vec<f64>>>>) -> Arc<RunpodClient> {
    let log: fvstudio::Log = match logs {
        Some(l) => Arc::new(move |s: &str| l.lock().unwrap().push(s.to_string())),
        None => fvstudio::null_log(),
    };
    let sleep: rp::Sleep = match sleeps {
        Some(s) => Arc::new(move |d: f64| s.lock().unwrap().push(d)),
        None => Arc::new(|_| {}),
    };
    RunpodClient::new(f.env(), log, sleep)
}

fn q(v: &std::collections::HashMap<String, Vec<String>>) -> Value {
    json!(v)
}

#[test]
fn one_job_round_trip() {
    let f = FakeRunpod::start();
    f.state.lock().unwrap().jobs.push(json!({"id": "job-1", "input": {"engine": "h3", "prompt": "a fox"}, "webhook": null}));
    let c = client(&f, None, None);
    let seen = Mutex::new(Vec::new());
    let cc = c.clone();
    let handler = |job: &Value| {
        seen.lock().unwrap().push(job.clone());
        let id = job["id"].as_str().unwrap();
        cc.progress_update(id, &json!({"stage": "staging", "elapsed_s": 0.1}));
        cc.progress_update(id, &json!({"stage": "denoising", "step": 3, "total_steps": 8, "ünïcode": "✓"}));
        std::thread::sleep(std::time::Duration::from_millis(250)); // a ping or two while the job runs
        json!({"video_url": "https://r2/v.mp4", "frames": 124})
    };
    let n = rp::serve(&handler, &c, Arc::new(AtomicBool::new(false)), Some(1), 0.0);
    assert_eq!(n, 1);
    assert_eq!(*seen.lock().unwrap(), vec![json!({"id": "job-1", "input": {"engine": "h3", "prompt": "a fox"}, "webhook": null})]);
    let st = f.state.lock().unwrap();
    // take: $ID → worker id, the template's query kept, job_in_progress appended
    let (path, qq, h) = &st.takes[0];
    assert_eq!(path, "/v2/ep1/job-take/w42");
    assert_eq!(q(qq), json!({"gpu": ["NVIDIA H200"], "job_in_progress": ["0"]}));
    assert_eq!(h["authorization"], "secret-key");
    drop(st);
    // progress: in order, before the result, verbatim objects
    let prog = f.progress_posts();
    assert_eq!(prog.iter().map(|p| p.3["output"].clone()).collect::<Vec<_>>(), vec![json!({"stage": "staging", "elapsed_s": 0.1}), json!({"stage": "denoising", "step": 3, "total_steps": 8, "ünïcode": "✓"})]);
    for (path, qq, h, body) in &prog {
        assert_eq!(path, "/v2/ep1/job-done/w42/job-1");
        assert_eq!(q(qq), json!({"gpu": ["NVIDIA H200"], "isStream": ["false"]}));
        assert_eq!((h["authorization"].as_str(), h["x-request-id"].as_str(), h["content-type"].as_str()), ("secret-key", "job-1", "application/x-www-form-urlencoded"));
        assert_eq!(body["status"], "IN_PROGRESS");
    }
    let posts = f.state.lock().unwrap().posts.clone();
    let done = f.done_posts();
    assert_eq!(done.len(), 1);
    let last_prog = posts.iter().position(|p| p == prog.last().unwrap()).unwrap();
    let first_done = posts.iter().position(|p| p == &done[0]).unwrap();
    assert!(last_prog < first_done);
    // done: {"output": …}, no status, isStream=false, same headers
    let (path, qq, h, body) = &done[0];
    assert_eq!(path, "/v2/ep1/job-done/w42/job-1");
    assert_eq!((qq["isStream"].clone(), qq["gpu"].clone()), (vec!["false".to_string()], vec!["NVIDIA H200".to_string()]));
    assert_eq!(body, &json!({"output": {"video_url": "https://r2/v.mp4", "frames": 124}}));
    assert_eq!((h["authorization"].as_str(), h["x-request-id"].as_str(), h["charset"].as_str()), ("secret-key", "job-1", "utf-8"));
    // ping: $RUNPOD_POD_ID → worker id, the running job's id, runpod_version
    let pings = f.state.lock().unwrap().pings.clone();
    assert!(!pings.is_empty());
    assert!(pings.iter().all(|p| p.0 == "/v2/ep1/ping/w42" && p.2["authorization"] == "secret-key"));
    assert!(pings.iter().any(|p| p.1.get("job_id") == Some(&vec!["job-1".to_string()])), "{pings:?}");
    assert!(pings.iter().all(|p| p.1["runpod_version"] == vec![VERSION.to_string()]));
    assert_eq!((c.progress_sent.load(std::sync::atomic::Ordering::SeqCst), c.progress_failed.load(std::sync::atomic::Ordering::SeqCst)), (2, 0));
}

#[test]
fn handler_error_posts_failed_with_the_error_lifted_out() {
    let f = FakeRunpod::start();
    f.state.lock().unwrap().jobs.push(json!({"id": "bad-1", "input": {"frames": 100}}));
    let c = client(&f, None, None);
    rp::serve(&|_| json!({"error": "frames 100 is not on the 17n+5 grid", "where": "validate"}), &c, Arc::new(AtomicBool::new(false)), Some(1), 0.0);
    let done = f.done_posts();
    assert_eq!(done.len(), 1);
    let (_, qq, h, body) = &done[0];
    assert_eq!(body, &json!({"error": "frames 100 is not on the 17n+5 grid", "output": {"where": "validate"}}));
    assert_eq!(qq["isStream"], vec!["false".to_string()]);
    assert_eq!(h["x-request-id"], "bad-1");
}

#[test]
fn handler_panic_is_a_failed_job_with_the_sdk_report() {
    let f = FakeRunpod::start();
    f.state.lock().unwrap().jobs.push(json!({"id": "boom-1", "input": {}}));
    let logs: Logs = Arc::new(Mutex::new(Vec::new()));
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    rp::serve(&|_| panic!("engine died"), &client(&f, Some(logs.clone()), None), Arc::new(AtomicBool::new(false)), Some(1), 0.0);
    std::panic::set_hook(prev);
    let done = f.done_posts();
    assert_eq!(done.len(), 1);
    let body = &done[0].3;
    assert_eq!(body.as_object().unwrap().keys().collect::<Vec<_>>(), ["error"]);
    let report: Value = serde_json::from_str(body["error"].as_str().unwrap()).unwrap();
    assert_eq!(report["error_message"], "engine died");
    assert!(report["error_traceback"].as_str().unwrap().contains("Traceback"));
    assert_eq!((report["worker_id"].as_str(), report["hostname"].as_str(), report["runpod_version"].as_str()), (Some("w42"), Some("host-1"), Some(VERSION)));
    assert!(logs.lock().unwrap().iter().any(|m| m.contains("boom-1 failed")));
}

#[test]
fn done_post_retries_with_backoff() {
    let f = FakeRunpod::start();
    f.state.lock().unwrap().jobs.push(json!({"id": "r-1", "input": {}}));
    f.state.lock().unwrap().fail_done = 2;
    let (sleeps, logs): (Arc<Mutex<Vec<f64>>>, Logs) = (Arc::new(Mutex::new(Vec::new())), Arc::new(Mutex::new(Vec::new())));
    let c = client(&f, Some(logs.clone()), Some(sleeps.clone()));
    rp::serve(&|_| json!({"ok": 1}), &c, Arc::new(AtomicBool::new(false)), Some(1), 0.0);
    let posts = f.done_posts();
    assert_eq!(posts.len(), 3);
    assert!(posts.iter().all(|p| p.3 == json!({"output": {"ok": 1}}) && p.2["x-request-id"] == "r-1"));
    assert_eq!(*sleeps.lock().unwrap(), vec![1.0, 1.0]);
    let l = logs.lock().unwrap();
    assert_eq!(l.iter().filter(|m| m.contains("attempt 1/3")).count(), 1);
    assert_eq!(l.iter().filter(|m| m.contains("attempt 2/3")).count(), 1);
    assert!(l.iter().any(|m| m.contains("r-1 done") && m.contains("result sent")));
}

#[test]
fn done_post_gives_up_after_three_attempts() {
    let f = FakeRunpod::start();
    f.state.lock().unwrap().jobs.push(json!({"id": "r-2", "input": {}}));
    f.state.lock().unwrap().fail_done = 5;
    let (sleeps, logs): (Arc<Mutex<Vec<f64>>>, Logs) = (Arc::new(Mutex::new(Vec::new())), Arc::new(Mutex::new(Vec::new())));
    let c = client(&f, Some(logs.clone()), Some(sleeps.clone()));
    assert_eq!(rp::serve(&|_| json!({"ok": 1}), &c, Arc::new(AtomicBool::new(false)), Some(1), 0.0), 1);
    assert_eq!(f.done_posts().len(), 3);
    assert_eq!(*sleeps.lock().unwrap(), vec![1.0, 1.0]);
    assert!(logs.lock().unwrap().iter().any(|m| m.contains("result NOT accepted")));
    assert_eq!(f.state.lock().unwrap().fail_done, 2); // no fourth attempt
}

#[test]
fn take_backs_off_on_429_and_errors() {
    let f = FakeRunpod::start();
    f.state.lock().unwrap().take_statuses = vec![429, 500, 500];
    f.state.lock().unwrap().jobs.push(json!({"id": "t-1", "input": {}}));
    let (sleeps, logs): (Arc<Mutex<Vec<f64>>>, Logs) = (Arc::new(Mutex::new(Vec::new())), Arc::new(Mutex::new(Vec::new())));
    let c = client(&f, Some(logs.clone()), Some(sleeps.clone()));
    rp::serve(&|_| json!({"ok": 1}), &c, Arc::new(AtomicBool::new(false)), Some(1), 0.0);
    assert_eq!(*sleeps.lock().unwrap(), vec![5.0, 1.0, 2.0]);
    let st = f.state.lock().unwrap();
    assert_eq!((st.takes.len(), st.taken.clone()), (4, vec!["t-1".to_string()]));
    let l = logs.lock().unwrap();
    assert!(l.iter().any(|m| m.contains("429")) && l.iter().any(|m| m.contains("HTTP 500")));
}

#[test]
fn take_ignores_malformed_jobs_and_lists() {
    let f = FakeRunpod::start();
    f.state.lock().unwrap().jobs.push(json!([{"id": "l-1", "input": {"a": 1}}, {"input": {"no": "id"}}, "junk"]));
    let logs: Logs = Arc::new(Mutex::new(Vec::new()));
    let c = client(&f, Some(logs.clone()), None);
    let jobs = c.take().unwrap();
    assert_eq!(jobs, vec![json!({"id": "l-1", "input": {"a": 1}})]);
    assert_eq!(logs.lock().unwrap().iter().filter(|m| m.contains("without id")).count(), 2);
}

#[test]
fn stop_ends_the_loop_after_the_current_job() {
    let f = FakeRunpod::start();
    f.state.lock().unwrap().jobs.extend([json!({"id": "s-1", "input": {}}), json!({"id": "s-2", "input": {}})]);
    let stop = Arc::new(AtomicBool::new(false));
    let c = client(&f, None, None);
    let s2 = stop.clone();
    let handler = move |job: &Value| {
        s2.store(true, std::sync::atomic::Ordering::SeqCst); // as SIGTERM would, mid-job
        json!({"done": job["id"]})
    };
    assert_eq!(rp::serve(&handler, &c, stop, None, 0.0), 1);
    let st = f.state.lock().unwrap();
    assert_eq!((st.taken.clone(), st.jobs.clone()), (vec!["s-1".to_string()], vec![json!({"id": "s-2", "input": {}})]));
    drop(st);
    assert_eq!(f.done_posts()[0].3, json!({"output": {"done": "s-1"}}));
}

#[test]
fn job_in_progress_flag_follows_the_running_job() {
    let f = FakeRunpod::start();
    let c = client(&f, None, None);
    assert!(f.env().take_url(c.current_job().is_some()).contains("job_in_progress=0"));
    c.set_current_job(Some("x".into()));
    assert!(f.env().take_url(c.current_job().is_some()).contains("job_in_progress=1"));
}
