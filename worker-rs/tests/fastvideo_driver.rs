//! The fastvideo driver against a fake fv-serve HTTP server in-process.

mod common;

use common::{catalog, FakeFvServe};
use fvstudio::catalog::Catalog;
use fvstudio::engines::{fastvideo, Ctx, EngineEnv};
use fvstudio::flags::EngineSettings;
use fvstudio::progress::Reporter;
use fvstudio::r2::{LocalStore, ObjectStore};
use fvstudio::sidecar::SidecarManager;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

struct Env {
    settings: EngineSettings,
    sidecars: SidecarManager,
    ports: HashMap<String, u16>,
    env: EngineEnv,
}

fn env_for(port: u16) -> Env {
    Env {
        settings: EngineSettings::default(),
        sidecars: SidecarManager::default(),
        ports: [("fastvideo".to_string(), port)].into_iter().collect(),
        env: [("fastvideo".to_string(), HashMap::new())]
            .into_iter()
            .collect(),
    }
}

fn ctx<'a>(
    e: &'a Env,
    c: &'a Catalog,
    dir: std::path::PathBuf,
    store: Option<Arc<dyn ObjectStore>>,
    bucket: Option<&str>,
    rep: &'a Reporter<'a>,
) -> Ctx<'a> {
    Ctx {
        settings: &e.settings,
        knobs: &c.knobs,
        sidecars: &e.sidecars,
        ports: &e.ports,
        env: &e.env,
        job_dir: dir.join("job"),
        store,
        bucket: bucket.map(str::to_string),
        log: fvstudio::null_log(),
        report: rep,
        fv_serve_bin: "fv-serve".into(),
        fv_config: fastvideo::fv_config_path().to_string_lossy().into_owned(),
        fv_state_dir: dir.join("state").to_string_lossy().into_owned(),
        fv_weights: "/w".into(),
        fv_fake: true,
        models: &[],
    }
}

#[test]
fn t2v_job_runs_to_an_mp4() {
    let c = catalog();
    let dir = fvstudio::tempdir("fvdrv");
    let fake = FakeFvServe::start(dir.join("files"), 4, 0.6, None);
    let e = env_for(fake.fake.port);
    let sent = Mutex::new(Vec::new());
    let send = |v: &Value| sent.lock().unwrap().push(v.clone());
    let rep = Reporter::with_interval(Some(&send), "fastvideo", "x", 0.0);
    let cx = ctx(&e, &c, dir.clone(), None, None, &rep);
    let job = c.validate(&json!({"engine": "fastvideo", "model": "fastwan22-ti2v-5b", "prompt": "a fox", "width": 832, "height": 480, "frames": 49, "seed": 11, "options": {"steps": 6}})).unwrap();
    let res = fastvideo::run(&cx, &job).unwrap();
    assert!(res.video_mp4.as_ref().unwrap().metadata().unwrap().len() > 100);
    assert_eq!(
        (res.width, res.height, res.frames, res.fps, res.seed),
        (832, 480, 49, 24, 11)
    );
    assert_eq!(
        (
            res.timings.denoise_s,
            res.timings.encode_s,
            res.timings.decode_s
        ),
        (1.5, 0.2, 0.3)
    );
    assert_eq!(
        fake.requests().last().unwrap(),
        &json!({"model": "fastwan22-ti2v-5b", "prompt": "a fox", "seed": 11, "size": "832x480", "num_frames": 49, "steps": 6})
    );
    let stages: Vec<String> = sent
        .lock()
        .unwrap()
        .iter()
        .map(|r| r["stage"].as_str().unwrap().to_string())
        .collect();
    assert!(stages.contains(&"denoising".to_string()) && stages.last().unwrap() == "decoding");
    assert!(sent.lock().unwrap().iter().any(|r| r
        .get("detail")
        .and_then(|d| d.as_str())
        .map(|d| d.ends_with('%'))
        .unwrap_or(false)));
    assert!(res.command.contains("curl") && res.command.contains("fv/v1/jobs"));
}

#[test]
fn media_become_urls() {
    let c = catalog();
    let dir = fvstudio::tempdir("fvmedia");
    let fake = FakeFvServe::start(dir.join("files"), 2, 0.2, None);
    let e = env_for(fake.fake.port);
    let store = LocalStore::new(dir.join("r2"));
    std::fs::write(dir.join("up.png"), b"\x89PNG fake").unwrap();
    store
        .put_file("studio", "uploads/u1.png", &dir.join("up.png"), None)
        .unwrap();
    let rep = Reporter::with_interval(None, "fastvideo", "x", 0.0);
    let cx = ctx(
        &e,
        &c,
        dir.clone(),
        Some(Arc::new(store)),
        Some("studio"),
        &rep,
    );
    let job = c.validate(&json!({"engine": "fastvideo", "model": "fasth3-4step-vsa", "task": "fl2v", "prompt": "p", "width": 832, "height": 480,
                                 "inputs": {"first_image": {"r2": "uploads/u1.png"}, "last_image": {"b64": "iVBORw0KGgo=", "mime": "image/png"}}})).unwrap();
    fastvideo::run(&cx, &job).unwrap();
    let body = fake.requests().last().unwrap().clone();
    let img = body["image_url"].as_str().unwrap();
    assert!(img.starts_with("file://") && img.contains("uploads/u1.png"));
    assert_eq!(body["last_image_url"], "data:image/png;base64,iVBORw0KGgo=");
}

#[test]
fn failed_and_refused_jobs_report_engine() {
    let c = catalog();
    let dir = fvstudio::tempdir("fvfail");
    let fake = FakeFvServe::start(dir.join("files"), 2, 0.2, Some("sol-h3"));
    let e = env_for(fake.fake.port);
    let rep = Reporter::with_interval(None, "fastvideo", "x", 0.0);
    let cx = ctx(&e, &c, dir.clone(), None, None, &rep);
    let job = c
        .validate(&json!({"engine": "fastvideo", "model": "sol-h3", "prompt": "p"}))
        .unwrap();
    let err = fastvideo::run(&cx, &job).unwrap_err();
    assert!(
        err.where_ == "engine" && err.message.contains("refuses"),
        "{}",
        err.message
    );
    // A body fv-serve rejects (an unknown field) is an engine refusal naming the field.
    let r = fvstudio::httpc::post_json(
        &format!("{}/fv/v1/jobs", fake.fake.base()),
        &json!({"model": "sol-h3", "prompt": "p", "bogus": 1}),
        10.0,
    );
    match r {
        Err(fvstudio::httpc::Error::Status(e)) => {
            assert!(e.status == 400 && e.body.contains("bogus"))
        }
        other => panic!("{other:?}"),
    }
}
