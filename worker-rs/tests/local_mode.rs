//! `studio-worker --test-input job.json` without RunPod: one job through
//! the real `Worker` (validation only), no SDK.

use serde_json::{json, Value};
use std::process::Command;

#[test]
fn binary_runs_a_local_job() {
    let dir = fvstudio::tempdir("local");
    let f = dir.join("job.json");
    std::fs::write(&f, r#"{"input": {"op": "nope"}}"#).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_studio-worker"))
        .arg("--test-input").arg(&f)
        .env_clear()
        .env("PATH", std::env::var("PATH").unwrap_or_default())
        .env("STUDIO_CATALOG", std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../catalog/catalog.json"))
        .env("STUDIO_R2_LOCAL_DIR", dir.join("r2"))
        .env("STUDIO_VOLUME", dir.join("vol"))
        .env("STUDIO_JOBS_DIR", dir.join("jobs"))
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1), "{}", String::from_utf8_lossy(&out.stderr));
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["output"]["where"], "validate");
    assert!(v["error"].as_str().unwrap().contains("nope"));
    assert_eq!(String::from_utf8_lossy(&out.stdout), "{\n \"output\": {\n  \"where\": \"validate\"\n },\n \"error\": \"unknown op 'nope'\"\n}\n");
    // An info op succeeds (exit 0) and prints the jobs.md keys.
    std::fs::write(&f, r#"{"id": "i7", "input": {"op": "info"}}"#).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_studio-worker")).arg("--test-input").arg(&f).env_clear()
        .env("PATH", std::env::var("PATH").unwrap_or_default())
        .env("STUDIO_CATALOG", std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../catalog/catalog.json"))
        .env("STUDIO_VOLUME", dir.join("vol")).env("STUDIO_JOBS_DIR", dir.join("jobs")).output().unwrap();
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["output"]["op"], "info");
    assert_eq!(v["output"]["catalog_version"], json!(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("running") && String::from_utf8_lossy(&out.stderr).contains("as job i7"));
    // A missing file is exit 2.
    let out = Command::new(env!("CARGO_BIN_EXE_studio-worker")).arg("--test-input").arg(dir.join("missing.json")).env_clear()
        .env("PATH", std::env::var("PATH").unwrap_or_default())
        .env("STUDIO_CATALOG", std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../catalog/catalog.json")).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
}
