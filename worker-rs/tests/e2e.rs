//! End to end against the real engines in `--synthetic` mode on the CPU, a
//! recording progress sink in place of the RunPod loop, a fake fv-serve and
//! a directory as the object store (the port of `worker/tests/test_e2e.py`):
//!
//!     cargo build --release -p h3-cli        (in h3-rs,      $H3_RS)
//!     cargo build --release -p fastwan-cli   (in fastwan-rs, $FASTWAN_RS)
//!     cargo test --release -- --ignored e2e
//!
//! The scenarios run in order inside one test (the sidecar-reuse and crash
//! steps depend on the h3 serve started by the first job), so `cargo test`'s
//! parallelism does not interleave them. A missing binary is built with
//! `cargo build --release` first, as the Python test does.

mod common;

use common::FakeFvServe;
use fvstudio::catalog::b64_encode;
use fvstudio::worker::Worker;
use fvstudio::EnvMap;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

fn build(repo: &Path, crate_: &str, binary: &str) -> PathBuf {
    let var = format!(
        "CARGO_TARGET_DIR_{}",
        crate_.to_uppercase().replace('-', "_")
    );
    let target = std::env::var(&var)
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo.join("target"));
    let out = target.join("release").join(binary);
    let t0 = Instant::now();
    let st = std::process::Command::new("cargo")
        .args(["build", "--release", "-p", crate_])
        .current_dir(repo)
        .env("CARGO_TARGET_DIR", &target)
        .env_remove("CARGO_BUILD_JOBS")
        .status()
        .unwrap();
    assert!(st.success(), "cargo build -p {crate_} failed");
    println!(
        "built {binary} in {:.0} s: {}",
        t0.elapsed().as_secs_f64(),
        out.display()
    );
    assert!(out.is_file());
    out
}

struct Recorder(Mutex<Vec<(String, Value)>>);

fn check_mp4(url: &str, frames: Option<i64>, audio: bool) {
    assert!(url.starts_with("file://"));
    let path = PathBuf::from(url["file://".len()..].split('?').next().unwrap());
    assert!(path.is_file() && path.metadata().unwrap().len() > 1000);
    if fvstudio::py::which("ffprobe").is_some() {
        let p = fvstudio::mux::probe(&path, None).unwrap();
        if let Some(f) = frames {
            assert_eq!(p.frames, f);
        }
        assert_eq!(p.audio, audio);
        println!("  ffprobe: {p:?}");
    } else {
        println!("  (no ffprobe: mp4 structure not checked)");
    }
}

fn png(w: usize, h: usize, rgb: [u8; 3]) -> Vec<u8> {
    use std::io::Write;
    let mut raw = Vec::new();
    for _ in 0..h {
        raw.push(0u8);
        for _ in 0..w {
            raw.extend_from_slice(&rgb);
        }
    }
    // zlib stored (uncompressed) blocks + adler32
    let mut z = vec![0x78u8, 0x01];
    for chunk in raw.chunks(65535) {
        let last = chunk.as_ptr_range().end == raw.as_ptr_range().end;
        z.push(if last { 1 } else { 0 });
        z.extend_from_slice(&(chunk.len() as u16).to_le_bytes());
        z.extend_from_slice(&(!(chunk.len() as u16)).to_le_bytes());
        z.extend_from_slice(chunk);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &x in &raw {
        a = (a + x as u32) % 65521;
        b = (b + a) % 65521;
    }
    z.extend_from_slice(&((b << 16) | a).to_be_bytes());
    let crc = |data: &[u8]| -> u32 {
        let mut c = 0xffff_ffffu32;
        for &x in data {
            c ^= x as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 {
                    0xedb8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
        }
        !c
    };
    let chunk = |t: &[u8], d: &[u8]| -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&(d.len() as u32).to_be_bytes());
        v.extend_from_slice(t);
        v.extend_from_slice(d);
        v.extend_from_slice(&crc(&[t, d].concat()).to_be_bytes());
        v
    };
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&(w as u32).to_be_bytes());
    ihdr.extend_from_slice(&(h as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    let mut out = Vec::new();
    out.write_all(b"\x89PNG\r\n\x1a\n").unwrap();
    out.extend(chunk(b"IHDR", &ihdr));
    out.extend(chunk(b"IDAT", &z));
    out.extend(chunk(b"IEND", b""));
    out
}

#[test]
#[ignore]
fn e2e() {
    let h3_rs = PathBuf::from(std::env::var("H3_RS").unwrap_or_else(|_| "/home/user/h3-rs".into()));
    let fastwan_rs = PathBuf::from(
        std::env::var("FASTWAN_RS").unwrap_or_else(|_| "/home/user/fastwan-rs".into()),
    );
    let h3 = build(&h3_rs, "h3-cli", "h3");
    let fastwan = build(&fastwan_rs, "fastwan-cli", "fastwan");

    let root = fvstudio::tempdir("studio-e2e");
    let bins = root.join("bin");
    std::fs::create_dir_all(&bins).unwrap();
    std::os::unix::fs::symlink(&h3, bins.join("h3")).unwrap();
    std::os::unix::fs::symlink(&fastwan, bins.join("fastwan")).unwrap();
    let fake = FakeFvServe::start(root.join("fvfiles"), 4, 0.6, None);
    let s = |p: PathBuf| p.to_string_lossy().into_owned();
    let mut env: EnvMap = fvstudio::process_env();
    for (k, v) in [
        (
            "STUDIO_CATALOG",
            s(Path::new(env!("CARGO_MANIFEST_DIR")).join("../catalog/catalog.json")),
        ),
        ("STUDIO_BIN_DIR", s(bins.clone())),
        ("STUDIO_BACKEND", "cpu".into()),
        ("STUDIO_SYNTHETIC", "1".into()),
        ("STUDIO_FV_FAKE", "1".into()),
        ("STUDIO_FV_PORT", fake.fake.port.to_string()),
        ("STUDIO_H3_PORT", "7891".into()),
        ("STUDIO_FASTWAN_PORT", "7892".into()),
        ("STUDIO_JOBS_DIR", s(root.join("jobs"))),
        ("STUDIO_R2_LOCAL_DIR", s(root.join("r2"))),
        ("STUDIO_R2_BUCKET", "studio".into()),
        ("STUDIO_VOLUME", s(root.join("volume"))),
        ("STUDIO_VOLUME_WEIGHTS", s(root.join("volume/weights"))),
        ("STUDIO_VOLUME_EXTRAS", s(root.join("volume/h3-rs"))),
        ("STUDIO_STAGE_DIR", s(root.join("stage"))),
        ("STUDIO_LINK_ROOT", s(root.join("w"))),
        ("STUDIO_STAGE_DRY_RUN", "1".into()),
        ("STUDIO_IMAGE", "e2e-test".into()),
        ("STUDIO_KEEP_JOB_DIRS", "1".into()),
    ] {
        env.insert(k.to_string(), v);
    }
    // Fake volume trees so staging is a no-op link.
    for t in [
        "h3-8step",
        "h3-8step-mxfp8",
        "fastwan",
        "auxiliary/tae",
        "fastwan22-ti2v-5b",
        "h3-base/transformer",
        "h3-base/text_encoder",
        "h3-base/tokenizer",
        "h3-base/vae",
        "h3-base/audio_vae",
        "h3-ref2va",
    ] {
        std::fs::create_dir_all(root.join("volume/weights").join(t)).unwrap();
    }
    for f in [
        "auxiliary/tae/taew2_1.safetensors",
        "auxiliary/tae/taew2_2.safetensors",
    ] {
        std::fs::write(root.join("volume/weights").join(f), [0u8; 16]).unwrap();
    }
    for t in ["tables", "plug4"] {
        std::fs::create_dir_all(root.join("volume/h3-rs").join(t)).unwrap();
    }
    std::fs::create_dir_all(root.join("stage/fastwan")).unwrap();
    std::fs::write(root.join("stage/fastwan/.complete"), "{}\n").unwrap();

    let w = Worker::from_env_map(&env, fvstudio::stderr_log()).unwrap();
    let rec = Recorder(Mutex::new(Vec::new()));
    let run = |jid: &str, inp: Value| -> Value {
        let t0 = Instant::now();
        let progress = |d: &Value| rec.0.lock().unwrap().push((jid.to_string(), d.clone()));
        let out = w.handle(&json!({"id": jid, "input": inp}), Some(&progress));
        let short: Value = out
            .as_object()
            .map(|o| {
                Value::Object(
                    o.iter()
                        .filter(|(k, _)| *k != "timings" && *k != "command")
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect(),
                )
            })
            .unwrap_or(out.clone());
        println!(
            "\n{jid}: {:.1} s → {}",
            t0.elapsed().as_secs_f64(),
            short.to_string().chars().take(600).collect::<String>()
        );
        if let Some(t) = out.get("timings") {
            println!("  timings {t}\n  command {}", out["command"]);
        }
        out
    };
    let stages = || -> Vec<String> {
        rec.0
            .lock()
            .unwrap()
            .iter()
            .map(|(_, d)| d["stage"].as_str().unwrap().to_string())
            .collect()
    };
    let has_ffmpeg = fvstudio::py::which("ffmpeg").is_some();

    // ---- info
    let out = run("info-1", json!({"op": "info"}));
    assert_eq!(out["catalog_version"], json!(1));
    assert!(out["versions"].get("h3").is_some() && out["image"] == "e2e-test");
    assert!(out["trees"]["weights/h3-8step"]["volume"] == true);
    assert!(
        out["versions"]["h3"].as_str().unwrap().starts_with("h3 (")
            && out["versions"]["fastwan"]
                .as_str()
                .unwrap()
                .starts_with("fastwan ("),
        "{}",
        out["versions"]
    );
    assert_eq!(out["upsampler"]["resident"], false);

    // ---- h3 synthetic generate
    rec.0.lock().unwrap().clear();
    let out = run(
        "h3-1",
        json!({"engine": "h3", "model": "fasth3-v2", "prompt": "a fox", "width": 480, "height": 272, "frames": 124, "seed": 7, "options": {"pv8": true, "tables": true}, "client": {"job": "c1"}}),
    );
    assert!(out.get("error").is_none(), "{out}");
    assert_eq!(
        (
            out["engine"].as_str(),
            out["model"].as_str(),
            out["seed"].as_i64(),
            out["frames"].as_i64(),
            out["fps"].as_i64(),
            out["audio"].as_bool()
        ),
        (
            Some("h3"),
            Some("fasth3-v2"),
            Some(7),
            Some(124),
            Some(24),
            Some(true)
        )
    );
    assert!(out["width"] == 480 && out["height"] == 272 && out["client"] == json!({"job": "c1"}));
    assert!(
        out["command"]
            .as_str()
            .unwrap()
            .starts_with("h3 generate --backend cpu --variant fasth3-v2 --task t2va"),
        "{}",
        out["command"]
    );
    assert_eq!(out["timings"]["steps_s"].as_array().unwrap().len(), 8);
    assert!(
        out["worker"]["image"] == "e2e-test" && out["expires_at"].as_str().unwrap().ends_with('Z')
    );
    assert!(out["expanded_prompt"].is_null());
    let st = stages();
    for s in [
        "staging",
        "loading",
        "denoising",
        "decoding",
        "muxing",
        "uploading",
        "done",
    ] {
        assert!(st.contains(&s.to_string()), "{st:?}");
    }
    assert!(rec
        .0
        .lock()
        .unwrap()
        .iter()
        .all(|(_, d)| ["stage", "elapsed_s", "engine", "model"]
            .iter()
            .all(|k| d.get(*k).is_some())));
    if has_ffmpeg {
        check_mp4(out["video_url"].as_str().unwrap(), Some(124), true);
        assert!(out["audio_url"].is_string() && out["thumbnail_url"].is_string());
    } else {
        println!("ffmpeg missing: the mux assertion is skipped");
    }

    // ---- FL2VA through the resident serve (h3 serve takes first_image / last_image per request)
    rec.0.lock().unwrap().clear();
    let first = json!({"b64": b64_encode(&png(64, 32, [200, 30, 30])), "mime": "image/png"});
    let last = json!({"b64": b64_encode(&png(64, 32, [30, 30, 200])), "mime": "image/png"});
    let out = run(
        "h3-fl2v",
        json!({"engine": "h3", "model": "h3-plug4", "task": "fl2v", "prompt": "a fox", "width": 480, "height": 272, "frames": 124, "seed": 3, "inputs": {"first_image": first, "last_image": last}}),
    );
    assert!(out.get("error").is_none(), "{out}");
    assert!(out["task"] == "fl2v" && out["frames"] == 124);
    let details: Vec<String> = rec
        .0
        .lock()
        .unwrap()
        .iter()
        .map(|(_, d)| {
            d.get("detail")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string()
        })
        .collect();
    assert!(
        !details.iter().any(|d| d.contains("per-job CLI")),
        "{details:?}"
    );
    assert!(
        out["command"].as_str().unwrap().contains("--task fl2va")
            && out["command"].as_str().unwrap().contains("--first-image")
    );

    // ---- the sidecar is reused and survives a crash
    let sc = w.sidecars.get("h3").unwrap();
    assert!(sc.alive());
    let pid = sc.pid();
    let out = run(
        "h3-2",
        json!({"engine": "h3", "model": "fasth3-v2", "prompt": "a cat", "width": 480, "height": 272, "frames": 124}),
    );
    assert!(out.get("error").is_none() && w.sidecars.get("h3").unwrap().pid() == pid); // reused
    sc.kill_now();
    let out = run(
        "h3-3",
        json!({"engine": "h3", "model": "fasth3-v2", "prompt": "a dog", "width": 480, "height": 272, "frames": 124}),
    );
    assert!(out.get("error").is_none() && w.sidecars.get("h3").unwrap().pid() != pid);

    // ---- a crash mid-job fails with where=engine and the log tail
    let sc = w.sidecars.get("h3").unwrap();
    assert!(sc.alive());
    let killer = sc.clone();
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_secs(3));
        killer.kill_now();
    });
    let out = run(
        "h3-4",
        json!({"engine": "h3", "model": "fasth3-v2", "prompt": "a crash", "width": 480, "height": 272, "frames": 124}),
    );
    assert!(
        out["where"] == "engine" && out["error"].as_str().unwrap().contains("died"),
        "{out}"
    );
    assert!(out["log_tail"]
        .as_array()
        .map(|a| a.len() <= 50)
        .unwrap_or(false));

    // ---- validation error shape
    let out = run(
        "bad-1",
        json!({"engine": "h3", "model": "fasth3-v2", "prompt": "x", "frames": 100}),
    );
    assert!(out["where"] == "validate" && out["error"].as_str().unwrap().contains("frames"));

    // ---- fastwan synthetic generate with upscale
    rec.0.lock().unwrap().clear();
    let out = run(
        "fw-1",
        json!({"engine": "fastwan", "model": "fastwan-1.3b", "prompt": "a fox", "width": 480, "height": 272, "frames": 17, "seed": 3, "upscale": 2}),
    );
    assert!(out.get("error").is_none(), "{out}");
    assert_eq!(
        (
            out["engine"].as_str(),
            out["frames"].as_i64(),
            out["fps"].as_i64(),
            out["audio"].as_bool()
        ),
        (Some("fastwan"), Some(17), Some(16), Some(false))
    );
    assert_eq!(
        out["upscale"],
        json!({"factor": 2, "width": 960, "height": 544})
    );
    assert!(out["command"]
        .as_str()
        .unwrap()
        .starts_with("fastwan generate --backend cpu"));
    assert!(out["command"].as_str().unwrap().contains("--upscale-x"));
    if has_ffmpeg {
        check_mp4(out["video_url"].as_str().unwrap(), Some(17), false);
    }
    assert!(out["audio_url"].is_null());

    // ---- fastvideo (fake fv-serve)
    let out = run(
        "fv-1",
        json!({"engine": "fastvideo", "model": "fastwan22-ti2v-5b", "prompt": "a fox", "width": 832, "height": 480, "frames": 49, "seed": 5, "options": {"steps": 8}}),
    );
    assert!(out.get("error").is_none(), "{out}");
    assert_eq!(
        (
            out["engine"].as_str(),
            out["model"].as_str(),
            out["seed"].as_i64(),
            out["frames"].as_i64()
        ),
        (
            Some("fastvideo"),
            Some("fastwan22-ti2v-5b"),
            Some(5),
            Some(49)
        )
    );
    let last = fake.requests().last().unwrap().clone();
    assert!(last["steps"] == 8 && last["size"] == "832x480");
    assert_eq!(out["timings"]["denoise_s"], json!(1.5));
    if has_ffmpeg {
        check_mp4(out["video_url"].as_str().unwrap(), None, true);
    }
    let job_json = root.join("r2/studio/jobs/fv-1/job.json");
    let meta: Value = serde_json::from_slice(&std::fs::read(&job_json).unwrap()).unwrap();
    assert_eq!(meta["output"]["model"], "fastwan22-ti2v-5b");
    assert_eq!(meta["input"]["prompt"], "a fox");

    // ---- warm
    let out = run(
        "warm-1",
        json!({"op": "warm", "engine": "fastwan", "model": "fastwan-1.3b"}),
    );
    assert!(
        out["op"] == "warm" && out["sidecars"]["fastwan"]["alive"] == true,
        "{out}"
    );

    w.sidecars.stop_all();
    println!("\ne2e: info, h3 t2v, h3 fl2v (resident serve), sidecar reuse + crash recovery, crash mid-job, validation, fastwan upscale, fastvideo fake, warm: all passed");
}
