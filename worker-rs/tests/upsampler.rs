//! Prompt expansion against a fake llama-server: the request shape, the
//! vision part, the fallbacks (500 / timeout / connection refused / empty)
//! and the readiness probe. The pure parts are tested in `src/upsampler.rs`.

mod common;

use common::FakeLlama;
use fvstudio::catalog::b64_encode;
use fvstudio::upsampler::{llama_probe, Upsampler, ALIAS, MAX_IMAGE_DATA_URL};
use serde_json::json;
use std::sync::{Arc, Mutex};

#[test]
fn expand_sends_the_family_prompt_and_cleans_the_answer() {
    let f = FakeLlama::start("\"A red fox trots across fresh snow at dawn.\"\n", false, None, 0.0);
    let mut u = Upsampler::new(&f.base());
    let out = u.expand("a fox", "fastwan", "balanced", None, None, "image/jpeg", 60.0);
    assert_eq!(out, "A red fox trots across fresh snow at dawn.");
    assert!(u.last_error.is_none() && u.calls == 1 && u.failures == 0);
    let req = f.last();
    assert_eq!((req["model"].as_str(), req["temperature"].as_f64(), req["max_tokens"].as_u64()), (Some(ALIAS), Some(0.6), Some(512)));
    assert!(f.system_of().contains("makes no sound") && f.system_of().contains("60 to 100 words"));
    assert_eq!(f.user_of(), json!("a fox"));
    // Quality mode, H3 family through the catalog's fv-h3 id.
    u.expand("a fox", "fv-h3", "quality", None, None, "image/jpeg", 60.0);
    assert!(f.system_of().contains("[Shot 1]") && f.last()["max_tokens"] == 1024);
    // Off / empty: no call.
    assert_eq!(u.expand("a fox", "h3", "off", None, None, "image/jpeg", 60.0), "a fox");
    assert_eq!(u.expand("  ", "h3", "quality", None, None, "image/jpeg", 60.0), "  ");
    assert_eq!(f.state.lock().unwrap().requests.len(), 2);
}

#[test]
fn h3_answer_keeps_sections_through_expand() {
    let ans = "integrated_multimodal_description: A fox runs.\noverall_soundscape: Wind.\nnon_diegetic_music: None.";
    let f = FakeLlama::start(ans, false, None, 0.0);
    assert_eq!(Upsampler::new(&f.base()).expand("a fox", "h3", "balanced", None, None, "image/jpeg", 60.0), ans);
}

#[test]
fn vision_part_when_an_image_is_given_and_the_projector_is_loaded() {
    let png = b64_encode(&[b"\x89PNG\r\n\x1a\n".as_slice(), &[0u8; 64]].concat());
    let f = FakeLlama::start("A red fox.", false, None, 0.0);
    let mut u = Upsampler::new(&f.base());
    u.expand("make it run", "ltx", "balanced", None, Some(&png), "image/png", 60.0);
    let user = f.user_of();
    assert_eq!(user[0], json!({"type": "text", "text": "make it run"}));
    assert_eq!(user[1]["type"], "image_url");
    assert_eq!(user[1]["image_url"]["url"], format!("data:image/png;base64,{png}"));
    assert!(f.system_of().contains("attached image") && !f.system_of().contains("cannot see"));
    // Text-only server: the image stays out, the system prompt says it is unseen.
    let mut t = Upsampler::new(&f.base());
    t.vision = false;
    t.expand("make it run", "ltx", "balanced", None, Some(&png), "image/jpeg", 60.0);
    assert_eq!(f.user_of(), json!("make it run"));
    assert!(f.system_of().contains("image you cannot see"));
    // No image: neither line.
    Upsampler::new(&f.base()).expand("make it run", "ltx", "balanced", None, None, "image/jpeg", 60.0);
    assert!(!f.system_of().contains("first frame"));
    // An image over the data-URL cap is not attached.
    let big = "A".repeat(MAX_IMAGE_DATA_URL + 1);
    Upsampler::new(&f.base()).expand("x", "ltx", "balanced", None, Some(&big), "image/jpeg", 60.0);
    assert_eq!(f.user_of(), json!("x"));
    assert!(f.system_of().contains("image you cannot see"));
}

#[test]
fn negative_prompt_reaches_the_system_prompt_only() {
    let f = FakeLlama::start("A fox.", false, None, 0.0);
    Upsampler::new(&f.base()).expand("a fox", "wan", "balanced", Some("blurry"), None, "image/jpeg", 60.0);
    assert!(f.system_of().contains("The user excludes: blurry."));
    assert_eq!(f.user_of(), json!("a fox"));
}

#[test]
fn fallbacks_leave_the_original_prompt() {
    let f = FakeLlama::start("A fox.", false, Some(500), 0.0);
    let logs = Arc::new(Mutex::new(Vec::new()));
    let l = logs.clone();
    let mut u = Upsampler::new(&f.base());
    u.log = Arc::new(move |s: &str| l.lock().unwrap().push(s.to_string()));
    assert_eq!(u.expand("a fox", "wan", "balanced", None, None, "image/jpeg", 60.0), "a fox");
    assert!(u.last_error.as_deref().unwrap().starts_with("HTTP 500") && u.failures == 1);
    let l = logs.lock().unwrap();
    assert!(!l.is_empty() && l[0].contains("prompt expansion failed") && l[0].contains("original prompt"));
    drop(l);
    // timeout
    let f = FakeLlama::start("A fox.", false, None, 1.5);
    let mut u = Upsampler::new(&f.base());
    assert_eq!(u.expand("a fox", "h3", "quality", None, None, "image/jpeg", 0.3), "a fox");
    let e = u.last_error.clone().unwrap().to_lowercase();
    assert!(e.contains("timeout") || e.contains("timed out"), "{e}");
    // connection refused
    let port = { let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap(); l.local_addr().unwrap().port() };
    let mut u = Upsampler::new(&format!("http://127.0.0.1:{port}"));
    assert_eq!(u.expand("a fox", "ltx", "balanced", None, None, "image/jpeg", 10.0), "a fox");
    let e = u.last_error.clone().unwrap().to_lowercase();
    assert!(e.contains("refused") || e.contains("connect"), "{e}");
    // empty or garbage answers
    let f = FakeLlama::start("   ", false, None, 0.0);
    let mut u = Upsampler::new(&f.base());
    assert_eq!(u.expand("a fox", "wan", "balanced", None, None, "image/jpeg", 60.0), "a fox");
    assert!(u.last_error.as_deref().unwrap().contains("nothing left"), "{:?}", u.last_error);
    let f = FakeLlama::start("\"\"", false, None, 0.0);
    assert_eq!(Upsampler::new(&f.base()).expand("a fox", "wan", "balanced", None, None, "image/jpeg", 60.0), "a fox");
}

#[test]
fn llama_probe_health_then_one_token() {
    let f = FakeLlama::start("A fox.", true, None, 0.0);
    let probe = llama_probe(&f.base(), ALIAS);
    assert_eq!(probe(), (false, None));
    assert!(f.state.lock().unwrap().requests.is_empty());
    f.state.lock().unwrap().loading = false;
    assert_eq!(probe(), (true, None));
    let last = f.last();
    assert!(last["max_tokens"] == 1 && last["model"] == ALIAS);
    assert_eq!(llama_probe("http://127.0.0.1:1", ALIAS)(), (false, None));
}
