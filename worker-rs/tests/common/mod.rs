//! In-process fakes for the integration tests, on `tiny_http`: a RunPod
//! endpoint, an S3-style object store, fv-serve's native jobs API and
//! llama-server's chat API. Each records what it saw.

#![allow(dead_code)]

use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use tiny_http::{Header, Method, Request, Response, Server};

pub fn catalog() -> fvstudio::catalog::Catalog {
    fvstudio::catalog::Catalog::load(Some(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../catalog/catalog.json"),
    ))
    .unwrap()
}

/// A served fake: the port and a stop handle.
pub struct Fake {
    pub port: u16,
    server: Arc<Server>,
    thread: Option<JoinHandle<()>>,
}

impl Fake {
    pub fn start<F: Fn(Request) + Send + Sync + 'static>(handle: F) -> Fake {
        let server = Arc::new(Server::http("127.0.0.1:0").unwrap());
        let port = server.server_addr().to_ip().unwrap().port();
        let s = server.clone();
        let thread = std::thread::spawn(move || {
            for req in s.incoming_requests() {
                handle(req);
            }
        });
        Fake {
            port,
            server,
            thread: Some(thread),
        }
    }

    pub fn base(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    pub fn stop(&mut self) {
        self.server.unblock();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for Fake {
    fn drop(&mut self) {
        self.stop();
    }
}

pub fn body_of(req: &mut Request) -> Vec<u8> {
    let mut v = Vec::new();
    let _ = req.as_reader().read_to_end(&mut v);
    v
}

pub fn headers_of(req: &Request) -> HashMap<String, String> {
    req.headers()
        .iter()
        .map(|h| {
            (
                h.field.as_str().to_string().to_lowercase(),
                h.value.as_str().to_string(),
            )
        })
        .collect()
}

pub fn split_url(url: &str) -> (String, HashMap<String, Vec<String>>) {
    let (path, query) = url.split_once('?').unwrap_or((url, ""));
    let mut q: HashMap<String, Vec<String>> = HashMap::new();
    if !query.is_empty() {
        for pair in query.split('&') {
            let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
            q.entry(unquote(k)).or_default().push(unquote(v));
        }
    }
    (path.to_string(), q)
}

/// `urllib.parse.unquote` + `parse_qs`'s `+` → space.
pub fn unquote(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(if b[i] == b'+' { b' ' } else { b[i] });
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

pub fn respond_json(req: Request, code: u16, body: &Value) {
    let data = serde_json::to_vec(body).unwrap();
    let _ = req.respond(
        Response::from_data(data)
            .with_status_code(code)
            .with_header(Header::from_bytes("Content-Type", "application/json").unwrap()),
    );
}

pub fn respond_bytes(req: Request, code: u16, body: Vec<u8>, ctype: &str) {
    // Always a Content-Length (tiny_http chunks bodies over 32 KiB by default; R2 does not).
    let r = Response::from_data(body)
        .with_status_code(code)
        .with_chunked_threshold(usize::MAX);
    let r = if ctype.is_empty() {
        r
    } else {
        r.with_header(Header::from_bytes("Content-Type", ctype).unwrap())
    };
    let _ = req.respond(r);
}

// ---- RunPod -----------------------------------------------------------------------

/// `GET /v2/ep/job-take/<worker>`: the next of `jobs` (200) or 204;
/// `take_statuses` forces statuses on the first takes. `POST
/// /v2/ep/job-done/<worker>/<job>`: recorded; the first `fail_done`
/// posts with a `status` other than IN_PROGRESS get a 500. `GET
/// /v2/ep/ping/<worker>`: recorded.
#[derive(Default)]
pub struct RunpodState {
    pub jobs: Vec<Value>,
    pub taken: Vec<String>,
    pub takes: Vec<(
        String,
        HashMap<String, Vec<String>>,
        HashMap<String, String>,
    )>,
    pub posts: Vec<(
        String,
        HashMap<String, Vec<String>>,
        HashMap<String, String>,
        Value,
    )>,
    pub pings: Vec<(
        String,
        HashMap<String, Vec<String>>,
        HashMap<String, String>,
    )>,
    pub take_statuses: Vec<u16>,
    pub fail_done: usize,
}

pub struct FakeRunpod {
    pub fake: Fake,
    pub state: Arc<Mutex<RunpodState>>,
}

impl FakeRunpod {
    pub fn start() -> FakeRunpod {
        let state = Arc::new(Mutex::new(RunpodState::default()));
        let st = state.clone();
        let fake = Fake::start(move |mut req: Request| {
            let (path, q) = split_url(req.url());
            let h = headers_of(&req);
            match req.method() {
                Method::Get if path.contains("/job-take/") => {
                    let mut s = st.lock().unwrap();
                    s.takes.push((path.clone(), q, h));
                    if !s.take_statuses.is_empty() {
                        let code = s.take_statuses.remove(0);
                        drop(s);
                        respond_bytes(req, code, Vec::new(), "");
                        return;
                    }
                    if !s.jobs.is_empty() {
                        let job = s.jobs.remove(0);
                        s.taken.push(
                            job.get("id")
                                .map(fvstudio::py::py_str)
                                .unwrap_or_else(|| "<list>".into()),
                        );
                        drop(s);
                        respond_json(req, 200, &job);
                        return;
                    }
                    drop(s);
                    respond_bytes(req, 204, Vec::new(), "");
                }
                Method::Get if path.contains("/ping/") => {
                    st.lock().unwrap().pings.push((path, q, h));
                    respond_json(req, 200, &json!({"status": "ok"}));
                }
                Method::Post => {
                    let body: Value =
                        serde_json::from_slice(&body_of(&mut req)).unwrap_or(Value::Null);
                    let mut s = st.lock().unwrap();
                    let progress = body
                        .get("status")
                        .map(|v| v == "IN_PROGRESS")
                        .unwrap_or(false);
                    s.posts.push((path, q, h, body));
                    if !progress && s.fail_done > 0 {
                        s.fail_done -= 1;
                        drop(s);
                        respond_bytes(req, 500, b"nope".to_vec(), "text/plain");
                        return;
                    }
                    drop(s);
                    respond_json(req, 200, &json!({"status": "ok"}));
                }
                _ => respond_bytes(req, 404, Vec::new(), ""),
            }
        });
        FakeRunpod { fake, state }
    }

    pub fn env(&self) -> fvstudio::rp::RunpodEnv {
        let base = format!("{}/v2/ep1", self.fake.base());
        let m: fvstudio::EnvMap = [
            (
                "RUNPOD_WEBHOOK_GET_JOB",
                format!("{base}/job-take/$ID?gpu=NVIDIA+H200"),
            ),
            (
                "RUNPOD_WEBHOOK_POST_OUTPUT",
                format!("{base}/job-done/$RUNPOD_POD_ID/$ID?gpu=NVIDIA+H200"),
            ),
            (
                "RUNPOD_WEBHOOK_POST_STREAM",
                format!("{base}/job-stream/$RUNPOD_POD_ID/$ID?gpu=NVIDIA+H200"),
            ),
            ("RUNPOD_WEBHOOK_PING", format!("{base}/ping/$RUNPOD_POD_ID")),
            ("RUNPOD_PING_INTERVAL", "100".into()),
            ("RUNPOD_AI_API_KEY", "secret-key".into()),
            ("RUNPOD_POD_ID", "w42".into()),
            ("RUNPOD_POD_HOSTNAME", "host-1".into()),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        fvstudio::rp::RunpodEnv::from_env(&m).unwrap().unwrap()
    }

    pub fn done_posts(
        &self,
    ) -> Vec<(
        String,
        HashMap<String, Vec<String>>,
        HashMap<String, String>,
        Value,
    )> {
        self.state
            .lock()
            .unwrap()
            .posts
            .iter()
            .filter(|p| {
                p.3.get("status")
                    .map(|v| v != "IN_PROGRESS")
                    .unwrap_or(true)
            })
            .cloned()
            .collect()
    }

    pub fn progress_posts(
        &self,
    ) -> Vec<(
        String,
        HashMap<String, Vec<String>>,
        HashMap<String, String>,
        Value,
    )> {
        self.state
            .lock()
            .unwrap()
            .posts
            .iter()
            .filter(|p| {
                p.3.get("status")
                    .map(|v| v == "IN_PROGRESS")
                    .unwrap_or(false)
            })
            .cloned()
            .collect()
    }
}

// ---- S3 ------------------------------------------------------------------------------

/// `PUT /b/k`, `GET /b/k`, `HEAD /b/k`, `GET /b?list-type=2&prefix=…` with
/// two keys per page. Objects in `objects[(bucket, key)]`, every request's
/// (method, path, lower-cased headers) in `log`.
#[derive(Default)]
pub struct S3State {
    pub objects: HashMap<(String, String), (Vec<u8>, Option<String>)>,
    pub log: Vec<(String, String, HashMap<String, String>)>,
    pub fail_next: usize,
}

pub struct FakeS3 {
    pub fake: Fake,
    pub state: Arc<Mutex<S3State>>,
}

impl FakeS3 {
    pub fn start() -> FakeS3 {
        let state = Arc::new(Mutex::new(S3State::default()));
        let st = state.clone();
        let page = 2usize;
        let fake = Fake::start(move |mut req: Request| {
            let url = req.url().to_string();
            let (path, q) = split_url(&url);
            let h = headers_of(&req);
            let parts: Vec<&str> = path.trim_start_matches('/').splitn(2, '/').collect();
            let bucket = unquote(parts[0]);
            let key = parts.get(1).map(|k| unquote(k)).unwrap_or_default();
            let method = req.method().to_string().to_uppercase();
            if method == "PUT" {
                let body = body_of(&mut req);
                let mut s = st.lock().unwrap();
                s.log.push(("PUT".into(), url.clone(), h.clone()));
                s.objects.insert(
                    (bucket, key),
                    (body.clone(), h.get("content-type").cloned()),
                );
                drop(s);
                let etag = format!("\"{}\"", fvstudio::r2::sha256_hex(&body));
                let _ = req.respond(
                    Response::from_data(Vec::new())
                        .with_status_code(200)
                        .with_header(Header::from_bytes("ETag", etag).unwrap()),
                );
                return;
            }
            if method == "GET"
                && key.is_empty()
                && q.get("list-type")
                    .map(|v| v == &["2".to_string()])
                    .unwrap_or(false)
            {
                let mut s = st.lock().unwrap();
                s.log.push(("GET".into(), url.clone(), h.clone()));
                let prefix = q
                    .get("prefix")
                    .and_then(|v| v.first())
                    .cloned()
                    .unwrap_or_default();
                let mut keys: Vec<String> = s
                    .objects
                    .keys()
                    .filter(|(b, k)| *b == bucket && k.starts_with(&prefix))
                    .map(|(_, k)| k.clone())
                    .collect();
                keys.sort();
                let start: usize = q
                    .get("continuation-token")
                    .and_then(|v| v.first())
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
                let end = (start + page).min(keys.len());
                let truncated = start + page < keys.len();
                let mut xml = vec![
                    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>".to_string(),
                    "<ListBucketResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">".to_string(),
                    format!("<Name>{bucket}</Name><Prefix>{prefix}</Prefix><KeyCount>{}</KeyCount><MaxKeys>{page}</MaxKeys>", end - start),
                    format!("<IsTruncated>{}</IsTruncated>", if truncated { "true" } else { "false" }),
                ];
                if truncated {
                    xml.push(format!(
                        "<NextContinuationToken>{}</NextContinuationToken>",
                        start + page
                    ));
                }
                for k in &keys[start..end] {
                    xml.push(format!("<Contents><Key>{k}</Key><Size>{}</Size><StorageClass>STANDARD</StorageClass></Contents>", s.objects[&(bucket.clone(), k.clone())].0.len()));
                }
                xml.push("</ListBucketResult>".into());
                drop(s);
                respond_bytes(req, 200, xml.join("\n").into_bytes(), "application/xml");
                return;
            }
            let head = method == "HEAD";
            let mut s = st.lock().unwrap();
            s.log.push((
                if head { "HEAD".into() } else { "GET".into() },
                url.clone(),
                h.clone(),
            ));
            if s.fail_next > 0 {
                s.fail_next -= 1;
                drop(s);
                respond_bytes(req, 503, Vec::new(), "");
                return;
            }
            let o = s.objects.get(&(bucket, key)).cloned();
            drop(s);
            match o {
                None => respond_bytes(req, 404, Vec::new(), ""),
                Some((body, ctype)) => {
                    let ct = ctype.unwrap_or_else(|| "application/octet-stream".into());
                    if head {
                        // tiny_http drops the body of a HEAD answer and keeps its Content-Length.
                        respond_bytes(req, 200, vec![0u8; body.len()], &ct);
                    } else {
                        respond_bytes(req, 200, body, &ct);
                    }
                }
            }
        });
        FakeS3 { fake, state }
    }
}

// ---- fv-serve -------------------------------------------------------------------------

pub fn make_mp4(
    path: &std::path::Path,
    width: u32,
    height: u32,
    frames: u32,
    fps: u32,
    audio: bool,
) -> bool {
    let Some(ff) = fvstudio::py::which("ffmpeg") else {
        // No ffmpeg: a stub with the ftyp box and 1 KiB of padding.
        let mut v = b"\x00\x00\x00\x18ftypmp42".to_vec();
        v.extend(vec![0u8; 1024]);
        std::fs::write(path, v).unwrap();
        return false;
    };
    let d = format!("{:.3}", frames as f64 / fps as f64);
    let mut cmd = std::process::Command::new(ff);
    cmd.args([
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        &format!("color=c=blue:s={width}x{height}:r={fps}:d={d}"),
    ]);
    if audio {
        cmd.args([
            "-f",
            "lavfi",
            "-i",
            &format!("sine=f=440:d={d}"),
            "-c:a",
            "aac",
        ]);
    }
    cmd.args([
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-frames:v",
        &frames.to_string(),
        "-shortest",
    ])
    .arg(path);
    assert!(cmd.status().unwrap().success());
    true
}

#[derive(Default)]
pub struct FvState {
    pub jobs: Map<String, Value>,
    pub requests: Vec<Value>,
    /// Live streams (`/fv/v1/streams`): id → the stream object.
    pub streams: Map<String, Value>,
    pub commands: Vec<(String, Value)>,
}

/// A tiny fake of fv-serve's native jobs API (`/ping`, `/fv/v1/capabilities`,
/// `POST /fv/v1/jobs`, `GET /fv/v1/jobs/{id}`, `/files/…`): a job runs
/// through queued → running (progress) → succeeded in a few polls and serves
/// a real MP4 (made with ffmpeg when available) at its output URL.
pub struct FakeFvServe {
    pub fake: Fake,
    pub state: Arc<Mutex<FvState>>,
    pub files_dir: std::path::PathBuf,
}

const FV_ALLOWED: &[&str] = &[
    "model",
    "prompt",
    "negative_prompt",
    "seed",
    "size",
    "width",
    "height",
    "aspect_ratio",
    "short_edge",
    "seconds",
    "num_frames",
    "fps",
    "steps",
    "guidance",
    "image_url",
    "last_image_url",
    "reference_urls",
    "reference_strength",
    "reference_lora_strength",
    "audio_url",
    "video_url",
    "start_s",
    "end_s",
    "retake_mode",
    "extend_s",
    "extend_at",
    "context_s",
    "prompt_expansion",
];

impl FakeFvServe {
    pub fn start(
        files_dir: std::path::PathBuf,
        steps: u32,
        step_s: f64,
        fail_model: Option<&str>,
    ) -> FakeFvServe {
        let state = Arc::new(Mutex::new(FvState::default()));
        let st = state.clone();
        let fd = files_dir.clone();
        let fail_model = fail_model.map(str::to_string);
        let port_cell = Arc::new(std::sync::atomic::AtomicU16::new(0));
        let pc = port_cell.clone();
        let fake = Fake::start(move |mut req: Request| {
            let (path, _) = split_url(req.url());
            let port = pc.load(std::sync::atomic::Ordering::SeqCst);
            match (req.method().clone(), path.as_str()) {
                (Method::Get, "/ping") => respond_json(req, 200, &json!({"status": "healthy"})),
                (Method::Get, "/fv/v1/capabilities") => respond_json(
                    req,
                    200,
                    &json!({"auth": {"mode": "none"}, "protocols": ["native"], "models": []}),
                ),
                (Method::Post, "/fv/v1/streams") => {
                    let body: Value = serde_json::from_slice(&body_of(&mut req)).unwrap_or(json!({}));
                    let mut s = st.lock().unwrap();
                    s.requests.push(body.clone());
                    let model = body.get("model").and_then(|m| m.as_str()).unwrap_or("").to_string();
                    if model.is_empty() || body.get("prompt").and_then(|p| p.as_str()).unwrap_or("").is_empty() {
                        drop(s);
                        respond_json(req, 400, &json!({"error": {"kind": "invalid_request", "message": "model and prompt are required"}}));
                        return;
                    }
                    if s.streams.values().any(|v| v["status"]["state"] == "running") {
                        drop(s);
                        respond_json(req, 429, &json!({"error": {"kind": "busy", "message": "a streaming session holds this GPU"}}));
                        return;
                    }
                    let id = format!("fvstream_{}", s.streams.len() + 1);
                    let mode = if model.starts_with("sfwan") || model.starts_with("longlive") { "causal" } else { "clip" };
                    let obj = json!({
                        "id": id, "model": model, "mode": mode,
                        "status": {"state": "running", "output_state": "publishing"},
                        "output": {"kind": body["output"]["kind"].as_str().unwrap_or("cloudflare"), "credentials": "set"},
                        "playback": {"whep_url": format!("https://customer-x.cloudflarestream.com/{id}/webRTC/play")},
                        "session": {"prompt": body["prompt"]},
                    });
                    s.streams.insert(id.clone(), obj.clone());
                    drop(s);
                    respond_json(req, 201, &obj);
                }
                (Method::Get, "/fv/v1/streams") => {
                    let s = st.lock().unwrap();
                    let data: Vec<Value> = s.streams.values().cloned().collect();
                    drop(s);
                    respond_json(req, 200, &json!({"data": data}));
                }
                (m, p) if p.starts_with("/fv/v1/streams/") => {
                    let rest = &p["/fv/v1/streams/".len()..];
                    let (sid, action) = match rest.split_once('/') {
                        Some((a, b)) => (a.to_string(), Some(b.to_string())),
                        None => (rest.to_string(), None),
                    };
                    let mut s = st.lock().unwrap();
                    let Some(mut obj) = s.streams.get(&sid).cloned() else {
                        drop(s);
                        respond_json(req, 404, &json!({"error": {"kind": "not_found", "message": "no such stream"}}));
                        return;
                    };
                    match (m, action.as_deref()) {
                        (Method::Get, None) => {
                            drop(s);
                            respond_json(req, 200, &obj);
                        }
                        (Method::Post, Some("commands")) => {
                            let body: Value = serde_json::from_slice(&body_of(&mut req)).unwrap_or(json!({}));
                            let t = body["type"].as_str().unwrap_or("").to_string();
                            s.commands.push((sid.clone(), body.clone()));
                            if t == "set_prompt" || t == "enqueue" {
                                obj["session"]["prompt"] = body["data"]["prompt"].clone();
                            }
                            s.streams.insert(sid, obj);
                            drop(s);
                            respond_json(req, 200, &json!({"reply": {"ok": true, "type": t}}));
                        }
                        (Method::Delete, None) => {
                            obj["status"]["state"] = json!("closed");
                            obj["status"]["output_state"] = json!("ended");
                            s.streams.insert(sid, obj);
                            drop(s);
                            respond_bytes(req, 204, Vec::new(), "");
                        }
                        _ => {
                            drop(s);
                            respond_json(req, 405, &json!({"error": {"kind": "method", "message": "no"}}));
                        }
                    }
                }
                (Method::Get, p) if p.starts_with("/fv/v1/jobs/") => {
                    let jid = p.rsplit('/').next().unwrap().to_string();
                    let mut s = st.lock().unwrap();
                    let Some(mut j) = s.jobs.get(&jid).cloned() else {
                        drop(s);
                        respond_json(
                            req,
                            404,
                            &json!({"error": {"kind": "not_found", "message": "no such job"}}),
                        );
                        return;
                    };
                    advance(&mut j, steps, step_s, fail_model.as_deref(), &fd, port);
                    s.jobs.insert(jid, j.clone());
                    drop(s);
                    respond_json(req, 200, &view(&j));
                }
                (Method::Get, p) if p.starts_with("/files/") => {
                    let f = fd.join(p["/files/".len()..].split('?').next().unwrap());
                    match std::fs::read(&f) {
                        Ok(data) => respond_bytes(req, 200, data, "video/mp4"),
                        Err(_) => respond_bytes(req, 404, Vec::new(), ""),
                    }
                }
                (Method::Post, "/fv/v1/jobs") => {
                    let body: Value =
                        serde_json::from_slice(&body_of(&mut req)).unwrap_or(json!({}));
                    let mut s = st.lock().unwrap();
                    s.requests.push(body.clone());
                    let mut unknown: Vec<&String> = body
                        .as_object()
                        .map(|o| {
                            o.keys()
                                .filter(|k| !FV_ALLOWED.contains(&k.as_str()))
                                .collect()
                        })
                        .unwrap_or_default();
                    unknown.sort();
                    if let Some(u) = unknown.first() {
                        drop(s);
                        respond_json(
                            req,
                            400,
                            &json!({"error": {"kind": "invalid_request", "message": format!("unknown field `{u}`"), "param": u}}),
                        );
                        return;
                    }
                    let truthy = |k: &str| body.get(k).map(fvstudio::py::truthy).unwrap_or(false);
                    if !truthy("prompt") && !truthy("video_url") && !truthy("audio_url") {
                        drop(s);
                        respond_json(
                            req,
                            400,
                            &json!({"error": {"kind": "invalid_request", "message": "`prompt` must not be empty", "param": "prompt"}}),
                        );
                        return;
                    }
                    let jid = format!(
                        "fvjob_{:x}{}",
                        fvstudio::now_s().to_bits(),
                        s.requests.len()
                    );
                    let size = body
                        .get("size")
                        .and_then(|v| v.as_str())
                        .unwrap_or("832x480")
                        .to_string();
                    let (w, h) = size.split_once('x').unwrap();
                    let j = json!({"id": jid, "model": body["model"], "created": fvstudio::now_s(), "state": "queued", "progress": 0.0,
                                   "width": w.parse::<i64>().unwrap(), "height": h.parse::<i64>().unwrap(),
                                   "num_frames": body.get("num_frames").and_then(|v| v.as_i64()).unwrap_or(9), "fps": body.get("fps").and_then(|v| v.as_i64()).unwrap_or(24),
                                   "seed": body.get("seed").and_then(|v| v.as_i64()).unwrap_or(42), "body": body});
                    s.jobs.insert(jid, j.clone());
                    drop(s);
                    respond_json(req, 202, &view(&j));
                }
                _ => respond_bytes(req, 404, Vec::new(), ""),
            }
        });
        port_cell.store(fake.port, std::sync::atomic::Ordering::SeqCst); // the output URL carries this server's port
        FakeFvServe {
            fake,
            state,
            files_dir,
        }
    }

    pub fn requests(&self) -> Vec<Value> {
        self.state.lock().unwrap().requests.clone()
    }
}

fn advance(
    j: &mut Value,
    steps: u32,
    step_s: f64,
    fail_model: Option<&str>,
    files_dir: &std::path::Path,
    port: u16,
) {
    let t = fvstudio::now_s() - j["created"].as_f64().unwrap();
    let state = j["state"].as_str().unwrap().to_string();
    if state == "succeeded" || state == "failed" {
        return;
    }
    if fail_model.map(|f| j["model"] == f).unwrap_or(false) {
        j["state"] = json!("failed");
        j["error"] = json!({"kind": "engine", "message": "the fake engine refuses this model"});
        return;
    }
    let k = (t / step_s) as u32;
    if k == 0 {
        j["state"] = json!("queued");
    } else if k <= steps {
        j["state"] = json!("running");
        j["progress"] = json!(k as f64 / steps as f64);
    } else {
        j["state"] = json!("succeeded");
        j["progress"] = json!(1.0);
        let out = files_dir.join(format!("{}.mp4", j["id"].as_str().unwrap()));
        if !out.exists() {
            std::fs::create_dir_all(files_dir).unwrap();
            make_mp4(
                &out,
                j["width"].as_u64().unwrap() as u32,
                j["height"].as_u64().unwrap() as u32,
                j["num_frames"].as_u64().unwrap().min(9) as u32,
                j["fps"].as_u64().unwrap() as u32,
                true,
            );
        }
        j["output"] = json!({"url": format!("http://127.0.0.1:{port}/files/{}?sig=x", out.file_name().unwrap().to_string_lossy()), "mime": "video/mp4", "bytes": out.metadata().unwrap().len(),
                             "width": j["width"], "height": j["height"], "frames": j["num_frames"], "fps": j["fps"], "audio": {"rate": 48000, "channels": 2}});
        j["metrics"] = json!({"inference_s": 1.5, "stage_durations": {"text_encoder": 0.2, "vae_decode": 0.3}});
    }
}

fn view(j: &Value) -> Value {
    json!({"id": j["id"], "object": "fv.job", "status": j["state"], "progress": (j["progress"].as_f64().unwrap_or(0.0) * 100.0).round() / 100.0, "queue_position": null,
           "model": j["model"], "resolved_model": j["model"], "task": "t2v", "width": j["width"], "height": j["height"],
           "num_frames": j["num_frames"], "fps": j["fps"], "seed": j["seed"], "error": j.get("error").cloned().unwrap_or(Value::Null), "output": j.get("output").cloned().unwrap_or(Value::Null),
           "protocol": "native", "metrics": j.get("metrics").cloned().unwrap_or(json!({"inference_s": null, "stage_durations": {}}))})
}

// ---- llama-server ------------------------------------------------------------------------

#[derive(Default)]
pub struct LlamaState {
    pub answer: String,
    pub loading: bool,
    pub fail_status: Option<u16>,
    pub hang_s: f64,
    pub requests: Vec<Value>,
    pub health_calls: usize,
}

/// A tiny fake of llama-server's HTTP API: `GET /health` (503 while
/// "loading", 200 after) and `POST /v1/chat/completions` answering a
/// canned rewrite. Every request body is recorded.
pub struct FakeLlama {
    pub fake: Fake,
    pub state: Arc<Mutex<LlamaState>>,
}

impl FakeLlama {
    pub fn start(answer: &str, loading: bool, fail_status: Option<u16>, hang_s: f64) -> FakeLlama {
        let state = Arc::new(Mutex::new(LlamaState {
            answer: answer.into(),
            loading,
            fail_status,
            hang_s,
            ..Default::default()
        }));
        let st = state.clone();
        let fake = Fake::start(move |mut req: Request| {
            let (path, _) = split_url(req.url());
            match (req.method().clone(), path.as_str()) {
                (Method::Get, "/health") => {
                    let mut s = st.lock().unwrap();
                    s.health_calls += 1;
                    let loading = s.loading;
                    drop(s);
                    if loading {
                        respond_json(
                            req,
                            503,
                            &json!({"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}}),
                        );
                    } else {
                        respond_json(req, 200, &json!({"status": "ok"}));
                    }
                }
                (Method::Post, "/v1/chat/completions") => {
                    let body: Value =
                        serde_json::from_slice(&body_of(&mut req)).unwrap_or(json!({}));
                    let mut s = st.lock().unwrap();
                    s.requests.push(body.clone());
                    let (hang, fail, mut answer) = (s.hang_s, s.fail_status, s.answer.clone());
                    drop(s);
                    if hang > 0.0 {
                        std::thread::sleep(std::time::Duration::from_secs_f64(hang));
                    }
                    if let Some(code) = fail {
                        respond_json(
                            req,
                            code,
                            &json!({"error": {"code": code, "message": "boom"}}),
                        );
                        return;
                    }
                    if body.get("max_tokens") == Some(&json!(1)) {
                        answer = "ok".into(); // the readiness probe's 1-token completion
                    }
                    respond_json(
                        req,
                        200,
                        &json!({"id": "chatcmpl-fake", "object": "chat.completion", "created": 0, "model": body.get("model").cloned().unwrap_or(json!("fake")),
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}),
                    );
                }
                _ => respond_json(req, 404, &json!({"error": "not found"})),
            }
        });
        FakeLlama { fake, state }
    }

    pub fn base(&self) -> String {
        self.fake.base()
    }

    pub fn last(&self) -> Value {
        self.state.lock().unwrap().requests.last().cloned().unwrap()
    }

    pub fn system_of(&self) -> String {
        self.last()["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["role"] == "system")
            .unwrap()["content"]
            .as_str()
            .unwrap()
            .to_string()
    }

    pub fn user_of(&self) -> Value {
        self.last()["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["role"] == "user")
            .unwrap()["content"]
            .clone()
    }
}
