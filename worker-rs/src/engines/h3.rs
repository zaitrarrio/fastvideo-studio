//! h3-rs: `h3 serve` for the tasks it advertises, `h3 generate` /
//! `h3 regen2k` per job for the rest (design §1).

use super::common::{run_cli, run_serve_job};
use super::{random_seed, Ctx, RunResult};
use crate::catalog::{b64_encode, Job};
use crate::errors::{JobError, JobResult};
use crate::flags::{self, H3Media};
use crate::frames as fr;
use crate::media::{self as md, LocalMedia};
use crate::mux;
use crate::sidecar::{http_info_probe, Sidecar};
use crate::{httpc, py};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

pub const ENGINE: &str = "h3";
/// h3 serve's cap on a body with media (inputs.max_body_bytes)
pub const SERVE_BODY_MAX: u64 = 64 * 1024 * 1024;
const B64_RATIO: f64 = 4.0 / 3.0;

/// `/api/info.tasks` (the parallel h3-rs change) lists what serve takes;
/// without it, text-to-video only.
pub fn serve_supports(info: Option<&Value>, task: &str) -> bool {
    if task == "t2v" {
        return true;
    }
    let Some(tasks) = info.and_then(|i| i.get("tasks")).and_then(|t| t.as_array()) else {
        return false;
    };
    let want = flags::h3_task_name(task);
    tasks.iter().any(|t| t.as_str() == want)
}

fn b64_file(path: &str) -> std::io::Result<String> {
    Ok(b64_encode(&std::fs::read(path)?))
}

fn size_of(path: &str) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// The decoded size of everything `serve_media_fields` would send; the
/// base64 body is 4/3 of it. Over `SERVE_BODY_MAX` the job runs through the CLI.
pub fn serve_media_bytes(local: &H3Media) -> u64 {
    let mut n = 0;
    for p in [&local.first_image, &local.last_image]
        .into_iter()
        .flatten()
    {
        n += size_of(p);
    }
    for (_, path, audio) in &local.references {
        n += size_of(path) + audio.as_deref().map(size_of).unwrap_or(0);
    }
    if let Some((y4m, wav)) = &local.continue_ {
        n += size_of(y4m) + wav.as_deref().map(size_of).unwrap_or(0);
    }
    n
}

pub fn serve_fits(local: &H3Media) -> bool {
    (serve_media_bytes(local) as f64) * B64_RATIO < SERVE_BODY_MAX as f64 * 0.95
}

fn image_mime(path: &str) -> &'static str {
    if path.to_lowercase().ends_with(".png") {
        "image/png"
    } else {
        "image/jpeg"
    }
}

/// `h3 serve`'s per-request inputs (h3-rs serve.rs): `task`, `first_image`
/// / `last_image` as {b64, mime}, `refs` as [{kind, b64, mime?, audio_b64?}]
/// (video as Y4M bytes, audio as WAV), `continue` as {b64, audio_b64?}.
pub fn serve_media_fields(task: &str, local: &H3Media) -> std::io::Result<Map<String, Value>> {
    let mut out = Map::new();
    out.insert(
        "task".into(),
        json!(flags::h3_task_name(task).unwrap_or(task)),
    );
    for (key, p) in [
        ("first_image", &local.first_image),
        ("last_image", &local.last_image),
    ] {
        if let Some(path) = p {
            out.insert(
                key.into(),
                json!({"b64": b64_file(path)?, "mime": image_mime(path)}),
            );
        }
    }
    let mut refs = Vec::new();
    for (kind, path, audio) in &local.references {
        match kind.as_str() {
            "image" => refs
                .push(json!({"kind": "image", "b64": b64_file(path)?, "mime": image_mime(path)})),
            "video" => {
                let mut r =
                    json!({"kind": "video", "b64": b64_file(path)?, "mime": "video/x-yuv4mpeg"});
                if let Some(a) = audio {
                    r["audio_b64"] = json!(b64_file(a)?);
                }
                refs.push(r);
            }
            _ => refs.push(json!({"kind": "audio", "b64": b64_file(path)?, "mime": "audio/wav"})),
        }
    }
    if !refs.is_empty() {
        out.insert("refs".into(), Value::Array(refs));
    }
    if let Some((y4m, wav)) = &local.continue_ {
        let mut c = json!({"b64": b64_file(y4m)?});
        if let Some(w) = wav {
            c["audio_b64"] = json!(b64_file(w)?);
        }
        out.insert("continue".into(), c);
    }
    Ok(out)
}

pub fn media_kind(lm: &LocalMedia) -> JobResult<&'static str> {
    match lm.kind() {
        "image" => return Ok("image"),
        "video" => return Ok("video"),
        "audio" => return Ok("audio"),
        _ => {}
    }
    let ext = lm
        .path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "y4m" | "mp4" | "mov" | "webm" | "mkv" => Ok("video"),
        "wav" | "mp3" | "m4a" | "flac" | "ogg" => Ok("audio"),
        "png" | "jpg" | "jpeg" | "webp" | "ppm" => Ok("image"),
        _ => Err(JobError::validate(format!(
            "{}: cannot tell image / video / audio from {}",
            lm.path
                .file_name()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or_default(),
            lm.mime
        ))),
    }
}

fn s(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

/// Inputs → local files in the forms `h3 generate` takes.
pub fn prepare_media(ctx: &Ctx, job: &Job) -> JobResult<H3Media> {
    let mut out = H3Media::default();
    let mdir = ctx.job_dir.join("in");
    let fps = job.fps_for();
    for key in ["first_image", "last_image"] {
        if let Some(m) = job.inputs.get(key) {
            let lm = md::fetch_media(m, &mdir, key, ctx.store_ref(), ctx.bucket.as_deref())?;
            if media_kind(&lm)? != "image" {
                return Err(JobError::validate(format!("{key} must be an image")));
            }
            if key == "first_image" {
                out.first_image = Some(s(&lm.path));
            } else {
                out.last_image = Some(s(&lm.path));
            }
        }
    }
    for (i, m) in job.inputs.references.iter().enumerate() {
        let lm = md::fetch_media(
            m,
            &mdir,
            &format!("ref{i}"),
            ctx.store_ref(),
            ctx.bucket.as_deref(),
        )?;
        match media_kind(&lm)? {
            "image" => out.references.push(("image".into(), s(&lm.path), None)),
            "video" => {
                let is_y4m = lm.path.extension().map(|e| e == "y4m").unwrap_or(false);
                let y4m = if is_y4m {
                    lm.path.clone()
                } else {
                    mux::to_y4m(
                        &lm.path,
                        &mdir.join(format!("ref{i}.y4m")),
                        None,
                        Some(fps),
                        None,
                        None,
                    )
                    .map_err(|e| anyhow::anyhow!("RuntimeError: {e}"))?
                };
                let mut wav = None;
                let p = if is_y4m {
                    None
                } else {
                    mux::probe(&lm.path, None)
                };
                if p.map(|p| p.audio).unwrap_or(false) {
                    wav = Some(s(&mux::to_wav(
                        &lm.path,
                        &mdir.join(format!("ref{i}.wav")),
                        None,
                    )
                    .map_err(|e| anyhow::anyhow!("RuntimeError: {e}"))?));
                }
                out.references.push(("video".into(), s(&y4m), wav));
            }
            _ => {
                let is_wav = lm.path.extension().map(|e| e == "wav").unwrap_or(false);
                let wav = if is_wav {
                    lm.path.clone()
                } else {
                    mux::to_wav(&lm.path, &mdir.join(format!("ref{i}.wav")), None)
                        .map_err(|e| anyhow::anyhow!("RuntimeError: {e}"))?
                };
                out.references.push(("audio".into(), s(&wav), None));
            }
        }
    }
    if let Some(m) = job.inputs.get("video") {
        // extend → --continue
        let lm = md::fetch_media(m, &mdir, "continue", ctx.store_ref(), ctx.bucket.as_deref())?;
        let y4m = mux::to_y4m(
            &lm.path,
            &mdir.join("continue.y4m"),
            Some((job.width, job.height)),
            Some(fps),
            Some(flags::CONTINUE_FRAMES),
            None,
        )
        .map_err(|e| anyhow::anyhow!("RuntimeError: {e}"))?;
        let mut wav = None;
        if mux::probe(&lm.path, None).map(|p| p.audio).unwrap_or(false) {
            wav = Some(s(&mux::to_wav(&lm.path, &mdir.join("continue.wav"), None)
                .map_err(|e| anyhow::anyhow!("RuntimeError: {e}"))?));
        }
        out.continue_ = Some((s(&y4m), wav));
    }
    if let Some(m) = job.inputs.get("draft") {
        let lm = md::fetch_media(m, &mdir, "draft", ctx.store_ref(), ctx.bucket.as_deref())?;
        let is_y4m = lm.path.extension().map(|e| e == "y4m").unwrap_or(false);
        out.draft = Some(if is_y4m {
            s(&lm.path)
        } else {
            s(&mux::to_y4m(
                &lm.path,
                &mdir.join("draft.y4m"),
                None,
                Some(fps),
                None,
                None,
            )
            .map_err(|e| anyhow::anyhow!("RuntimeError: {e}"))?)
        });
    }
    Ok(out)
}

pub fn run(ctx: &Ctx, job: &Job) -> JobResult<RunResult> {
    let st = ctx.settings;
    let seed = job.seed.unwrap_or_else(random_seed);
    let local = prepare_media(ctx, job)?;
    let out = ctx.job_dir.join("out.y4m");
    if job.task == "regen2k" {
        let draft = PathBuf::from(
            local
                .draft
                .clone()
                .ok_or_else(|| JobError::validate("regen2k needs input 'draft'"))?,
        );
        let argv = flags::h3_regen2k_argv(job, st, ctx.knobs, &draft, &out, seed)?;
        return run_cli_job(ctx, job, &argv, &out, seed, None);
    }
    let command = flags::command_line(&flags::h3_generate_argv(
        job,
        st,
        ctx.knobs,
        &local,
        Path::new("out.y4m"),
        seed,
    )?);
    if job.model.cli_only {
        let argv = flags::h3_generate_argv(job, st, ctx.knobs, &local, &out, seed)?;
        return run_cli_job(ctx, job, &argv, &out, seed, Some(command));
    }
    let port = ctx.port(ENGINE);
    let base = format!("http://127.0.0.1:{port}");
    let key = flags::h3_serve_key(job, st, ctx.knobs)?;
    let serve_argv = flags::h3_serve_argv(job, st, ctx.knobs, port)?;
    let env = ctx.env_of(ENGINE);
    let make = || {
        Sidecar::new(
            ENGINE,
            &key,
            serve_argv.clone(),
            env.clone(),
            http_info_probe(&format!("{base}/api/info")),
        )
    };
    let loading = |_: &str, t: f64| ctx.report.detail("loading", &format!("{t:.0} s"));

    // A sidecar already up for this recipe answers whether it takes the task.
    let cur = ctx.sidecars.get(ENGINE);
    let mut info: Option<Value> = None;
    if cur
        .as_ref()
        .map(|c| c.alive() && c.key == key)
        .unwrap_or(false)
    {
        info = httpc::get_json(&format!("{base}/api/info"), 5.0).ok();
    } else if job.task != "t2v" {
        let (sc, _load_s) =
            ctx.sidecars
                .ensure(ENGINE, &key, &make, job.model.vram_gib, Some(&loading))?;
        ctx.report.force("loading");
        info = Some(
            httpc::get_json(&format!("{base}/api/info"), 5.0)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
        );
        if !serve_supports(info.as_ref(), &job.task) {
            // serve cannot take it: stop it so the CLI has the VRAM.
            sc.stop(15.0);
            ctx.sidecars.remove(ENGINE);
        }
    }
    if !serve_supports(info.as_ref(), &job.task) || !serve_fits(&local) {
        let argv = flags::h3_generate_argv(job, st, ctx.knobs, &local, &out, seed)?;
        return run_cli_job(ctx, job, &argv, &out, seed, Some(command));
    }
    let (sc, load_s) =
        ctx.sidecars
            .ensure(ENGINE, &key, &make, job.model.vram_gib, Some(&loading))?;
    let mut body = Map::new();
    body.insert("prompt".into(), json!(job.prompt));
    body.insert("width".into(), json!(job.width));
    body.insert("height".into(), json!(job.height));
    body.insert("frames".into(), json!(job.frames));
    body.insert("seed".into(), json!(seed));
    body.insert(
        "upscale".into(),
        if job.upscale != 0 {
            json!(job.upscale)
        } else {
            json!(false)
        },
    );
    for (k, v) in serve_media_fields(&job.task, &local)? {
        body.insert(k, v);
    }
    let clip = run_serve_job(
        &base,
        &sc,
        &Value::Object(body),
        &ctx.job_dir,
        ctx.report,
        job.model.audio,
        0.5,
        3600.0,
    )?;
    ctx.report.detail_force("muxing", "frames → y4m");
    let fps = job.fps_for();
    fr::rgb_to_y4m(&clip.rgb, clip.width, clip.height, fps, &out, None)
        .map_err(|e| anyhow::anyhow!("RuntimeError: {e}"))?;
    let mut res = RunResult::new(
        seed,
        clip.width as i64,
        clip.height as i64,
        clip.frames as i64,
        fps,
    );
    res.video_y4m = Some(out);
    res.audio_wav = clip.audio_wav.clone();
    res.timings = clip.timings.clone();
    res.command = command;
    res.timings.load_s = load_s;
    if !clip.hd_jpegs.is_empty() {
        let hd = ctx.job_dir.join("out-hd.y4m");
        fr::jpegs_to_y4m(&clip.hd_jpegs, fps, &hd, None)
            .map_err(|e| anyhow::anyhow!("RuntimeError: {e}"))?;
        let info_hd = fr::read_y4m_info(&hd).map_err(|e| anyhow::anyhow!("ValueError: {e}"))?;
        res.video_y4m = Some(hd);
        res.upscaled = Some((
            clip.up_factor.unwrap_or(job.upscale),
            info_hd.width as i64,
            info_hd.height as i64,
        ));
    }
    Ok(res)
}

fn run_cli_job(
    ctx: &Ctx,
    job: &Job,
    argv: &[String],
    out: &Path,
    seed: i64,
    command: Option<String>,
) -> JobResult<RunResult> {
    // The CLI needs the whole GPU: no resident sidecar beside it.
    ctx.sidecars.stop_all();
    ctx.report.detail_force("loading", "h3 (per-job CLI)");
    let (_, t) = run_cli(
        argv,
        &ctx.env_of(ENGINE),
        ctx.report,
        &ctx.log,
        7200.0,
        Some(&ctx.job_dir),
    )?;
    if !out.is_file() {
        return Err(JobError::engine(
            format!(
                "h3 wrote no {}",
                out.file_name()
                    .map(|f| f.to_string_lossy().into_owned())
                    .unwrap_or_default()
            ),
            vec![],
        ));
    }
    let info = fr::read_y4m_info(out).map_err(|e| anyhow::anyhow!("ValueError: {e}"))?;
    let wav = out.with_extension("wav");
    let mut res = RunResult::new(
        seed,
        info.width as i64,
        info.height as i64,
        info.frames as i64,
        job.fps_for(),
    );
    res.video_y4m = Some(out.to_path_buf());
    res.audio_wav = if wav.is_file() { Some(wav) } else { None };
    res.timings = t;
    res.command = command.unwrap_or_else(|| {
        let name = out
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_default();
        let line: Vec<String> = argv
            .iter()
            .map(|a| {
                if a == &py::py_str(&json!(s(out))) {
                    name.clone()
                } else {
                    a.clone()
                }
            })
            .collect();
        flags::command_line(&line)
    });
    if job.upscale != 0 && (info.width as i64, info.height as i64) != (job.width, job.height) {
        res.width = job.width;
        res.height = job.height;
        res.upscaled = Some((job.upscale, info.width as i64, info.height as i64));
    }
    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tempdir;

    #[test]
    fn serve_media_fields_shape() {
        let dir = tempdir("h3media");
        let w = |n: &str, b: &[u8]| {
            let p = dir.join(n);
            std::fs::write(&p, b).unwrap();
            s(&p)
        };
        let (png, jpg, y4m, wav, cont) = (
            w("a.png", b"\x89PNG-first"),
            w("b.jpg", b"\xff\xd8-last"),
            w("r.y4m", b"YUV4MPEG2 ref"),
            w("r.wav", b"RIFF ref"),
            w("c.y4m", b"YUV4MPEG2 cont"),
        );
        let local = H3Media {
            first_image: Some(png.clone()),
            last_image: Some(jpg),
            references: vec![
                ("image".into(), png.clone(), None),
                ("video".into(), y4m, Some(wav.clone())),
                ("audio".into(), wav, None),
            ],
            continue_: Some((cont.clone(), None)),
            draft: None,
        };
        let b = serve_media_fields("ref2v", &local).unwrap();
        assert_eq!(b["task"], "ref2va");
        assert_eq!(
            b["first_image"],
            json!({"b64": b64_encode(b"\x89PNG-first"), "mime": "image/png"})
        );
        assert_eq!(b["last_image"]["mime"], "image/jpeg");
        let kinds: Vec<&str> = b["refs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["kind"].as_str().unwrap())
            .collect();
        assert_eq!(kinds, ["image", "video", "audio"]);
        assert_eq!(b["refs"][1]["mime"], "video/x-yuv4mpeg");
        assert!(b["refs"][1].get("audio_b64").is_some());
        assert_eq!(b["refs"][2]["mime"], "audio/wav");
        assert!(b["refs"][2].get("audio_b64").is_none());
        assert_eq!(b["continue"], json!({"b64": b64_encode(b"YUV4MPEG2 cont")}));
        assert!(serve_fits(&local));
        let fl = serve_media_fields(
            "fl2v",
            &H3Media {
                first_image: Some(png),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            Value::Object(fl),
            json!({"task": "fl2va", "first_image": b["first_image"]})
        );
        assert_eq!(
            Value::Object(serve_media_fields("t2v", &H3Media::default()).unwrap()),
            json!({"task": "t2va"})
        );
    }

    #[test]
    fn serve_media_over_the_body_cap_goes_to_the_cli() {
        let dir = tempdir("h3big");
        let big = dir.join("big.y4m");
        let f = std::fs::File::create(&big).unwrap();
        f.set_len(50 * 1024 * 1024).unwrap(); // 50 MiB raw → 66.7 MiB of base64, over serve's 64 MiB
        assert!(!serve_fits(&H3Media {
            references: vec![("video".into(), s(&big), None)],
            ..Default::default()
        }));
        let small = dir.join("small.y4m");
        std::fs::write(&small, [b'x'; 1024]).unwrap();
        assert!(serve_fits(&H3Media {
            references: vec![("video".into(), s(&small), None)],
            ..Default::default()
        }));
    }

    #[test]
    fn serve_supports_reads_info_tasks() {
        assert!(serve_supports(
            Some(&json!({"tasks": ["t2va", "fl2va"]})),
            "fl2v"
        ));
        assert!(!serve_supports(
            Some(&json!({"tasks": ["t2va", "fl2va"]})),
            "ref2v"
        ));
        assert!(serve_supports(Some(&json!({"ready": true})), "t2v")); // a serve without `tasks` (older h3-rs) still takes plain t2v
        assert!(!serve_supports(None, "fl2v"));
    }
}
