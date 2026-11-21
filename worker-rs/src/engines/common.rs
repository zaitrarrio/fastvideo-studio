//! Shared by the h3 and fastwan drivers: the serve job loop over
//! `/api/generate` + `/api/jobs/<id>` (+ `/frames`, `/audio`, `/hd`), and a CLI
//! runner that streams `step i/n` lines into progress.

use crate::errors::{JobError, JobResult};
use crate::frames as fr;
use crate::httpc;
use crate::progress::{Detail, Reporter, Timings};
use crate::sidecar::{Sidecar, TAIL};
use crate::{now_s, Log};
use serde_json::{Map, Value};
use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

/// `step 3/8 2.5s` → (3, 8, 2.5)
pub fn parse_step(line: &str) -> Option<(i64, i64, f64)> {
    let i = line.find("step ")?;
    let rest = &line[i + 5..];
    let (a, rest) = rest.split_once('/')?;
    let a: i64 = a.parse().ok()?;
    let n_end = rest.find(' ')?;
    let n: i64 = rest[..n_end].parse().ok()?;
    let rest = &rest[n_end + 1..];
    let secs_end = rest.find('s')?;
    let secs: f64 = rest[..secs_end].parse().ok()?;
    Some((a, n, secs))
}

/// `block 2: denoise 0.84s  decode 0.10s … 12 frames (36 total …` → (block, denoise, decode, frames, total)
pub fn parse_block(line: &str) -> Option<(i64, f64, f64, i64, i64)> {
    let i = line.find("block ")?;
    let rest = &line[i + 6..];
    let (b, rest) = rest.split_once(": denoise ")?;
    let b: i64 = b.parse().ok()?;
    let (den, rest) = rest.split_once('s')?;
    let den: f64 = den.parse().ok()?;
    let rest = rest.trim_start();
    let rest = rest.strip_prefix("decode ")?;
    let (dec, rest) = rest.split_once('s')?;
    let dec: f64 = dec.parse().ok()?;
    let j = rest.find(" frames (")?;
    let frames: i64 = rest[..j].rsplit(' ').next()?.parse().ok()?;
    let rest = &rest[j + 9..];
    let k = rest.find(" total")?;
    let total: i64 = rest[..k].parse().ok()?;
    Some((b, den, dec, frames, total))
}

pub fn engine_stage(s: &str) -> &'static str {
    match s {
        "queued" => "queued",
        "encoding" => "encoding",
        "preparing" => "preparing",
        "denoising" => "denoising",
        "decoding" => "decoding",
        "upscaling" => "upscaling",
        "done" => "done",
        "failed" => "failed",
        _ => "denoising",
    }
}

pub struct ServeClip {
    pub rgb: PathBuf,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub audio_wav: Option<PathBuf>,
    pub hd_jpegs: Vec<Vec<u8>>,
    pub up_factor: Option<i64>,
    pub status: Map<String, Value>,
    pub timings: Timings,
}

fn f64_of(v: Option<&Value>) -> f64 {
    v.and_then(|v| v.as_f64()).unwrap_or(0.0)
}

fn i64_of(v: Option<&Value>) -> i64 {
    v.and_then(|v| v.as_f64()).map(|f| f as i64).unwrap_or(0)
}

/// Submit to a resident engine and stream its frames as they decode.
pub fn run_serve_job(
    base: &str,
    sidecar: &Sidecar,
    body: &Value,
    job_dir: &Path,
    report: &Reporter,
    want_audio: bool,
    poll_s: f64,
    timeout_s: f64,
) -> JobResult<ServeClip> {
    let sub = match httpc::post_json(&format!("{base}/api/generate"), body, 30.0) {
        Ok(Some(v)) => v,
        Ok(None) => {
            return Err(JobError::engine(
                format!("{} refused the job: empty answer", sidecar.engine),
                sidecar.tail(TAIL),
            ))
        }
        Err(httpc::Error::Status(e)) => {
            let msg = e
                .json()
                .and_then(|j| j.get("error").map(crate::py::py_str))
                .unwrap_or_else(|| e.to_string());
            return Err(JobError::engine(
                format!("{} refused the job: {msg}", sidecar.engine),
                sidecar.tail(TAIL),
            ));
        }
        Err(e) => {
            return Err(JobError::engine(
                format!("{} refused the job: {e}", sidecar.engine),
                sidecar.tail(TAIL),
            ))
        }
    };
    let jid = crate::py::py_str(&sub["id"]);
    let rgb_path = job_dir.join("frames.rgb");
    let mut clip = fr::RgbClip::new(rgb_path.clone());
    let t0 = now_s();
    let mut status: Map<String, Value> = Map::new();
    let mut hd: Vec<Vec<u8>> = Vec::new();
    let result = (|| -> JobResult<()> {
        loop {
            if !sidecar.alive() {
                return Err(JobError::engine(
                    format!("{} sidecar died during the job", sidecar.engine),
                    sidecar.tail(TAIL),
                ));
            }
            status = httpc::get_json(&format!("{base}/api/jobs/{jid}"), 30.0)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .as_object()
                .cloned()
                .unwrap_or_default();
            let stage = engine_stage(status.get("stage").and_then(|s| s.as_str()).unwrap_or(""));
            let steps: Vec<Value> = status
                .get("steps")
                .and_then(|s| s.as_array())
                .cloned()
                .unwrap_or_default();
            let total_steps = status
                .get("total_steps")
                .filter(|v| !v.is_null())
                .map(|v| i64_of(Some(v)));
            let got = i64_of(status.get("frames"));
            let total_frames = status
                .get("total_frames")
                .filter(|v| !v.is_null())
                .map(|v| i64_of(Some(v)));
            if got > clip.frames as i64 {
                let body = httpc::get_bytes(
                    &format!("{base}/api/jobs/{jid}/frames?from={}", clip.frames),
                    120.0,
                )
                .map_err(|e| anyhow::anyhow!("{e}"))?;
                let chunk = fr::parse_frame_stream(&body)
                    .map_err(|e| anyhow::anyhow!("ValueError: {e}"))?;
                clip.append(&chunk)
                    .map_err(|e| anyhow::anyhow!("ValueError: {e}"))?;
            }
            let up_frames = i64_of(status.get("up_frames"));
            while (hd.len() as i64) < up_frames {
                hd.push(
                    httpc::get_bytes(
                        &format!("{base}/api/jobs/{jid}/hd?frame={}", hd.len()),
                        120.0,
                    )
                    .map_err(|e| anyhow::anyhow!("{e}"))?,
                );
            }
            if stage == "failed" {
                let err = status
                    .get("error")
                    .filter(|e| crate::py::truthy(e))
                    .map(crate::py::py_str)
                    .unwrap_or_else(|| "job failed".into());
                return Err(JobError::engine(
                    format!("{}: {err}", sidecar.engine),
                    sidecar.tail(TAIL),
                ));
            }
            if stage == "done" {
                break;
            }
            let detail = if stage == "denoising" && total_steps.map(|t| t != 0).unwrap_or(false) {
                Some(format!("step {}/{}", steps.len(), total_steps.unwrap()))
            } else if stage == "decoding" && total_frames.map(|t| t != 0).unwrap_or(false) {
                Some(format!("{got}/{} frames", total_frames.unwrap()))
            } else if stage == "upscaling" && total_frames.map(|t| t != 0).unwrap_or(false) {
                Some(format!("{up_frames}/{} frames", total_frames.unwrap()))
            } else {
                None
            };
            report.report(
                stage,
                Detail {
                    detail: detail.as_deref(),
                    step: Some(steps.len() as i64),
                    total_steps,
                    frames: Some(got),
                    total_frames,
                    force: false,
                },
            );
            if now_s() - t0 > timeout_s {
                return Err(JobError::engine(
                    format!("{} job timed out after {timeout_s:.0} s", sidecar.engine),
                    sidecar.tail(TAIL),
                ));
            }
            std::thread::sleep(Duration::from_secs_f64(poll_s));
        }
        Ok(())
    })();
    clip.close();
    result?;
    let mut audio = None;
    if want_audio && status.get("audio").map(crate::py::truthy).unwrap_or(false) {
        let p = job_dir.join("audio.wav");
        std::fs::write(
            &p,
            httpc::get_bytes(&format!("{base}/api/jobs/{jid}/audio"), 120.0)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
        )?;
        audio = Some(p);
    }
    let steps_s: Vec<f64> = status
        .get("steps")
        .and_then(|s| s.as_array())
        .map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect())
        .unwrap_or_default();
    let t = Timings {
        encode_s: f64_of(status.get("encode_s")),
        denoise_s: steps_s.iter().sum(),
        decode_s: f64_of(status.get("decode_s")),
        upscale_s: {
            let u = f64_of(status.get("up_s"));
            if u != 0.0 {
                u
            } else {
                f64_of(status.get("upscale_s"))
            }
        },
        steps_s,
        ..Default::default()
    };
    if let Some(e) = status.get("up_error").filter(|e| crate::py::truthy(e)) {
        return Err(JobError::engine(
            format!(
                "{} upscale failed: {}",
                sidecar.engine,
                crate::py::py_str(e)
            ),
            sidecar.tail(TAIL),
        ));
    }
    let up_factor = status
        .get("upscale")
        .filter(|v| crate::py::truthy(v))
        .map(|v| i64_of(Some(v)));
    Ok(ServeClip {
        rgb: rgb_path,
        width: clip.width,
        height: clip.height,
        frames: clip.frames,
        audio_wav: audio,
        hd_jpegs: hd,
        up_factor,
        status,
        timings: t,
    })
}

/// Run an engine CLI, turning its `step i/n` / `block b:` lines into
/// progress. Returns the last log lines and the timings parsed.
pub fn run_cli(
    argv: &[String],
    env: &HashMap<String, String>,
    report: &Reporter,
    log: &Log,
    timeout_s: f64,
    cwd: Option<&Path>,
) -> JobResult<(Vec<String>, Timings)> {
    log(&format!("run: {}", argv.join(" ")));
    let mut t = Timings::default();
    let mut tail: VecDeque<String> = VecDeque::with_capacity(400);
    let t0 = now_s();
    let (reader, writer) = std::io::pipe()?;
    let writer2 = writer.try_clone()?;
    let mut cmd = Command::new(&argv[0]);
    cmd.args(&argv[1..])
        .env_clear()
        .envs(env)
        .stdin(Stdio::null())
        .stdout(Stdio::from(writer))
        .stderr(Stdio::from(writer2));
    if let Some(c) = cwd {
        cmd.current_dir(c);
    }
    std::os::unix::process::CommandExt::process_group(&mut cmd, 0);
    let mut proc = cmd
        .spawn()
        .map_err(|e| JobError::engine(format!("{}: {e}", argv[0]), vec![]))?;
    drop(cmd); // the pipe's write ends live in `cmd`: drop them, or EOF never comes
    report.detail("preparing", "loading the model");
    let mut frames_done = 0i64;
    let push = |tail: &mut VecDeque<String>, line: String| {
        if tail.len() >= 400 {
            tail.pop_front();
        }
        tail.push_back(line);
    };
    let outcome = (|| -> JobResult<i32> {
        for line in BufReader::new(reader).split(b'\n').map_while(|l| l.ok()) {
            let line = String::from_utf8_lossy(&line)
                .trim_end_matches('\r')
                .to_string();
            push(&mut tail, line.clone());
            log(&format!("  {line}"));
            if let Some((i, n, secs)) = parse_step(&line) {
                t.steps_s.push(secs);
                report.report(
                    "denoising",
                    Detail {
                        detail: Some(&format!("step {i}/{n}")),
                        step: Some(i),
                        total_steps: Some(n),
                        force: i == n,
                        ..Default::default()
                    },
                );
                continue;
            }
            if let Some((b, den, dec, _frames, total)) = parse_block(&line) {
                t.steps_s.push(den);
                t.decode_s += dec;
                frames_done = total;
                report.report(
                    "denoising",
                    Detail {
                        detail: Some(&format!("block {}", b + 1)),
                        frames: Some(frames_done),
                        ..Default::default()
                    },
                );
                continue;
            }
            let low = line.to_lowercase();
            if low.contains("decode") && !low.contains("step") {
                report.detail(
                    "decoding",
                    &line.trim().chars().take(80).collect::<String>(),
                );
            }
            if now_s() - t0 > timeout_s {
                let _ = proc.kill();
                return Err(JobError::engine(
                    format!("{} timed out after {timeout_s:.0} s", argv[0]),
                    tail.iter()
                        .rev()
                        .take(50)
                        .cloned()
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect(),
                ));
            }
        }
        let status = proc.wait()?;
        Ok(status.code().unwrap_or(-1))
    })();
    if matches!(proc.try_wait(), Ok(None)) {
        let _ = proc.kill();
        let _ = proc.wait();
    }
    let code = outcome?;
    let last50: Vec<String> = tail
        .iter()
        .rev()
        .take(50)
        .cloned()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    if code != 0 {
        let name = argv[0].rsplit('/').next().unwrap_or(&argv[0]);
        return Err(JobError::engine(
            format!(
                "{name} {} exited with code {code}",
                argv.get(1).map(String::as_str).unwrap_or("")
            ),
            last50,
        ));
    }
    t.denoise_s = t.steps_s.iter().sum();
    t.total_s = now_s() - t0;
    Ok((tail.into_iter().collect(), t))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_and_block_lines_parse() {
        assert_eq!(
            parse_step("step 3/8 2.512s (eta 12 s)"),
            Some((3, 8, 2.512))
        );
        assert_eq!(parse_step("  step 8/8 1.0s"), Some((8, 8, 1.0)));
        assert_eq!(parse_step("stepping"), None);
        assert_eq!(
            parse_block("block 2: denoise 0.84s  decode 0.10s  12 frames (36 total) 14.2 fps"),
            Some((2, 0.84, 0.10, 12, 36))
        );
        assert_eq!(parse_block("block x: denoise"), None);
        assert_eq!(engine_stage("weird"), "denoising");
    }

    #[test]
    fn run_cli_streams_steps_and_fails_on_exit_code() {
        let got = std::sync::Mutex::new(Vec::new());
        let send = |v: &Value| got.lock().unwrap().push(v.clone());
        let rep = Reporter::with_interval(Some(&send), "h3", "m", 0.0);
        let argv: Vec<String> = ["/bin/sh", "-c", "echo 'step 1/2 0.5s'; echo 'step 2/2 0.7s'; echo 'decode 1.1s'; echo 'block 0: denoise 0.5s  decode 0.1s  4 frames (4 total)'"].iter().map(|s| s.to_string()).collect();
        let env: HashMap<String, String> = HashMap::new();
        let (tail, t) = run_cli(&argv, &env, &rep, &crate::null_log(), 60.0, None).unwrap();
        assert_eq!(tail.len(), 4);
        assert_eq!(t.steps_s, vec![0.5, 0.7, 0.5]);
        assert!((t.denoise_s - 1.7).abs() < 1e-9 && (t.decode_s - 0.1).abs() < 1e-9);
        let stages: Vec<String> = got
            .lock()
            .unwrap()
            .iter()
            .map(|v| v["stage"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            stages,
            [
                "preparing",
                "denoising",
                "denoising",
                "decoding",
                "denoising"
            ]
        );
        let bad: Vec<String> = ["/bin/sh", "-c", "echo oops; exit 2"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let e = run_cli(&bad, &env, &rep, &crate::null_log(), 60.0, None).unwrap_err();
        assert_eq!(
            (e.where_, e.message.as_str()),
            ("engine", "sh -c exited with code 2")
        );
        assert_eq!(e.log_tail, ["oops"]);
    }
}
