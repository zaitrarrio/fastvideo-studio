//! fastvideo-rs: one resident `fv-serve` per process group (a family:
//! swap = true, the requested model resident, the family's others loaded on
//! demand) driven through its native jobs API `/fv/v1/jobs`.

use super::{Ctx, RunResult};
use crate::catalog::{Job, Media};
use crate::errors::{JobError, JobResult};
use crate::flags::{self, MediaUrls};
use crate::httpc;
use crate::media as md;
use crate::progress::{Detail, Timings};
use crate::py::{dumps, py_str, shlex_quote};
use crate::sidecar::{ping_probe, Sidecar, TAIL};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};
use std::time::Duration;

pub const ENGINE: &str = "fastvideo";
pub const KEY: &str = "fv-serve";
pub const POLL_S: f64 = 1.0;
pub const PRESIGN_S: u64 = 3600;

/// https URLs pass through; R2 uploads become presigned GETs; inline
/// bytes a data URI (fv-serve takes both).
pub fn media_url(ctx: &Ctx, m: &Media, name: &str) -> JobResult<String> {
    if m.kind == "url" {
        return Ok(m.value.clone());
    }
    if m.kind == "r2" {
        let (Some(store), Some(bucket)) = (
            ctx.store_ref(),
            ctx.bucket.as_deref().filter(|b| !b.is_empty()),
        ) else {
            return Err(JobError::validate(format!(
                "{name}: r2 uploads need the Studio bucket (no R2 credentials)"
            )));
        };
        return Ok(store.presign_get(bucket, &m.value, PRESIGN_S));
    }
    let lm = md::fetch_media(
        m,
        &ctx.job_dir.join("in"),
        name,
        ctx.store_ref(),
        ctx.bucket.as_deref(),
    )?;
    Ok(md::to_data_uri(&lm)?)
}

pub fn media_urls(ctx: &Ctx, job: &Job) -> JobResult<MediaUrls> {
    let mut out = MediaUrls::default();
    for k in &job.inputs.order {
        if k == "references" {
            for (i, m) in job.inputs.references.iter().enumerate() {
                out.references.push(media_url(ctx, m, &format!("ref{i}"))?);
            }
            continue;
        }
        let u = media_url(ctx, job.inputs.get(k).unwrap(), k)?;
        match k.as_str() {
            "first_image" => out.first_image = Some(u),
            "last_image" => out.last_image = Some(u),
            "audio" => out.audio = Some(u),
            "video" => out.video = Some(u),
            _ => {}
        }
    }
    Ok(out)
}

/// The `fv-serve` process a model runs in (`recipe.process` in the catalog,
/// else the variant alone): fv-serve refuses one process for models that
/// need different process-wide settings (H3 wants `FASTVIDEO_BF16_ACT=1`,
/// LTX does not; FastWan 2.1's VSA flag is process-wide, the Wan 2.2 TI2V
/// recipe runs without it), and it serves every model of one family from
/// one process with `[engine] swap = true` (fastvideo-rs's own
/// `runpod-family-{h3,ltx,streaming}.toml`), loading a non-resident model on
/// its first request and swapping the resident DiT out.
pub fn process_of(model: &crate::catalog::Model) -> String {
    model
        .recipe
        .get("process")
        .map(py_str)
        .filter(|p| !p.is_empty())
        .unwrap_or_else(|| model.recipe.get("variant").map(py_str).unwrap_or_default())
}

/// The variants that share `variant`'s process, the requested one first;
/// `[variant]` alone when the catalog is not at hand (unit tests).
pub fn group_members(models: &[crate::catalog::Model], variant: &str) -> Vec<String> {
    let mine = models.iter().find(|m| {
        m.engine == "fastvideo" && m.recipe.get("variant").map(py_str).as_deref() == Some(variant)
    });
    let mut out = vec![variant.to_string()];
    if let Some(mine) = mine {
        let process = process_of(mine);
        for m in models {
            if m.engine != "fastvideo" || process_of(m) != process {
                continue;
            }
            let v = m.recipe.get("variant").map(py_str).unwrap_or_default();
            if !v.is_empty() && !out.contains(&v) {
                out.push(v);
            }
        }
    }
    out
}

/// The worker's config with the process group's models only, `resident`
/// marked `resident = true` (fv-serve refuses a config where nothing loads
/// before readiness) and the others `resident = false` (loaded on demand,
/// `[engine] swap = true` in the base).
pub fn group_config(base: &str, members: &[String], resident: &str) -> JobResult<String> {
    let mut parts = base.split("[[models]]");
    let head = parts.next().unwrap_or("");
    let variant_of = |b: &str| -> Option<String> {
        b.lines().find_map(|l| {
            let t = l.trim();
            let rest = t
                .strip_prefix("variant")?
                .trim_start()
                .strip_prefix('=')?
                .trim();
            rest.strip_prefix('"')?
                .split('"')
                .next()
                .map(str::to_string)
        })
    };
    let mut keep: Vec<String> = Vec::new();
    let mut found = false;
    for b in parts {
        let Some(v) = variant_of(b) else { continue };
        if !members.contains(&v) {
            continue;
        }
        let is_resident = v == resident;
        found |= is_resident;
        let want = if is_resident {
            "resident = true"
        } else {
            "resident = false"
        };
        let mut lines: Vec<String> = Vec::new();
        let mut seen = false;
        for l in b.lines() {
            if l.trim_start().starts_with("resident")
                && l.trim_start()[8..].trim_start().starts_with('=')
            {
                lines.push(want.into());
                seen = true;
            } else {
                lines.push(l.to_string());
            }
        }
        let mut k = lines.join("\n");
        if !seen {
            k = format!("{}\n{want}\n", k.trim_end_matches('\n'));
        }
        keep.push(format!("[[models]]{}\n", k.trim_end_matches('\n')));
    }
    if !found {
        return Err(JobError::validate(format!(
            "fastvideo config lists no model {}",
            crate::py::py_str_repr(resident)
        )));
    }
    Ok(format!(
        "{}\n\n{}",
        head.trim_end_matches('\n'),
        keep.join("\n")
    ))
}

pub fn serve_argv(ctx: &Ctx, variant: Option<&str>) -> JobResult<Vec<String>> {
    let Some(variant) = variant else {
        return Ok(vec![
            ctx.fv_serve_bin.clone(),
            "--config".into(),
            ctx.fv_config.clone(),
        ]);
    };
    let cfg = PathBuf::from(&ctx.fv_state_dir).join("fastvideo.toml");
    if let Some(p) = cfg.parent() {
        std::fs::create_dir_all(p)?;
    }
    let base = std::fs::read_to_string(&ctx.fv_config)?;
    std::fs::write(
        &cfg,
        group_config(&base, &group_members(ctx.models, variant), variant)?,
    )?;
    Ok(vec![
        ctx.fv_serve_bin.clone(),
        "--config".into(),
        cfg.to_string_lossy().into_owned(),
    ])
}

/// The resident `fv-serve` for `model`'s process group (started when it is
/// not up: the group's config with `model` resident), its base URL and the
/// load it cost. Jobs and the pod's live streams / segments share it.
pub fn ensure_serve(
    ctx: &Ctx,
    model: &crate::catalog::Model,
) -> JobResult<(String, Option<Sidecar>, f64)> {
    let port = ctx.port(ENGINE);
    let base = format!("http://127.0.0.1:{port}");
    if ctx.fv_fake {
        return Ok((base, None, 0.0));
    }
    let variant = model.recipe.get("variant").map(py_str).unwrap_or_default();
    let key = format!("{KEY}:{}", process_of(model)); // one fv-serve per process group (a family: its models swap inside it)
    let argv = serve_argv(ctx, Some(&variant))?;
    let env = ctx.env_of(ENGINE);
    let make = || {
        Sidecar::new(
            ENGINE,
            &key,
            argv.clone(),
            env.clone(),
            ping_probe(&format!("{base}/ping")),
        )
    };
    let loading = |_: &str, t: f64| ctx.report.detail("loading", &format!("{t:.0} s"));
    let (s, l) = ctx
        .sidecars
        .ensure(ENGINE, &key, &make, model.vram_gib, Some(&loading))?;
    // A model switch inside fv-serve (swap) needs the VRAM too.
    ctx.sidecars.make_room(ENGINE, model.vram_gib);
    Ok((base, Some(s), l))
}

pub fn run(ctx: &Ctx, job: &Job) -> JobResult<RunResult> {
    let (base, sc, load_s) = ensure_serve(ctx, &job.model)?;
    let tail = |sc: &Option<Sidecar>| sc.as_ref().map(|s| s.tail(TAIL)).unwrap_or_default();
    let urls = media_urls(ctx, job)?;
    let body = Value::Object(flags::fastvideo_body(job, ctx.knobs, &urls, job.seed)?);
    let command = format!(
        "curl -sS -X POST {base}/fv/v1/jobs -H 'content-type: application/json' -d {}",
        shlex_quote(&json_short(&body))
    );
    let j = match httpc::post_json(&format!("{base}/fv/v1/jobs"), &body, 60.0) {
        Ok(Some(j)) => j,
        Ok(None) => {
            return Err(JobError::engine(
                "fv-serve refused the job: empty answer",
                tail(&sc),
            ))
        }
        Err(httpc::Error::Status(e)) => {
            let msg = e
                .json()
                .and_then(|j| j.get("error").cloned())
                .and_then(|err| err.get("message").map(py_str))
                .unwrap_or_else(|| e.to_string());
            return Err(JobError::engine(
                format!("fv-serve refused the job: {msg}"),
                tail(&sc),
            ));
        }
        Err(e) => {
            return Err(JobError::engine(
                format!("fv-serve refused the job: {e}"),
                tail(&sc),
            ))
        }
    };
    let jid = py_str(&j["id"]);
    let t0 = crate::now_s();
    let mut j: Value;
    loop {
        if let Some(s) = &sc {
            if !s.alive() {
                return Err(JobError::engine(
                    "fv-serve died during the job",
                    s.tail(TAIL),
                ));
            }
        }
        j = httpc::get_json(&format!("{base}/fv/v1/jobs/{jid}"), 30.0)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let st = j.get("status").and_then(|s| s.as_str()).unwrap_or("");
        let p = j.get("progress").and_then(|p| p.as_f64()).unwrap_or(0.0);
        match st {
            "queued" => {
                let pos = j
                    .get("queue_position")
                    .filter(|v| crate::py::truthy(v))
                    .map(|v| format!("position {}", py_str(v)));
                ctx.report.report(
                    "queued",
                    Detail {
                        detail: pos.as_deref(),
                        ..Default::default()
                    },
                );
            }
            "running" => {
                let steps = job.model.steps.filter(|s| *s != 0);
                ctx.report.report(
                    "denoising",
                    Detail {
                        detail: Some(&format!("{} %", (p * 100.0) as i64)),
                        step: steps.map(|s| (p * s as f64).round() as i64),
                        total_steps: steps,
                        ..Default::default()
                    },
                );
            }
            "succeeded" => break,
            "failed" | "cancelled" => {
                let err = j.get("error").cloned().unwrap_or(Value::Null);
                let kind = err
                    .get("kind")
                    .map(py_str)
                    .unwrap_or_else(|| st.to_string());
                let msg = err.get("message").map(py_str).unwrap_or_default();
                return Err(JobError::engine(
                    format!("fv-serve: {kind}: {msg}"),
                    tail(&sc),
                ));
            }
            _ => {}
        }
        std::thread::sleep(Duration::from_secs_f64(POLL_S));
    }
    let out = j
        .get("output")
        .cloned()
        .filter(|o| o.is_object())
        .unwrap_or(Value::Object(Map::new()));
    let Some(url) = out.get("url").filter(|u| crate::py::truthy(u)).map(py_str) else {
        return Err(JobError::engine(
            "fv-serve reported success without an output",
            tail(&sc),
        ));
    };
    ctx.report.detail_force("decoding", "downloading the clip");
    let mp4 = ctx.job_dir.join("fv.mp4");
    std::fs::create_dir_all(&ctx.job_dir)?;
    httpc::download(&url, &mp4, 300.0, None).map_err(|e| anyhow::anyhow!("{e}"))?;
    let m = j.get("metrics").cloned().unwrap_or(Value::Null);
    let stages = m.get("stage_durations").cloned().unwrap_or(Value::Null);
    let first = |keys: &[&str]| {
        keys.iter()
            .find_map(|k| {
                stages
                    .get(*k)
                    .and_then(|v| v.as_f64())
                    .filter(|v| *v != 0.0)
            })
            .unwrap_or(0.0)
    };
    let t = Timings {
        load_s,
        denoise_s: m.get("inference_s").and_then(|v| v.as_f64()).unwrap_or(0.0),
        total_s: crate::now_s() - t0,
        encode_s: first(&["text_encoder", "encode", "prompt"]),
        decode_s: first(&["vae_decode", "decode", "vae"]),
        ..Default::default()
    };
    let int = |a: Option<&Value>, b: Option<&Value>, d: i64| {
        a.and_then(|v| v.as_f64())
            .filter(|v| *v != 0.0)
            .or_else(|| b.and_then(|v| v.as_f64()).filter(|v| *v != 0.0))
            .map(|f| f as i64)
            .unwrap_or(d)
    };
    let seed = match j
        .get("seed")
        .filter(|s| !s.is_null())
        .and_then(|s| s.as_f64())
    {
        Some(s) => s as i64,
        None => job.seed.unwrap_or(0),
    };
    let mut res = RunResult::new(
        seed,
        int(j.get("width"), out.get("width"), job.width),
        int(j.get("height"), out.get("height"), job.height),
        int(j.get("num_frames"), out.get("frames"), job.frames),
        int(j.get("fps"), out.get("fps"), job.fps_for()),
    );
    res.video_mp4 = Some(mp4);
    res.timings = t;
    res.command = command;
    res.extra.insert("fv_job".into(), json!(jid));
    res.extra.insert(
        "resolved_model".into(),
        j.get("resolved_model").cloned().unwrap_or(Value::Null),
    );
    res.extra.insert(
        "audio".into(),
        out.get("audio").cloned().unwrap_or(Value::Null),
    );
    Ok(res)
}

fn json_short(body: &Value) -> String {
    let mut short = Map::new();
    for (k, v) in body.as_object().unwrap() {
        let v = match v {
            Value::String(s) if s.starts_with("data:") => {
                Value::String(format!("{}…", s.chars().take(64).collect::<String>()))
            }
            Value::Array(a) if k == "reference_urls" => Value::Array(
                a.iter()
                    .map(|u| match u {
                        Value::String(s) if s.starts_with("data:") => {
                            Value::String(format!("{}…", s.chars().take(64).collect::<String>()))
                        }
                        other => other.clone(),
                    })
                    .collect(),
            ),
            other => other.clone(),
        };
        short.insert(k.clone(), v);
    }
    dumps(&Value::Object(short), None, true)
}

pub fn fv_config_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../worker/configs/fastvideo.toml")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_config_keeps_the_process_group_with_the_requested_model_resident() {
        let base = std::fs::read_to_string(fv_config_path()).unwrap();
        let cat = crate::catalog::Catalog::load(None).unwrap();
        let members = group_members(&cat.models, "fastwan22-ti2v-5b");
        assert_eq!(
            members,
            [
                "fastwan22-ti2v-5b",
                "wan22-ti2v-5b",
                "fastwan22-ti2v-5b-taehv"
            ]
        );
        let out = group_config(&base, &members, "fastwan22-ti2v-5b").unwrap();
        let parsed: toml::Value = toml::from_str(&out).unwrap();
        let models = parsed["models"].as_array().unwrap();
        assert_eq!(models.len(), 3);
        for m in models {
            let v = m["variant"].as_str().unwrap();
            assert_eq!(
                m["resident"],
                toml::Value::Boolean(v == "fastwan22-ti2v-5b"),
                "{v}"
            );
            assert!(members.contains(&v.to_string()));
        }
        assert!(out.contains("swap = true") && out.contains("bind = \"127.0.0.1:8000\""));
        assert!(!out.contains("variant = \"fasth3-4step-vsa\"")); // H3 and LTX / Wan cannot share a process
        assert!(!out.contains("variant = \"fastwan21-1.3b\"")); // FastWan 2.1's VSA flag is process-wide
                                                                // Every H3 model is one process, as fastvideo-rs's runpod-family-h3.toml.
        let h3 = group_members(&cat.models, "sol-h3");
        assert_eq!(
            h3.len(),
            cat.models
                .iter()
                .filter(|m| m.engine == "fastvideo" && m.family == "fv-h3")
                .count()
        );
        assert_eq!(h3[0], "sol-h3");
        // Without a catalog the group is the model alone (the old behaviour).
        let alone = group_config(&base, &group_members(&[], "ltx25-ref2v"), "ltx25-ref2v").unwrap();
        assert_eq!(alone.split("[[models]]").count(), 2);
        assert!(group_config(&base, &["no-such-model".to_string()], "no-such-model").is_err());
    }

    #[test]
    fn json_short_truncates_data_uris() {
        let b = json!({"model": "m", "image_url": format!("data:image/png;base64,{}", "A".repeat(200)), "reference_urls": ["https://e/a.png", format!("data:x;base64,{}", "B".repeat(100))]});
        let s = json_short(&b);
        assert!(
            s.contains("\"model\": \"m\"")
                && s.contains("\\u2026\"")
                && !s.contains(&"A".repeat(100))
        );
    }
}
