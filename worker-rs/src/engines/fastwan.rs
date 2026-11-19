//! fastwan-rs: `fastwan serve` for clips, `fastwan generate --causal` per
//! job for long videos (Self-Forcing).

use super::common::{run_cli, run_serve_job};
use super::{random_seed, Ctx, RunResult};
use crate::catalog::Job;
use crate::errors::{JobError, JobResult};
use crate::flags;
use crate::frames as fr;
use crate::sidecar::{http_info_probe, Sidecar};
use serde_json::json;
use std::path::Path;

pub const ENGINE: &str = "fastwan";

pub fn run(ctx: &Ctx, job: &Job) -> JobResult<RunResult> {
    let st = ctx.settings;
    let seed = job.seed.unwrap_or_else(random_seed);
    let out = ctx.job_dir.join("out.y4m");
    let command = flags::command_line(&flags::fastwan_generate_argv(
        job,
        st,
        ctx.knobs,
        Path::new("out.y4m"),
        seed,
    )?);
    if job.model.cli_only || job.task == "long" {
        ctx.sidecars.stop_all();
        ctx.report.detail_force("loading", "fastwan (per-job CLI)");
        let argv = flags::fastwan_generate_argv(job, st, ctx.knobs, &out, seed)?;
        let (_, t) = run_cli(
            &argv,
            &ctx.env_of(ENGINE),
            ctx.report,
            &ctx.log,
            7200.0,
            Some(&ctx.job_dir),
        )?;
        if !out.is_file() {
            return Err(JobError::engine("fastwan wrote no out.y4m", vec![]));
        }
        let info = fr::read_y4m_info(&out).map_err(|e| anyhow::anyhow!("ValueError: {e}"))?;
        let mut res = RunResult::new(
            seed,
            info.width as i64,
            info.height as i64,
            info.frames as i64,
            job.fps_for(),
        );
        res.video_y4m = Some(out);
        res.timings = t;
        res.command = command;
        if job.upscale != 0 && (info.width as i64, info.height as i64) != (job.width, job.height) {
            res.width = job.width;
            res.height = job.height;
            res.upscaled = Some((job.upscale, info.width as i64, info.height as i64));
        }
        return Ok(res);
    }
    let port = ctx.port(ENGINE);
    let base = format!("http://127.0.0.1:{port}");
    let key = flags::fastwan_serve_key(job, st, ctx.knobs)?;
    let serve_argv = flags::fastwan_serve_argv(job, st, ctx.knobs, port)?;
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
    let (sc, load_s) =
        ctx.sidecars
            .ensure(ENGINE, &key, &make, job.model.vram_gib, Some(&loading))?;
    let body = json!({"prompt": job.prompt, "width": job.width, "height": job.height, "frames": job.frames, "seed": seed, "upscale": job.upscale});
    let clip = run_serve_job(
        &base,
        &sc,
        &body,
        &ctx.job_dir,
        ctx.report,
        false,
        0.5,
        3600.0,
    )?;
    ctx.report.detail_force("muxing", "frames → y4m");
    let fps = job.fps_for();
    fr::rgb_to_y4m(&clip.rgb, clip.width, clip.height, fps, &out, None)
        .map_err(|e| anyhow::anyhow!("RuntimeError: {e}"))?;
    let mut res = RunResult::new(seed, job.width, job.height, clip.frames as i64, fps);
    res.video_y4m = Some(out);
    res.timings = clip.timings.clone();
    res.command = command;
    res.timings.load_s = load_s;
    if job.upscale != 0 {
        res.upscaled = Some((job.upscale, clip.width as i64, clip.height as i64));
    }
    Ok(res)
}
