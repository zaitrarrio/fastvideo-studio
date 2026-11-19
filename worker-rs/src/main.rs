//! RunPod serverless handler for fastvideo-studio (docs/design.md §1,
//! docs/jobs.md): the Rust port of `worker/handler.py`. One worker process,
//! concurrency 1, the engines as sidecars. The job loop is `fvstudio::rp`
//! (RunPod's protocol, no SDK); `--test-input FILE` runs one job locally.

use fvstudio::rp;
use fvstudio::worker::Worker;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

fn usage() -> ! {
    eprintln!("usage: studio-worker [serve | --test-input FILE]\n\n  serve              the pod mode: the job pipeline over HTTP (docs/pod-mode.md)\n  --test-input FILE  run this one job ({{\"input\": …}}) locally and exit, as the SDK's --test_input did");
    std::process::exit(2)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut test_input: Option<String> = None;
    let mut serve_mode = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "serve" => serve_mode = true,
            "--test-input" | "--test_input" => {
                test_input = Some(args.next().unwrap_or_else(|| usage()))
            }
            "-h" | "--help" => {
                println!("fastvideo-studio RunPod worker\n\nusage: studio-worker [serve | --test-input FILE]\n\n  serve              the pod mode: the job pipeline over HTTP (docs/pod-mode.md)\n  --test-input FILE  run this one job ({{\"input\": …}}) locally and exit, as the SDK's --test_input did");
                return;
            }
            s if s.starts_with("--test-input=") || s.starts_with("--test_input=") => {
                test_input = Some(s.split_once('=').unwrap().1.to_string())
            }
            _ => usage(),
        }
    }
    let env_map = fvstudio::process_env();
    let log = fvstudio::stderr_log();
    let runpod = match rp::RunpodEnv::from_env(&env_map) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("worker: {e}");
            std::process::exit(2);
        }
    };
    let worker = match Worker::from_env_map(&env_map, log.clone()) {
        Ok(w) => Arc::new(w),
        Err(e) => {
            eprintln!("worker: {e:#}");
            std::process::exit(2);
        }
    };
    if serve_mode {
        let cfg = match fvstudio::serve::ServeConfig::from_env(&env_map) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(2);
            }
        };
        log(&format!(
            "fastvideo-studio pod: catalog {} ({} models), image {}",
            worker.cfg.catalog_path.display(),
            worker.catalog.models.len(),
            if worker.cfg.image.is_empty() {
                "?"
            } else {
                &worker.cfg.image
            }
        ));
        let stop = Arc::new(AtomicBool::new(false));
        rp::install_signal_handlers(stop.clone(), log.clone());
        if let Err(e) = fvstudio::serve::serve(worker.clone(), cfg, log.clone(), stop, None) {
            eprintln!("{e}");
            std::process::exit(1);
        }
        return;
    }
    if test_input.is_some() || runpod.is_none() {
        let w = worker.clone();
        let handler = move |job: &serde_json::Value| {
            let id = fvstudio::py::py_str(&job["id"]);
            let progress = move |rec: &serde_json::Value| {
                rp::stderr(&format!(
                    "progress {id}: {}",
                    fvstudio::py::dumps(rec, None, false)
                ))
            };
            w.handle(job, Some(&progress))
        };
        let code = rp::run_local(
            &handler,
            test_input.as_deref(),
            &log,
            &mut std::io::stdout(),
        );
        worker.sidecars.stop_all();
        std::process::exit(code);
    }
    let client = rp::RunpodClient::new(
        runpod.unwrap(),
        log.clone(),
        Arc::new(|s| std::thread::sleep(std::time::Duration::from_secs_f64(s))),
    );
    let w = worker.clone();
    let c = client.clone();
    let handler = move |job: &serde_json::Value| {
        let id = fvstudio::py::py_str(&job["id"]);
        let c = c.clone();
        let progress = move |rec: &serde_json::Value| c.progress_update(&id, rec);
        w.handle(job, Some(&progress))
    };
    log(&format!(
        "fastvideo-studio worker: catalog {} ({} models), image {}",
        worker.cfg.catalog_path.display(),
        worker.catalog.models.len(),
        if worker.cfg.image.is_empty() {
            "?"
        } else {
            &worker.cfg.image
        }
    ));
    let stop = Arc::new(AtomicBool::new(false));
    rp::install_signal_handlers(stop.clone(), log.clone());
    rp::serve(&handler, &client, stop, None, 0.2);
    worker.sidecars.stop_all();
}
