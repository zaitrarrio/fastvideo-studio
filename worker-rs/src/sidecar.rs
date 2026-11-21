//! Resident engine processes (`h3 serve`, `fastwan serve`, `fv-serve`,
//! `llama-server`) on 127.0.0.1: start, wait for ready, keep the last log
//! lines, stop. One sidecar per engine, keyed by its argv so a job for
//! another model restarts that engine's process. Before a start the free VRAM
//! (nvidia-smi) is checked against the model's `vram_gib` and the other
//! engines are stopped when it does not fit (design §1).

use crate::errors::{JobError, JobResult};
use crate::gpu::{self, GpuInfo};
use crate::{httpc, now_s, Log};
use serde_json::{json, Map, Value};
use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// (ready, fatal error)
pub type Probe = Arc<dyn Fn() -> (bool, Option<String>) + Send + Sync>;
pub const LOG_KEEP: usize = 400;
pub const TAIL: usize = 50;

/// `h3 serve` / `fastwan serve`: `/api/info` → ready / error.
pub fn http_info_probe(url: &str) -> Probe {
    let url = url.to_string();
    Arc::new(move || {
        let Ok(info) = httpc::get_json(&url, 5.0) else {
            return (false, None);
        };
        if let Some(e) = info.get("error").filter(|e| crate::py::truthy(e)) {
            return (false, Some(crate::py::py_str(e)));
        }
        (
            info.get("ready").map(crate::py::truthy).unwrap_or(false),
            None,
        )
    })
}

/// fv-serve `/ping`: 204 loading, 200 ready, 503 failed.
pub fn ping_probe(url: &str) -> Probe {
    let url = url.to_string();
    Arc::new(move || {
        let Ok((status, body)) = httpc::get_status(&url, 5.0) else {
            return (false, None);
        };
        match status {
            200 => (true, None),
            503 => {
                let text: String = String::from_utf8_lossy(&body).chars().take(300).collect();
                // fv-serve answers `{"status":"failed"}` when its resident
                // model's load failed: say so, the log tail carries the why.
                let failed = text.contains("\"failed\"");
                (
                    false,
                    Some(if failed {
                        format!("its model load failed ({text})")
                    } else if text.is_empty() {
                        "fv-serve reports 503".into()
                    } else {
                        text
                    }),
                )
            }
            _ => (false, None),
        }
    })
}

extern "C" {
    fn kill(pid: i32, sig: i32) -> i32;
}

const SIGTERM: i32 = 15;
const SIGKILL: i32 = 9;

fn killpg(pid: u32, sig: i32) {
    // SAFETY: a plain libc call; a missing group is ignored (ESRCH).
    unsafe {
        kill(-(pid as i32), sig);
    }
}

struct Inner {
    proc: Option<Child>,
    lines: VecDeque<String>,
    started_at: f64,
    ready_at: Option<f64>,
}

/// A resident process; clones share it (the manager and a driver polling it).
#[derive(Clone)]
pub struct Sidecar {
    pub engine: String,
    /// what it was started for (model / recipe); a different key restarts
    pub key: String,
    pub argv: Vec<String>,
    pub env: HashMap<String, String>,
    pub probe: Probe,
    pub log: Log,
    inner: Arc<Mutex<Inner>>,
}

impl Sidecar {
    pub fn new(
        engine: &str,
        key: &str,
        argv: Vec<String>,
        env: HashMap<String, String>,
        probe: Probe,
    ) -> Sidecar {
        Sidecar {
            engine: engine.into(),
            key: key.into(),
            argv,
            env,
            probe,
            log: crate::null_log(),
            inner: Arc::new(Mutex::new(Inner {
                proc: None,
                lines: VecDeque::with_capacity(LOG_KEEP),
                started_at: 0.0,
                ready_at: None,
            })),
        }
    }

    pub fn start(&self) -> JobResult<()> {
        (self.log)(&format!("[{}] start: {}", self.engine, self.argv.join(" ")));
        let (reader, writer) = std::io::pipe()
            .map_err(|e| JobError::engine(format!("{}: pipe: {e}", self.engine), vec![]))?;
        let writer2 = writer
            .try_clone()
            .map_err(|e| JobError::engine(format!("{}: pipe: {e}", self.engine), vec![]))?;
        let mut cmd = Command::new(&self.argv[0]);
        cmd.args(&self.argv[1..])
            .env_clear()
            .envs(&self.env)
            .stdin(Stdio::null())
            .stdout(Stdio::from(writer))
            .stderr(Stdio::from(writer2));
        std::os::unix::process::CommandExt::process_group(&mut cmd, 0);
        let child = cmd.spawn().map_err(|e| {
            JobError::engine(
                format!("{} failed to start: {}: {e}", self.engine, self.argv[0]),
                vec![],
            )
        })?;
        drop(cmd); // the pipe's write ends live in `cmd`: drop them, or EOF never comes
        let mut inner = self.inner.lock().unwrap();
        inner.proc = Some(child);
        inner.started_at = now_s();
        inner.ready_at = None;
        drop(inner);
        let (inner, log, engine) = (
            Arc::clone(&self.inner),
            self.log.clone(),
            self.engine.clone(),
        );
        std::thread::Builder::new()
            .name(format!("{engine}-log"))
            .spawn(move || {
                for line in BufReader::new(reader).split(b'\n').map_while(|l| l.ok()) {
                    let line = String::from_utf8_lossy(&line)
                        .trim_end_matches('\r')
                        .to_string();
                    let mut g = inner.lock().unwrap();
                    if g.lines.len() >= LOG_KEEP {
                        g.lines.pop_front();
                    }
                    g.lines.push_back(line.clone());
                    drop(g);
                    log(&format!("[{engine}] {line}"));
                }
            })
            .expect("log thread");
        Ok(())
    }

    pub fn alive(&self) -> bool {
        let mut g = self.inner.lock().unwrap();
        match g.proc.as_mut() {
            Some(p) => matches!(p.try_wait(), Ok(None)),
            None => false,
        }
    }

    pub fn pid(&self) -> Option<u32> {
        self.inner.lock().unwrap().proc.as_ref().map(|p| p.id())
    }

    pub fn exit_code(&self) -> Option<i32> {
        let mut g = self.inner.lock().unwrap();
        g.proc
            .as_mut()
            .and_then(|p| p.try_wait().ok().flatten())
            .and_then(|s| s.code())
    }

    pub fn started_at(&self) -> f64 {
        self.inner.lock().unwrap().started_at
    }

    pub fn ready_at(&self) -> Option<f64> {
        self.inner.lock().unwrap().ready_at
    }

    pub fn tail(&self, n: usize) -> Vec<String> {
        let g = self.inner.lock().unwrap();
        g.lines
            .iter()
            .rev()
            .take(n)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }

    /// Tests: `proc.kill()` + `proc.wait()`.
    pub fn kill_now(&self) {
        let mut g = self.inner.lock().unwrap();
        if let Some(p) = g.proc.as_mut() {
            let _ = p.kill();
            let _ = p.wait();
        }
    }

    /// Seconds until ready; `JobError("engine")` with the log tail on a
    /// crash, a reported error or the timeout.
    pub fn wait_ready(&self, timeout_s: f64, progress: Option<&dyn Fn(f64)>) -> JobResult<f64> {
        let t0 = now_s();
        let mut last = 0.0;
        loop {
            if !self.alive() {
                let code = self.exit_code();
                return Err(JobError::engine(
                    format!(
                        "{} sidecar exited with code {} while loading",
                        self.engine,
                        code.map(|c| c.to_string()).unwrap_or_else(|| "None".into())
                    ),
                    self.tail(TAIL),
                ));
            }
            let (ready, err) = (self.probe)();
            if let Some(err) = err {
                self.stop(15.0);
                return Err(JobError::engine(
                    format!("{} failed to start: {err}", self.engine),
                    self.tail(TAIL),
                ));
            }
            if ready {
                let now = now_s();
                self.inner.lock().unwrap().ready_at = Some(now);
                return Ok(now - t0);
            }
            let now = now_s();
            if now - t0 > timeout_s {
                self.stop(15.0);
                return Err(JobError::engine(
                    format!("{} not ready after {timeout_s:.0} s", self.engine),
                    self.tail(TAIL),
                ));
            }
            if let Some(p) = progress {
                if now - last >= 5.0 {
                    last = now;
                    p(now - t0);
                }
            }
            std::thread::sleep(Duration::from_millis(500));
        }
    }

    pub fn stop(&self, grace_s: f64) {
        let mut g = self.inner.lock().unwrap();
        let Some(mut p) = g.proc.take() else { return };
        if matches!(p.try_wait(), Ok(None)) {
            (self.log)(&format!("[{}] stop (pid {})", self.engine, p.id()));
            killpg(p.id(), SIGTERM);
            let t0 = now_s();
            loop {
                if !matches!(p.try_wait(), Ok(None)) {
                    break;
                }
                if now_s() - t0 > grace_s {
                    killpg(p.id(), SIGKILL);
                    let _ = p.kill();
                    let _ = p.wait();
                    break;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

pub type GpuQuery = Arc<dyn Fn() -> Option<GpuInfo> + Send + Sync>;

pub struct SidecarManager {
    pub log: Log,
    /// engine → sidecar, in start order
    sidecars: Mutex<Vec<(String, Sidecar)>>,
    /// sidecar key → the model variant it holds. fv-serve serves a whole
    /// process group but keeps **one** model resident, and its live-stream
    /// and segment sessions refuse a model that is not (its jobs swap on
    /// demand instead), so the stream path has to know what is loaded.
    variants: Mutex<HashMap<String, String>>,
    pub load_timeout_s: f64,
    pub vram_margin_gib: f64,
    pub gpu_query: GpuQuery,
}

impl Default for SidecarManager {
    fn default() -> Self {
        Self::new(crate::null_log(), 1800.0)
    }
}

impl SidecarManager {
    pub fn new(log: Log, load_timeout_s: f64) -> SidecarManager {
        SidecarManager {
            log,
            sidecars: Mutex::new(Vec::new()),
            variants: Mutex::new(HashMap::new()),
            load_timeout_s,
            vram_margin_gib: 1.0,
            gpu_query: Arc::new(|| gpu::query(0)),
        }
    }

    pub fn resident(&self) -> Map<String, Value> {
        let mut out = Map::new();
        for (e, s) in self.sidecars.lock().unwrap().iter() {
            out.insert(
                e.clone(),
                json!({"key": s.key, "alive": s.alive(), "pid": s.pid(),
                       "variant": self.variant_of(&s.key),
                       "ready_s": s.ready_at().map(|r| crate::py::round_to(r - s.started_at(), 1))}),
            );
        }
        out
    }

    /// Records the model a sidecar now holds (see `variants`).
    pub fn note_variant(&self, key: &str, variant: &str) {
        self.variants
            .lock()
            .unwrap()
            .insert(key.to_string(), variant.to_string());
    }

    pub fn variant_of(&self, key: &str) -> Option<String> {
        self.variants.lock().unwrap().get(key).cloned()
    }

    pub fn get(&self, engine: &str) -> Option<Sidecar> {
        self.sidecars
            .lock()
            .unwrap()
            .iter()
            .find(|(e, _)| e == engine)
            .map(|(_, s)| s.clone())
    }

    pub fn engines(&self) -> Vec<String> {
        self.sidecars
            .lock()
            .unwrap()
            .iter()
            .map(|(e, _)| e.clone())
            .collect()
    }

    pub fn contains(&self, engine: &str) -> bool {
        self.get(engine).is_some()
    }

    /// `sidecars.pop(engine)` without stopping it.
    pub fn remove(&self, engine: &str) -> Option<Sidecar> {
        let mut g = self.sidecars.lock().unwrap();
        let i = g.iter().position(|(e, _)| e == engine)?;
        Some(g.remove(i).1)
    }

    /// The engine's sidecar, started for `key` (restarted when it was
    /// running for another key or died). Returns it and the load seconds (0
    /// when it was already up).
    pub fn ensure(
        &self,
        engine: &str,
        key: &str,
        make: &dyn Fn() -> Sidecar,
        vram_gib: f64,
        progress: Option<&dyn Fn(&str, f64)>,
    ) -> JobResult<(Sidecar, f64)> {
        let cur = self.get(engine);
        if let Some(cur) = &cur {
            if cur.alive() && cur.key == key {
                let (ready, err) = (cur.probe)();
                if ready && err.is_none() {
                    return Ok((cur.clone(), 0.0));
                }
                if let Some(err) = err {
                    (self.log)(&format!(
                        "[{engine}] resident sidecar reports an error, restarting: {err}"
                    ));
                }
            }
        }
        if let Some(cur) = cur {
            if !cur.alive() && cur.pid().is_some() {
                (self.log)(&format!(
                    "[{engine}] sidecar had exited (code {}); last lines: {}",
                    cur.exit_code()
                        .map(|c| c.to_string())
                        .unwrap_or_else(|| "None".into()),
                    cur.tail(5).join(" | ")
                ));
            }
            cur.stop(15.0);
            self.remove(engine);
        }
        self.make_room(engine, vram_gib);
        let mut sc = make();
        sc.log = self.log.clone();
        if let Some(p) = progress {
            p("loading", 0.0);
        }
        sc.start()?;
        self.sidecars
            .lock()
            .unwrap()
            .push((engine.to_string(), sc.clone()));
        let load_s = sc.wait_ready(
            self.load_timeout_s,
            progress
                .map(|p| Box::new(move |t: f64| p("loading", t)) as Box<dyn Fn(f64)>)
                .as_deref(),
        )?;
        Ok((sc, load_s))
    }

    pub fn make_room(&self, engine: &str, vram_gib: f64) {
        if vram_gib <= 0.0 {
            return;
        }
        let Some(g) = (self.gpu_query)() else { return };
        if g.free_gib() >= vram_gib + self.vram_margin_gib {
            return;
        }
        let others: Vec<String> = self.engines().into_iter().filter(|e| e != engine).collect();
        if others.is_empty() {
            (self.log)(&format!("[{engine}] only {:.1} GiB free for a {vram_gib:.0} GiB model and nothing else to stop", g.free_gib()));
            return;
        }
        (self.log)(&format!(
            "[{engine}] {:.1} GiB free < {vram_gib:.0} GiB: stopping {}",
            g.free_gib(),
            others.join(", ")
        ));
        for e in &others {
            if let Some(s) = self.remove(e) {
                s.stop(15.0);
            }
        }
        for _ in 0..20 {
            match (self.gpu_query)() {
                None => return,
                Some(g) if g.free_gib() >= vram_gib + self.vram_margin_gib => return,
                _ => std::thread::sleep(Duration::from_millis(500)),
            }
        }
        if let Some(g) = (self.gpu_query)() {
            (self.log)(&format!(
                "[{engine}] still only {:.1} GiB free; starting anyway",
                g.free_gib()
            ));
        }
    }

    pub fn stop_all(&self) {
        let all: Vec<(String, Sidecar)> = std::mem::take(&mut *self.sidecars.lock().unwrap());
        for (_, s) in all {
            s.stop(15.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sh(script: &str) -> Vec<String> {
        vec!["/bin/sh".into(), "-c".into(), script.into()]
    }

    #[test]
    fn a_sidecar_keeps_its_log_tail_and_reports_exits() {
        let sc = Sidecar::new(
            "x",
            "k",
            sh("echo one; echo two 1>&2; sleep 0.2; exit 3"),
            HashMap::new(),
            Arc::new(|| (false, None)),
        );
        sc.start().unwrap();
        let e = sc.wait_ready(5.0, None).unwrap_err();
        assert_eq!(e.where_, "engine");
        assert!(
            e.message.contains("exited with code 3 while loading"),
            "{}",
            e.message
        );
        let mut tail = e.log_tail.clone();
        tail.sort();
        assert_eq!(tail, ["one", "two"]);
        assert!(!sc.alive());
    }

    #[test]
    fn ready_probe_and_stop() {
        let flag = Arc::new(Mutex::new(0));
        let f = flag.clone();
        let probe: Probe = Arc::new(move || {
            let mut n = f.lock().unwrap();
            *n += 1;
            (*n >= 2, None)
        });
        let sc = Sidecar::new(
            "x",
            "k",
            sh("trap 'exit 0' TERM; while true; do sleep 0.1; done"),
            HashMap::new(),
            probe,
        );
        sc.start().unwrap();
        let t = sc.wait_ready(5.0, None).unwrap();
        assert!(t >= 0.4 && sc.alive() && sc.ready_at().is_some());
        sc.stop(2.0);
        assert!(!sc.alive() && sc.pid().is_none());
    }

    #[test]
    fn a_fatal_probe_stops_the_process() {
        let sc = Sidecar::new(
            "fv",
            "k",
            sh("sleep 10"),
            HashMap::new(),
            Arc::new(|| (false, Some("boom".into()))),
        );
        sc.start().unwrap();
        let e = sc.wait_ready(5.0, None).unwrap_err();
        assert!(e.message.contains("fv failed to start: boom"));
        assert!(!sc.alive());
    }

    #[test]
    fn manager_reuses_restarts_and_makes_room() {
        let logs = Arc::new(Mutex::new(Vec::new()));
        let l = logs.clone();
        let m = SidecarManager::new(
            Arc::new(move |s: &str| l.lock().unwrap().push(s.to_string())),
            5.0,
        );
        let make = |engine: &'static str, key: &'static str| {
            move || {
                Sidecar::new(
                    engine,
                    key,
                    sh("trap 'exit 0' TERM; while true; do sleep 0.1; done"),
                    HashMap::new(),
                    Arc::new(|| (true, None)),
                )
            }
        };
        let (a, t) = m.ensure("h3", "a", &make("h3", "a"), 0.0, None).unwrap();
        assert!(t >= 0.0 && a.alive());
        let pid = a.pid();
        let (a2, t2) = m.ensure("h3", "a", &make("h3", "a"), 0.0, None).unwrap();
        assert_eq!((a2.pid(), t2), (pid, 0.0)); // reused
        let (b, _) = m.ensure("h3", "b", &make("h3", "b"), 0.0, None).unwrap();
        assert_ne!(b.pid(), pid); // another key restarts
        assert!(!a.alive());
        a.kill_now();
        b.kill_now();
        let (c, _) = m.ensure("h3", "b", &make("h3", "b"), 0.0, None).unwrap(); // died: restarted
        assert_ne!(c.pid(), b.pid());
        assert!(logs
            .lock()
            .unwrap()
            .iter()
            .any(|s| s.contains("sidecar had exited")));
        // make_room: a tiny GPU evicts the other engine.
        let m2 = SidecarManager {
            gpu_query: Arc::new(|| {
                Some(GpuInfo {
                    name: "g".into(),
                    driver: "1".into(),
                    memory_total_mib: 8192,
                    memory_free_mib: 1024,
                    compute_cap: None,
                })
            }),
            ..SidecarManager::new(m.log.clone(), 5.0)
        };
        let (h, _) = m2.ensure("h3", "a", &make("h3", "a"), 0.0, None).unwrap();
        let (_f, _) = m2
            .ensure("fastwan", "a", &make("fastwan", "a"), 4.0, None)
            .unwrap();
        assert!(!h.alive() && m2.engines() == ["fastwan"]);
        assert!(logs
            .lock()
            .unwrap()
            .iter()
            .any(|s| s.contains("1.0 GiB free < 4 GiB: stopping h3")));
        let r = m2.resident();
        assert_eq!(r["fastwan"]["alive"], json!(true));
        assert!(r["fastwan"]["ready_s"].is_number());
        m.stop_all();
        m2.stop_all();
        assert!(m2.engines().is_empty());
    }
}
