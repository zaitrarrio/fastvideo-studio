//! The jobs.md progress record and timings, from whatever an engine reports.

use crate::now_s;
use crate::py::round_to;
use serde_json::{json, Map, Value};
use std::sync::Mutex;

pub const STAGES: [&str; 12] = ["queued", "staging", "loading", "encoding", "preparing", "denoising", "decoding", "upscaling", "muxing", "uploading", "done", "failed"];

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Timings {
    pub stage_s: f64,
    pub load_s: f64,
    pub encode_s: f64,
    pub denoise_s: f64,
    pub decode_s: f64,
    pub upscale_s: f64,
    pub mux_s: f64,
    pub total_s: f64,
    pub steps_s: Vec<f64>,
}

impl Timings {
    pub fn to_json(&self) -> Value {
        let r = |v: f64| round_to(v, 3);
        json!({
            "stage_s": r(self.stage_s), "load_s": r(self.load_s), "encode_s": r(self.encode_s), "denoise_s": r(self.denoise_s),
            "decode_s": r(self.decode_s), "upscale_s": r(self.upscale_s), "mux_s": r(self.mux_s), "total_s": r(self.total_s),
            "steps_s": self.steps_s.iter().map(|s| r(*s)).collect::<Vec<_>>(),
        })
    }
}

/// The optional fields of one progress record.
#[derive(Debug, Clone, Default)]
pub struct Detail<'a> {
    pub detail: Option<&'a str>,
    pub step: Option<i64>,
    pub total_steps: Option<i64>,
    pub frames: Option<i64>,
    pub total_frames: Option<i64>,
    pub force: bool,
}

/// The progress sink: RunPod's `progress_update` (or a recorder in tests).
pub type ProgressFn<'a> = &'a (dyn Fn(&Value) + Sync);

/// Sends progress records (rate-limited) and remembers the last one.
pub struct Reporter<'a> {
    send: Option<ProgressFn<'a>>,
    pub engine: String,
    pub model: String,
    pub t0: f64,
    state: Mutex<State>,
    pub min_interval_s: f64,
}

#[derive(Default)]
struct State {
    last: Map<String, Value>,
    last_sent: f64,
}

impl<'a> Reporter<'a> {
    pub fn new(send: Option<ProgressFn<'a>>, engine: &str, model: &str) -> Self {
        Self::with_interval(send, engine, model, 0.5)
    }

    pub fn with_interval(send: Option<ProgressFn<'a>>, engine: &str, model: &str, min_interval_s: f64) -> Self {
        Reporter { send, engine: engine.into(), model: model.into(), t0: now_s(), state: Mutex::new(State::default()), min_interval_s }
    }

    /// The last record sent or recorded.
    pub fn last(&self) -> Map<String, Value> {
        self.state.lock().unwrap().last.clone()
    }

    pub fn stage(&self, stage: &str) {
        self.report(stage, Detail::default());
    }

    pub fn force(&self, stage: &str) {
        self.report(stage, Detail { force: true, ..Default::default() });
    }

    pub fn detail(&self, stage: &str, detail: &str) {
        self.report(stage, Detail { detail: Some(detail), ..Default::default() });
    }

    pub fn detail_force(&self, stage: &str, detail: &str) {
        self.report(stage, Detail { detail: Some(detail), force: true, ..Default::default() });
    }

    pub fn report(&self, stage: &str, d: Detail) {
        assert!(STAGES.contains(&stage), "{stage}");
        let mut rec = Map::new();
        rec.insert("stage".into(), json!(stage));
        rec.insert("elapsed_s".into(), json!(round_to(now_s() - self.t0, 1)));
        rec.insert("engine".into(), json!(self.engine));
        rec.insert("model".into(), json!(self.model));
        if let Some(detail) = d.detail.filter(|s| !s.is_empty()) {
            rec.insert("detail".into(), json!(detail));
        }
        for (k, v) in [("step", d.step), ("total_steps", d.total_steps), ("frames", d.frames), ("total_frames", d.total_frames)] {
            if let Some(v) = v {
                rec.insert(k.into(), json!(v));
            }
        }
        let mut st = self.state.lock().unwrap();
        let strip = |m: &Map<String, Value>| -> Map<String, Value> { m.iter().filter(|(k, _)| *k != "elapsed_s").map(|(k, v)| (k.clone(), v.clone())).collect() };
        let changed = strip(&rec) != strip(&st.last);
        let stage_changed = st.last.get("stage").and_then(|v| v.as_str()) != Some(stage);
        st.last = rec.clone();
        let now = now_s();
        // A new stage always goes out; detail updates within a stage are rate-limited.
        if let Some(send) = self.send {
            if d.force || stage_changed || (changed && now - st.last_sent >= self.min_interval_s) || now - st.last_sent >= 5.0 {
                st.last_sent = now;
                drop(st);
                // A lost progress update is not a failed job.
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| send(&Value::Object(rec))));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_and_rate_limits() {
        let got = Mutex::new(Vec::new());
        let send = |v: &Value| got.lock().unwrap().push(v.clone());
        let r = Reporter::with_interval(Some(&send), "h3", "m", 100.0);
        r.force("queued");
        r.detail("staging", "weights: a");
        r.detail("staging", "weights: b"); // rate-limited: same stage, within the interval
        r.report("denoising", Detail { detail: Some("step 1/8"), step: Some(1), total_steps: Some(8), ..Default::default() });
        let g = got.lock().unwrap();
        assert_eq!(g.len(), 3);
        assert_eq!(g[0]["stage"], "queued");
        assert_eq!(g[1]["detail"], "weights: a");
        assert_eq!(g[2]["step"], 1);
        assert_eq!(g[2]["total_steps"], 8);
        assert!(g.iter().all(|d| d.get("engine").is_some() && d.get("elapsed_s").is_some() && d.get("model").is_some()));
        assert_eq!(r.last()["detail"], "step 1/8");
        let t = Timings { steps_s: vec![1.23456, 2.0], total_s: 3.0, ..Default::default() }.to_json();
        assert_eq!(t["steps_s"], json!([1.235, 2.0]));
        assert_eq!(t["total_s"], json!(3.0));
        assert_eq!(t.as_object().unwrap().keys().collect::<Vec<_>>(), ["stage_s", "load_s", "encode_s", "denoise_s", "decode_s", "upscale_s", "mux_s", "total_s", "steps_s"]);
    }
}
