//! The one error type a job can fail with: a message and `where`.

use serde_json::{json, Value};
use std::fmt;

/// "validate" | "stage" | "engine" | "mux" | "upload"
pub type Where = &'static str;

/// A refused or failed job: `{"error": message, "where": where}` (+ `log_tail`).
#[derive(Debug, Clone)]
pub struct JobError {
    pub where_: Where,
    pub message: String,
    pub log_tail: Vec<String>,
    /// Set when the error was not raised as a `JobError` by the code that
    /// failed (the Python handler's catch-all `except Exception`): the worker
    /// then attaches the engine's sidecar log tail and logs "job crashed".
    pub crash: bool,
}

impl JobError {
    pub fn new(where_: Where, message: impl Into<String>) -> Self {
        JobError {
            where_,
            message: message.into(),
            log_tail: Vec::new(),
            crash: false,
        }
    }

    pub fn with_tail(where_: Where, message: impl Into<String>, tail: Vec<String>) -> Self {
        JobError {
            where_,
            message: message.into(),
            log_tail: tail,
            crash: false,
        }
    }

    pub fn validate(message: impl Into<String>) -> Self {
        Self::new("validate", message)
    }

    pub fn stage(message: impl Into<String>) -> Self {
        Self::new("stage", message)
    }

    pub fn engine(message: impl Into<String>, tail: Vec<String>) -> Self {
        Self::with_tail("engine", message, tail)
    }

    pub fn to_output(&self) -> Value {
        let mut out = json!({"error": self.message, "where": self.where_});
        if !self.log_tail.is_empty() {
            out["log_tail"] = json!(self.log_tail);
        }
        out
    }
}

impl fmt::Display for JobError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for JobError {}

/// Any other error inside a job is an engine crash (`{type}: {message}` in
/// Python; here the error's own text).
impl From<anyhow::Error> for JobError {
    fn from(e: anyhow::Error) -> Self {
        if let Some(j) = e.downcast_ref::<JobError>() {
            return j.clone();
        }
        JobError {
            where_: "engine",
            message: format!("{e:#}"),
            log_tail: Vec::new(),
            crash: true,
        }
    }
}

impl From<std::io::Error> for JobError {
    fn from(e: std::io::Error) -> Self {
        JobError {
            where_: "engine",
            message: format!("OSError: {e}"),
            log_tail: Vec::new(),
            crash: true,
        }
    }
}

pub type JobResult<T> = Result<T, JobError>;
