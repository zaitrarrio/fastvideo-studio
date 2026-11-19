//! nvidia-smi queries. Everything degrades to `None` without a GPU.

use crate::py::which;
use std::process::Command;

#[derive(Debug, Clone, PartialEq)]
pub struct GpuInfo {
    pub name: String,
    pub driver: String,
    pub memory_total_mib: i64,
    pub memory_free_mib: i64,
    pub compute_cap: Option<String>,
}

impl GpuInfo {
    pub fn free_gib(&self) -> f64 {
        self.memory_free_mib as f64 / 1024.0
    }

    pub fn sm(&self) -> Option<i64> {
        let cap = self.compute_cap.as_ref()?;
        let (major, minor) = cap.split_once('.')?;
        Some(major.parse::<i64>().ok()? * 10 + minor.parse::<i64>().ok()?)
    }
}

pub fn query(index: u32) -> Option<GpuInfo> {
    let smi = which("nvidia-smi")?;
    let r = Command::new(&smi)
        .args([
            format!("--id={index}"),
            "--query-gpu=name,driver_version,memory.total,memory.free".into(),
            "--format=csv,noheader,nounits".into(),
        ])
        .output()
        .ok()?;
    let out = String::from_utf8_lossy(&r.stdout);
    if !r.status.success() || out.trim().is_empty() {
        return None;
    }
    let parts: Vec<&str> = out
        .trim()
        .lines()
        .next()?
        .split(',')
        .map(str::trim)
        .collect();
    if parts.len() < 4 {
        return None;
    }
    let mut cap = None;
    if let Ok(r2) = Command::new(&smi)
        .args([
            format!("--id={index}"),
            "--query-gpu=compute_cap".into(),
            "--format=csv,noheader".into(),
        ])
        .output()
    {
        let o = String::from_utf8_lossy(&r2.stdout);
        if r2.status.success() && !o.trim().is_empty() {
            cap = o.trim().lines().next().map(|s| s.trim().to_string());
        }
    }
    Some(GpuInfo {
        name: parts[0].into(),
        driver: parts[1].into(),
        memory_total_mib: parts[2].parse::<f64>().ok()? as i64,
        memory_free_mib: parts[3].parse::<f64>().ok()? as i64,
        compute_cap: cap,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sm_and_free() {
        let g = GpuInfo {
            name: "x".into(),
            driver: "580".into(),
            memory_total_mib: 98304,
            memory_free_mib: 2048,
            compute_cap: Some("12.0".into()),
        };
        assert_eq!(g.sm(), Some(120));
        assert_eq!(g.free_gib(), 2.0);
        assert_eq!(
            GpuInfo {
                compute_cap: None,
                ..g.clone()
            }
            .sm(),
            None
        );
    }
}
