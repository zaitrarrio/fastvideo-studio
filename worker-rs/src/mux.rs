//! ffmpeg: y4m (+ wav) → H.264 + AAC yuv420p MP4 with `+faststart`; a JPEG
//! thumbnail from the middle frame; a probe of an MP4's streams.

use crate::py::which;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

pub fn ffmpeg_path() -> Option<PathBuf> {
    which("ffmpeg")
}

pub fn ffprobe_path() -> Option<PathBuf> {
    which("ffprobe")
}

fn ff(ffmpeg: Option<&str>) -> Result<PathBuf, String> {
    match ffmpeg {
        Some(f) => Ok(PathBuf::from(f)),
        None => ffmpeg_path().ok_or_else(|| "ffmpeg is not installed".to_string()),
    }
}

fn stderr_tail(out: &std::process::Output, n: usize) -> String {
    let s = String::from_utf8_lossy(&out.stderr);
    let t = s.trim();
    let chars: Vec<char> = t.chars().collect();
    chars[chars.len().saturating_sub(n)..].iter().collect()
}

#[derive(Debug, Clone, PartialEq)]
pub struct MuxResult {
    pub mp4: PathBuf,
    pub seconds: f64,
}

pub fn mux(
    video_y4m: &Path,
    audio_wav: Option<&Path>,
    out: &Path,
    crf: u32,
    preset: &str,
    ffmpeg: Option<&str>,
) -> Result<MuxResult, String> {
    let bin = ff(ffmpeg)?;
    let t0 = Instant::now();
    let audio = audio_wav.filter(|a| a.is_file());
    let mut cmd = Command::new(bin);
    cmd.args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(video_y4m);
    if let Some(a) = audio {
        cmd.arg("-i").arg(a);
    }
    cmd.args([
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        &crf.to_string(),
        "-pix_fmt",
        "yuv420p",
    ]);
    if audio.is_some() {
        cmd.args(["-c:a", "aac", "-b:a", "192k", "-shortest"]);
    }
    cmd.args(["-movflags", "+faststart"]).arg(out);
    let r = cmd
        .output()
        .map_err(|e| format!("ffmpeg mux failed: {e}"))?;
    if !r.status.success() {
        return Err(format!("ffmpeg mux failed: {}", stderr_tail(&r, 1200)));
    }
    Ok(MuxResult {
        mp4: out.to_path_buf(),
        seconds: t0.elapsed().as_secs_f64(),
    })
}

pub fn thumbnail(
    mp4: &Path,
    out: &Path,
    frames: i64,
    _fps: i64,
    ffmpeg: Option<&str>,
    quality: u32,
) -> Result<PathBuf, String> {
    let bin = ff(ffmpeg)?;
    let mid = (frames / 2).max(0);
    let r = Command::new(&bin)
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(mp4)
        .args([
            "-vf",
            &format!("select=eq(n\\,{mid})"),
            "-frames:v",
            "1",
            "-q:v",
            &quality.to_string(),
        ])
        .arg(out)
        .output()
        .map_err(|e| format!("ffmpeg thumbnail failed: {e}"))?;
    if !r.status.success() || !out.is_file() {
        // The middle frame may be past a short/odd stream: fall back to the first.
        let r = Command::new(&bin)
            .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
            .arg(mp4)
            .args(["-frames:v", "1", "-q:v", &quality.to_string()])
            .arg(out)
            .output()
            .map_err(|e| format!("ffmpeg thumbnail failed: {e}"))?;
        if !r.status.success() {
            return Err(format!("ffmpeg thumbnail failed: {}", stderr_tail(&r, 800)));
        }
    }
    Ok(out.to_path_buf())
}

/// The audio track as 16-bit WAV, or None when there is none.
pub fn extract_wav(mp4: &Path, out: &Path, ffmpeg: Option<&str>) -> Option<PathBuf> {
    let bin = ff(ffmpeg).ok()?;
    let r = Command::new(bin)
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(mp4)
        .args(["-vn", "-acodec", "pcm_s16le"])
        .arg(out)
        .output()
        .ok()?;
    if !r.status.success() || !out.is_file() || out.metadata().map(|m| m.len() < 64).unwrap_or(true)
    {
        return None;
    }
    Some(out.to_path_buf())
}

#[derive(Debug, Clone, PartialEq)]
pub struct Probe {
    pub width: i64,
    pub height: i64,
    pub frames: i64,
    pub fps: f64,
    pub audio: bool,
}

pub fn probe(mp4: &Path, ffprobe: Option<&str>) -> Option<Probe> {
    let fp = match ffprobe {
        Some(f) => PathBuf::from(f),
        None => ffprobe_path()?,
    };
    let r = Command::new(fp)
        .args([
            "-v",
            "error",
            "-show_streams",
            "-count_frames",
            "-of",
            "json",
        ])
        .arg(mp4)
        .output()
        .ok()?;
    if !r.status.success() {
        return None;
    }
    let doc: Value = serde_json::from_slice(&r.stdout).unwrap_or(Value::Object(Default::default()));
    let streams = doc["streams"].as_array().cloned().unwrap_or_default();
    let v = streams.iter().find(|s| s["codec_type"] == "video")?;
    let rate = v["r_frame_rate"].as_str().unwrap_or("0/1");
    let (n, d) = rate.split_once('/').unwrap_or((rate, "1"));
    let (n, d): (f64, f64) = (n.parse().unwrap_or(0.0), d.parse().unwrap_or(0.0));
    let fps = if d != 0.0 { n / d } else { 0.0 };
    let frames = ["nb_read_frames", "nb_frames"]
        .iter()
        .find_map(|k| {
            v[*k]
                .as_str()
                .and_then(|s| s.parse::<i64>().ok())
                .filter(|n| *n != 0)
        })
        .unwrap_or(0);
    Some(Probe {
        width: v["width"].as_i64()?,
        height: v["height"].as_i64()?,
        frames,
        fps,
        audio: streams.iter().any(|s| s["codec_type"] == "audio"),
    })
}

/// Any video → y4m (yuv420p), optionally scaled to a canvas (cover-crop)
/// and trimmed to its last N frames (h3's `--continue` takes 5 / 22 / 39
/// frames at the canvas).
pub fn to_y4m(
    src: &Path,
    out: &Path,
    size: Option<(i64, i64)>,
    fps: Option<i64>,
    last_frames: Option<i64>,
    ffmpeg: Option<&str>,
) -> Result<PathBuf, String> {
    let bin = ff(ffmpeg)?;
    let mut vf = Vec::new();
    if let Some((w, h)) = size.filter(|(w, h)| *w != 0 && *h != 0) {
        vf.push(format!(
            "scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
        ));
    }
    if let Some(f) = fps.filter(|f| *f != 0) {
        vf.push(format!("fps={f}"));
    }
    let mut cmd = Command::new(bin);
    cmd.args(["-hide_banner", "-loglevel", "error", "-y"]);
    if let Some(n) = last_frames.filter(|n| *n != 0) {
        if let Some(p) = probe(src, None) {
            if p.frames > n && p.fps != 0.0 {
                let start = ((p.frames - n) as f64 / p.fps - 0.001).max(0.0);
                cmd.args(["-ss", &format!("{start:.4}")]);
            }
        }
    }
    cmd.arg("-i").arg(src);
    if !vf.is_empty() {
        cmd.args(["-vf", &vf.join(",")]);
    }
    if let Some(n) = last_frames.filter(|n| *n != 0) {
        cmd.args(["-frames:v", &n.to_string()]);
    }
    cmd.args(["-pix_fmt", "yuv420p", "-f", "yuv4mpegpipe"])
        .arg(out);
    let r = cmd
        .output()
        .map_err(|e| format!("ffmpeg →y4m failed: {e}"))?;
    if !r.status.success() {
        return Err(format!("ffmpeg →y4m failed: {}", stderr_tail(&r, 800)));
    }
    Ok(out.to_path_buf())
}

pub fn to_wav(src: &Path, out: &Path, ffmpeg: Option<&str>) -> Result<PathBuf, String> {
    let bin = ff(ffmpeg)?;
    let r = Command::new(bin)
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(src)
        .args(["-vn", "-acodec", "pcm_s16le"])
        .arg(out)
        .output()
        .map_err(|e| format!("ffmpeg →wav failed: {e}"))?;
    if !r.status.success() {
        return Err(format!("ffmpeg →wav failed: {}", stderr_tail(&r, 800)));
    }
    Ok(out.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::{read_y4m_info, rgb_to_y4m};
    use crate::tempdir;

    #[test]
    fn mux_probe_thumbnail_roundtrip() {
        if ffmpeg_path().is_none() || ffprobe_path().is_none() {
            eprintln!("ffmpeg / ffprobe not installed: skipped");
            return;
        }
        let dir = tempdir("mux");
        let (w, h, n) = (64u32, 48u32, 7usize);
        let rgb = dir.join("f.rgb");
        std::fs::write(&rgb, [10u8, 200, 30].repeat((w * h) as usize).repeat(n)).unwrap();
        let y4m = dir.join("f.y4m");
        rgb_to_y4m(&rgb, w, h, 24, &y4m, None).unwrap();
        let mp4 = dir.join("v.mp4");
        let r = mux(&y4m, None, &mp4, 18, "ultrafast", None).unwrap();
        assert!(r.seconds >= 0.0 && mp4.metadata().unwrap().len() > 500);
        let p = probe(&mp4, None).unwrap();
        assert_eq!((p.width, p.height, p.frames, p.audio), (64, 48, 7, false));
        assert!((p.fps - 24.0).abs() < 1e-6);
        let t = thumbnail(&mp4, &dir.join("t.jpg"), 7, 24, None, 3).unwrap();
        assert!(t.metadata().unwrap().len() > 100);
        assert!(extract_wav(&mp4, &dir.join("a.wav"), None).is_none());
        let back = to_y4m(
            &mp4,
            &dir.join("back.y4m"),
            Some((32, 32)),
            Some(12),
            Some(3),
            None,
        )
        .unwrap();
        let info = read_y4m_info(&back).unwrap();
        assert_eq!((info.width, info.height), (32, 32));
        assert!(info.frames <= 3);
    }
}
