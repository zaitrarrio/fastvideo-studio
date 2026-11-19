//! The engines' frame stream (`GET /api/jobs/<id>/frames?from=N`: a 16-byte
//! little-endian u32 header `width, height, first, count` then RGB8 frames) and
//! y4m files (parsed natively; written from RGB through ffmpeg's rawvideo
//! input, the fastest colour conversion available).

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub const HEADER_SIZE: usize = 16;

#[derive(Debug, Clone, PartialEq)]
pub struct FrameChunk {
    pub width: u32,
    pub height: u32,
    pub first: u32,
    pub count: u32,
    pub rgb: Vec<u8>,
}

impl FrameChunk {
    pub fn frame_bytes(&self) -> usize {
        self.width as usize * self.height as usize * 3
    }
}

pub fn parse_frame_stream(body: &[u8]) -> Result<FrameChunk, String> {
    if body.len() < HEADER_SIZE {
        return Err(format!("frame stream too short ({} bytes)", body.len()));
    }
    let u = |i: usize| u32::from_le_bytes(body[i * 4..i * 4 + 4].try_into().unwrap());
    let (width, height, first, count) = (u(0), u(1), u(2), u(3));
    let px = width as usize * height as usize * 3;
    let want = HEADER_SIZE + px * count as usize;
    if body.len() != want {
        return Err(format!("frame stream: {} bytes for {count} frames of {width}x{height} (want {want})", body.len()));
    }
    Ok(FrameChunk { width, height, first, count, rgb: body[HEADER_SIZE..].to_vec() })
}

/// The inverse, for tests and fakes.
pub fn encode_frame_stream(width: u32, height: u32, first: u32, frames: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    for v in [width, height, first, frames.len() as u32] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for f in frames {
        out.extend_from_slice(f);
    }
    out
}

/// Frames accumulated from successive chunks, written to a raw file.
pub struct RgbClip {
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    fh: Option<File>,
}

impl RgbClip {
    pub fn new(path: PathBuf) -> RgbClip {
        RgbClip { path, width: 0, height: 0, frames: 0, fh: None }
    }

    pub fn append(&mut self, chunk: &FrameChunk) -> Result<(), String> {
        if chunk.count == 0 {
            return Ok(());
        }
        if self.frames == 0 {
            self.width = chunk.width;
            self.height = chunk.height;
            self.fh = Some(File::create(&self.path).map_err(|e| e.to_string())?);
        } else if (chunk.width, chunk.height) != (self.width, self.height) {
            return Err("frame size changed mid-stream".into());
        }
        if chunk.first != self.frames {
            return Err(format!("frame stream gap: have {}, chunk starts at {}", self.frames, chunk.first));
        }
        self.fh.as_mut().unwrap().write_all(&chunk.rgb).map_err(|e| e.to_string())?;
        self.frames += chunk.count;
        Ok(())
    }

    pub fn close(&mut self) {
        self.fh = None;
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Y4mInfo {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub frames: u64,
    pub colorspace: String,
}

/// The stream header and a frame count from the file size.
pub fn read_y4m_info(path: &Path) -> Result<Y4mInfo, String> {
    let f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let size = f.metadata().map_err(|e| e.to_string())?.len();
    let mut header = Vec::new();
    BufReader::new(f).read_until(b'\n', &mut header).map_err(|e| e.to_string())?;
    if !header.starts_with(b"YUV4MPEG2 ") {
        return Err(format!("{}: not a y4m file", path.display()));
    }
    let (mut w, mut h, mut fps, mut cs) = (0u32, 0u32, 0.0f64, "420jpeg".to_string());
    for tok in String::from_utf8_lossy(&header[10..]).split_whitespace() {
        let (c, rest) = tok.split_at(1);
        match c {
            "W" => w = rest.parse().unwrap_or(0),
            "H" => h = rest.parse().unwrap_or(0),
            "F" => {
                if let Some((n, d)) = rest.split_once(':') {
                    let (n, d): (f64, f64) = (n.parse().unwrap_or(0.0), d.parse().unwrap_or(1.0));
                    fps = if d != 0.0 { n / d } else { 0.0 };
                }
            }
            "C" => cs = rest.to_string(),
            _ => {}
        }
    }
    let (w64, h64) = (w as u64, h as u64);
    let per = if cs.starts_with("444") {
        w64 * h64 * 3
    } else if cs.starts_with("422") {
        w64 * h64 * 2
    } else if cs == "mono" {
        w64 * h64
    } else {
        w64 * h64 * 3 / 2
    };
    let body = size.saturating_sub(header.len() as u64);
    let frame_block = 6 + per;
    Ok(Y4mInfo { width: w, height: h, fps, frames: if frame_block > 0 { body / frame_block } else { 0 }, colorspace: cs })
}

fn ffmpeg_bin(ffmpeg: Option<&str>) -> String {
    ffmpeg.map(str::to_string).unwrap_or_else(|| "ffmpeg".into())
}

fn tail_of(s: &str, n: usize) -> String {
    let t = s.trim();
    let chars: Vec<char> = t.chars().collect();
    chars[chars.len().saturating_sub(n)..].iter().collect()
}

/// Raw RGB8 frames → y4m (yuv420p) through ffmpeg.
pub fn rgb_to_y4m(rgb_path: &Path, width: u32, height: u32, fps: i64, out: &Path, ffmpeg: Option<&str>) -> Result<(), String> {
    let r = Command::new(ffmpeg_bin(ffmpeg))
        .args(["-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", &format!("{width}x{height}"), "-r", &fps.to_string(), "-i"])
        .arg(rgb_path)
        .args(["-pix_fmt", "yuv420p", "-f", "yuv4mpegpipe"])
        .arg(out)
        .output()
        .map_err(|e| format!("ffmpeg rgb→y4m failed: {e}"))?;
    if !r.status.success() {
        return Err(format!("ffmpeg rgb→y4m failed: {}", tail_of(&String::from_utf8_lossy(&r.stderr), 800)));
    }
    Ok(())
}

/// A JPEG sequence (upscaled frames) → y4m through ffmpeg's image2pipe.
pub fn jpegs_to_y4m(jpegs: &[Vec<u8>], fps: i64, out: &Path, ffmpeg: Option<&str>) -> Result<(), String> {
    let mut child = Command::new(ffmpeg_bin(ffmpeg))
        .args(["-hide_banner", "-loglevel", "error", "-y", "-f", "image2pipe", "-vcodec", "mjpeg", "-r", &fps.to_string(), "-i", "pipe:0", "-pix_fmt", "yuv420p", "-f", "yuv4mpegpipe"])
        .arg(out)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("ffmpeg jpeg→y4m failed: {e}"))?;
    {
        let mut stdin = child.stdin.take().unwrap();
        for j in jpegs {
            if let Err(e) = stdin.write_all(j) {
                if e.kind() != std::io::ErrorKind::BrokenPipe {
                    return Err(format!("ffmpeg jpeg→y4m failed: {e}"));
                }
                break;
            }
        }
    }
    let r = child.wait_with_output().map_err(|e| e.to_string())?;
    if !r.status.success() {
        return Err(format!("ffmpeg jpeg→y4m failed: {}", tail_of(&String::from_utf8_lossy(&r.stderr), 800)));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tempdir;

    #[test]
    fn frame_stream_roundtrip() {
        let f0: Vec<u8> = [1u8, 2, 3].repeat(4);
        let f1: Vec<u8> = [9u8, 8, 7].repeat(4);
        let body = encode_frame_stream(2, 2, 5, &[f0.clone(), f1.clone()]);
        assert_eq!(&body[..16], [2u8, 0, 0, 0, 2, 0, 0, 0, 5, 0, 0, 0, 2, 0, 0, 0]);
        let c = parse_frame_stream(&body).unwrap();
        assert_eq!((c.width, c.height, c.first, c.count), (2, 2, 5, 2));
        assert_eq!(c.rgb, [f0, f1].concat());
    }

    #[test]
    fn frame_stream_rejects_truncation() {
        let body = encode_frame_stream(2, 2, 0, &[vec![0; 12]]);
        assert!(parse_frame_stream(&body[..body.len() - 1]).is_err());
        assert!(parse_frame_stream(&[0u8; 8]).is_err());
    }

    #[test]
    fn rgb_clip_accumulates_in_order() {
        let dir = tempdir("frames");
        let mut clip = RgbClip::new(dir.join("f.rgb"));
        clip.append(&parse_frame_stream(&encode_frame_stream(2, 2, 0, &[vec![0; 12], vec![0; 12]])).unwrap()).unwrap();
        clip.append(&parse_frame_stream(&encode_frame_stream(2, 2, 2, &[vec![0; 12]])).unwrap()).unwrap();
        let e = clip.append(&parse_frame_stream(&encode_frame_stream(2, 2, 5, &[vec![0; 12]])).unwrap()).unwrap_err();
        assert!(e.contains("gap"));
        clip.close();
        assert_eq!(clip.frames, 3);
        assert_eq!(std::fs::metadata(dir.join("f.rgb")).unwrap().len(), 36);
    }

    #[test]
    fn y4m_info() {
        let dir = tempdir("y4m");
        let p = dir.join("a.y4m");
        let mut frame = b"FRAME\n".to_vec();
        frame.extend(vec![0u8; 4 * 2 * 3 / 2]);
        let mut data = b"YUV4MPEG2 W4 H2 F24:1 Ip A1:1 C420jpeg\n".to_vec();
        for _ in 0..3 {
            data.extend(&frame);
        }
        std::fs::write(&p, data).unwrap();
        let info = read_y4m_info(&p).unwrap();
        assert_eq!(info, Y4mInfo { width: 4, height: 2, fps: 24.0, frames: 3, colorspace: "420jpeg".into() });
    }

    #[test]
    fn rgb_to_y4m_through_ffmpeg() {
        if crate::py::which("ffmpeg").is_none() {
            eprintln!("ffmpeg not installed: skipped");
            return;
        }
        let dir = tempdir("rgb2y4m");
        let (w, h, n) = (32u32, 16u32, 5usize);
        let rgb = dir.join("f.rgb");
        std::fs::write(&rgb, [200u8, 30, 30].repeat((w * h) as usize).repeat(n)).unwrap();
        let out = dir.join("f.y4m");
        rgb_to_y4m(&rgb, w, h, 24, &out, None).unwrap();
        let info = read_y4m_info(&out).unwrap();
        assert_eq!((info.width, info.height, info.frames, info.fps), (w, h, n as u64, 24.0));
    }
}
