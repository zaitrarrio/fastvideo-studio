//! Job media (`{url}` / `{b64, mime}` / `{r2}`) → local files in the job dir,
//! and the conversions the engines' CLIs need (y4m / wav).

use crate::catalog::{b64_encode, Media};
use crate::errors::{JobError, JobResult};
use crate::r2::ObjectStore;
use std::path::{Path, PathBuf};

pub const MAX_DOWNLOAD_BYTES: u64 = 512 * 1024 * 1024;
pub const IMAGE_MIMES: [&str; 5] = [
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/x-portable-pixmap",
];

const EXT: &[(&str, &str)] = &[
    ("image/jpeg", ".jpg"),
    ("image/jpg", ".jpg"),
    ("image/png", ".png"),
    ("image/webp", ".webp"),
    ("audio/wav", ".wav"),
    ("audio/x-wav", ".wav"),
    ("audio/mpeg", ".mp3"),
    ("audio/mp4", ".m4a"),
    ("video/mp4", ".mp4"),
    ("video/quicktime", ".mov"),
    ("video/webm", ".webm"),
    ("video/x-yuv4mpeg", ".y4m"),
];

/// `mimetypes.guess_type`'s table for the types a job can carry (Python
/// reads the system's `mime.types` too; this list is the fixed part).
const MIME_BY_EXT: &[(&str, &str)] = &[
    ("png", "image/png"),
    ("jpg", "image/jpeg"),
    ("jpeg", "image/jpeg"),
    ("jpe", "image/jpeg"),
    ("webp", "image/webp"),
    ("gif", "image/gif"),
    ("bmp", "image/bmp"),
    ("ppm", "image/x-portable-pixmap"),
    ("tif", "image/tiff"),
    ("tiff", "image/tiff"),
    ("wav", "audio/x-wav"),
    ("mp3", "audio/mpeg"),
    ("m4a", "audio/mp4"),
    ("flac", "audio/flac"),
    ("ogg", "audio/ogg"),
    ("aac", "audio/aac"),
    ("mp4", "video/mp4"),
    ("m4v", "video/mp4"),
    ("mov", "video/quicktime"),
    ("webm", "video/webm"),
    ("mkv", "video/x-matroska"),
    ("avi", "video/x-msvideo"),
    ("y4m", "video/x-yuv4mpeg"),
    ("json", "application/json"),
    ("txt", "text/plain"),
    ("bin", "application/octet-stream"),
    ("safetensors", "application/octet-stream"),
    ("gguf", "application/octet-stream"),
    ("html", "text/html"),
    ("xml", "application/xml"),
];

#[derive(Debug, Clone)]
pub struct LocalMedia {
    pub path: PathBuf,
    pub mime: String,
    pub source: Media,
}

impl LocalMedia {
    pub fn kind(&self) -> &str {
        self.mime.split('/').next().unwrap_or("")
    }
}

fn suffix(name: &str) -> String {
    let base = name.rsplit('/').next().unwrap_or(name);
    match base.rfind('.') {
        Some(i) if i > 0 => base[i..].to_string(),
        _ => String::new(),
    }
}

/// The media as a file `dest_dir/<name><ext>`.
pub fn fetch_media(
    m: &Media,
    dest_dir: &Path,
    name: &str,
    store: Option<&dyn ObjectStore>,
    bucket: Option<&str>,
) -> JobResult<LocalMedia> {
    std::fs::create_dir_all(dest_dir)?;
    if m.kind == "b64" {
        let mime = m
            .mime
            .clone()
            .unwrap_or_else(|| "application/octet-stream".into());
        let path = dest_dir.join(format!("{name}{}", ext_of(&mime, "")));
        let data = m
            .decode_b64()
            .map_err(|e| JobError::validate(format!("{name}: bad base64 ({e})")))?;
        std::fs::write(&path, data)?;
        return Ok(LocalMedia {
            path,
            mime,
            source: m.clone(),
        });
    }
    if m.kind == "r2" {
        let (Some(store), Some(bucket)) = (store, bucket.filter(|b| !b.is_empty())) else {
            return Err(JobError::validate(format!(
                "{name}: r2 uploads need the Studio bucket (no R2 credentials)"
            )));
        };
        let mime = guess_mime(&m.value);
        let path = dest_dir.join(format!("{name}{}", ext_of(&mime, &suffix(&m.value))));
        store.get_file(bucket, &m.value, &path).map_err(|e| {
            JobError::validate(format!(
                "{name}: cannot read {} from the bucket ({e:#})",
                m.value
            ))
        })?;
        return Ok(LocalMedia {
            path,
            mime,
            source: m.clone(),
        });
    }
    // https
    let tmp = dest_dir.join(format!("{name}.download"));
    let (_, ctype, _) = match crate::httpc::download(&m.value, &tmp, 60.0, Some(MAX_DOWNLOAD_BYTES))
    {
        Ok(r) => r,
        Err(crate::httpc::Error::Transport(t)) if t.starts_with("over ") => {
            let _ = std::fs::remove_file(&tmp);
            return Err(JobError::validate(format!("{name}: {t}")));
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            return Err(JobError::validate(format!(
                "{name}: cannot download {} ({e})",
                m.value
            )));
        }
    };
    let mime = ctype.unwrap_or_else(|| guess_mime(&m.value));
    let path = dest_dir.join(format!(
        "{name}{}",
        ext_of(&mime, &suffix(m.value.split('?').next().unwrap_or("")))
    ));
    std::fs::rename(&tmp, &path)?;
    Ok(LocalMedia {
        path,
        mime,
        source: m.clone(),
    })
}

pub fn guess_mime(name: &str) -> String {
    let n = name.split('?').next().unwrap_or(name);
    let ext = suffix(n).trim_start_matches('.').to_ascii_lowercase();
    MIME_BY_EXT
        .iter()
        .find(|(e, _)| *e == ext)
        .map(|(_, m)| m.to_string())
        .unwrap_or_else(|| "application/octet-stream".into())
}

pub fn ext_of(mime: &str, fallback: &str) -> String {
    if let Some((_, e)) = EXT.iter().find(|(m, _)| *m == mime) {
        return e.to_string();
    }
    if let Some((e, _)) = MIME_BY_EXT.iter().find(|(_, m)| *m == mime) {
        return format!(".{e}");
    }
    if !fallback.is_empty() {
        return fallback.to_string();
    }
    ".bin".into()
}

pub fn to_data_uri(lm: &LocalMedia) -> std::io::Result<String> {
    Ok(format!(
        "data:{};base64,{}",
        lm.mime,
        b64_encode(&std::fs::read(&lm.path)?)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tempdir;

    #[test]
    fn b64_and_r2_media_land_in_the_job_dir() {
        let dir = tempdir("media");
        let m = Media {
            kind: "b64".into(),
            value: b64_encode(b"\x89PNG fake"),
            mime: Some("image/png".into()),
        };
        let lm = fetch_media(&m, &dir.join("in"), "first_image", None, None).unwrap();
        assert_eq!(lm.path.file_name().unwrap(), "first_image.png");
        assert_eq!((lm.kind(), lm.mime.as_str()), ("image", "image/png"));
        assert_eq!(
            to_data_uri(&lm).unwrap(),
            format!("data:image/png;base64,{}", b64_encode(b"\x89PNG fake"))
        );
        let store = crate::r2::LocalStore::new(dir.join("r2"));
        std::fs::write(dir.join("up.mp4"), b"mp4").unwrap();
        store
            .put_file("studio", "uploads/u1.mp4", &dir.join("up.mp4"), None)
            .unwrap();
        let m = Media::new("r2", "uploads/u1.mp4");
        let lm = fetch_media(&m, &dir.join("in"), "video", Some(&store), Some("studio")).unwrap();
        assert_eq!(lm.path.file_name().unwrap(), "video.mp4");
        assert_eq!(lm.mime, "video/mp4");
        let e = fetch_media(&m, &dir.join("in"), "video", None, None).unwrap_err();
        assert_eq!(e.where_, "validate");
        let e = fetch_media(
            &Media::new("r2", "uploads/missing.mp4"),
            &dir.join("in"),
            "video",
            Some(&store),
            Some("studio"),
        )
        .unwrap_err();
        assert!(e.message.contains("cannot read"));
        let bad = Media {
            kind: "b64".into(),
            value: "abcde".into(),
            mime: Some("image/png".into()),
        };
        assert!(fetch_media(&bad, &dir.join("in"), "x", None, None)
            .unwrap_err()
            .message
            .contains("bad base64"));
        assert_eq!(guess_mime("https://e/a.JPG?x=1"), "image/jpeg");
        assert_eq!(ext_of("audio/wav", ""), ".wav");
        assert_eq!(ext_of("application/x-unknown", ".foo"), ".foo");
        assert_eq!(ext_of("application/x-unknown", ""), ".bin");
    }
}
