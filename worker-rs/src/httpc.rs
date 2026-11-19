//! A tiny blocking client for the sidecars on 127.0.0.1 (no `Origin`
//! header: `h3 serve` / `fastwan serve` refuse foreign origins).

use serde_json::Value;
use std::io::Read;
use std::time::Duration;

#[derive(Debug, thiserror::Error)]
#[error("HTTP {status} from {url}: {}", body.chars().take(300).collect::<String>())]
pub struct HttpError {
    pub status: u16,
    pub body: String,
    pub url: String,
}

impl HttpError {
    pub fn json(&self) -> Option<Value> {
        serde_json::from_str(&self.body).ok()
    }
}

/// Why a request produced no answer: a transport error, or a non-2xx status.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Status(#[from] HttpError),
    #[error("{0}")]
    Transport(String),
}

pub fn agent(timeout: Duration) -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(10))
        .timeout(timeout)
        .build()
}

fn finish(
    url: &str,
    r: Result<ureq::Response, ureq::Error>,
) -> Result<(u16, Vec<u8>, Vec<(String, String)>), Error> {
    match r {
        Ok(resp) => {
            let headers: Vec<(String, String)> = resp
                .headers_names()
                .iter()
                .map(|n| (n.clone(), resp.header(n).unwrap_or("").to_string()))
                .collect();
            let status = resp.status();
            let mut body = Vec::new();
            resp.into_reader()
                .read_to_end(&mut body)
                .map_err(|e| Error::Transport(e.to_string()))?;
            Ok((status, body, headers))
        }
        Err(ureq::Error::Status(code, resp)) => {
            let mut body = Vec::new();
            let _ = resp.into_reader().read_to_end(&mut body);
            Err(Error::Status(HttpError {
                status: code,
                body: String::from_utf8_lossy(&body).into_owned(),
                url: url.to_string(),
            }))
        }
        Err(ureq::Error::Transport(t)) => Err(Error::Transport(t.to_string())),
    }
}

pub fn get_bytes(url: &str, timeout_s: f64) -> Result<Vec<u8>, Error> {
    Ok(finish(
        url,
        agent(Duration::from_secs_f64(timeout_s)).get(url).call(),
    )?
    .1)
}

pub fn get_json(url: &str, timeout_s: f64) -> Result<Value, Error> {
    let body = get_bytes(url, timeout_s)?;
    serde_json::from_slice(&body).map_err(|e| Error::Transport(format!("{url}: not JSON: {e}")))
}

/// Status and body without failing on 4xx/5xx.
pub fn get_status(url: &str, timeout_s: f64) -> Result<(u16, Vec<u8>), Error> {
    match finish(
        url,
        agent(Duration::from_secs_f64(timeout_s)).get(url).call(),
    ) {
        Ok((s, b, _)) => Ok((s, b)),
        Err(Error::Status(e)) => Ok((e.status, e.body.into_bytes())),
        Err(e) => Err(e),
    }
}

pub fn post_json(url: &str, body: &Value, timeout_s: f64) -> Result<Option<Value>, Error> {
    let data = serde_json::to_vec(body).map_err(|e| Error::Transport(e.to_string()))?;
    let r = agent(Duration::from_secs_f64(timeout_s))
        .post(url)
        .set("Content-Type", "application/json")
        .send_bytes(&data);
    let (_, out, _) = finish(url, r)?;
    if out.iter().all(|b| b.is_ascii_whitespace()) {
        return Ok(None);
    }
    serde_json::from_slice(&out)
        .map(Some)
        .map_err(|e| Error::Transport(format!("{url}: not JSON: {e}")))
}

pub fn delete(url: &str, timeout_s: f64) -> Result<(), Error> {
    match finish(
        url,
        agent(Duration::from_secs_f64(timeout_s)).delete(url).call(),
    ) {
        Ok(_) => Ok(()),
        Err(Error::Status(e)) if e.status == 404 => Ok(()),
        Err(e) => Err(e),
    }
}

/// A GET with a `User-Agent`, streamed to a file; returns the bytes written
/// and the `Content-Type`. `max_bytes` stops a runaway download.
pub fn download(
    url: &str,
    dest: &std::path::Path,
    timeout_s: f64,
    max_bytes: Option<u64>,
) -> Result<(u64, Option<String>, Option<u64>), Error> {
    let r = agent(Duration::from_secs_f64(timeout_s))
        .get(url)
        .set("User-Agent", "fastvideo-studio-worker/1")
        .call();
    let resp = match r {
        Ok(resp) => resp,
        Err(ureq::Error::Status(code, resp)) => {
            let mut body = Vec::new();
            let _ = resp.into_reader().read_to_end(&mut body);
            return Err(Error::Status(HttpError {
                status: code,
                body: String::from_utf8_lossy(&body).into_owned(),
                url: url.to_string(),
            }));
        }
        Err(ureq::Error::Transport(t)) => return Err(Error::Transport(t.to_string())),
    };
    let ctype = resp
        .header("Content-Type")
        .map(|s| s.split(';').next().unwrap_or("").trim().to_string())
        .filter(|s| !s.is_empty());
    let clen = resp
        .header("Content-Length")
        .and_then(|s| s.trim().parse::<u64>().ok());
    let mut reader = resp.into_reader();
    let mut f = std::fs::File::create(dest).map_err(|e| Error::Transport(e.to_string()))?;
    let mut buf = vec![0u8; 1 << 20];
    let mut total = 0u64;
    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| Error::Transport(e.to_string()))?;
        if n == 0 {
            break;
        }
        total += n as u64;
        if let Some(max) = max_bytes {
            if total > max {
                return Err(Error::Transport(format!("over {} MiB", max >> 20)));
            }
        }
        std::io::Write::write_all(&mut f, &buf[..n])
            .map_err(|e| Error::Transport(e.to_string()))?;
    }
    Ok((total, ctype, clen))
}
