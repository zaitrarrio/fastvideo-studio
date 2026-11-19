//! Object storage: Cloudflare R2 over its S3 API (SigV4 with `hmac` /
//! `sha2`, no AWS SDK), and a local-directory fake.
//!
//! Both expose the same small surface: `put_file`, `get_file`, `list_keys`,
//! `head_size`, `presign_get`, `get_bytes`. The worker reads weights from the
//! `fv-weights` bucket (fastvideo-rs's layout, docs/ops/r2-weights.md:
//! `weights/<dest>/<rev>/<path>` with `manifests/<dest>/<rev>.json`) and
//! writes job outputs under `jobs/<id>/` in the Studio bucket.

use crate::py::{amz_date, quote};
use crate::EnvMap;
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Clone, PartialEq)]
pub struct R2Config {
    pub endpoint: String,
    pub access_key_id: String,
    pub secret_access_key: String,
    /// the Studio bucket (outputs, uploads)
    pub bucket: String,
}

impl R2Config {
    pub fn from_env(e: &EnvMap) -> Option<R2Config> {
        let get = |a: &str, b: &str| {
            e.get(a)
                .filter(|v| !v.is_empty())
                .or_else(|| e.get(b).filter(|v| !v.is_empty()))
                .cloned()
        };
        Some(R2Config {
            endpoint: get("STUDIO_R2_ENDPOINT", "FV_R2_ENDPOINT")?,
            access_key_id: get("STUDIO_R2_ACCESS_KEY_ID", "FV_R2_ACCESS_KEY_ID")?,
            secret_access_key: get("STUDIO_R2_SECRET_ACCESS_KEY", "FV_R2_SECRET_ACCESS_KEY")?,
            bucket: get("STUDIO_R2_BUCKET", "FV_R2_BUCKET")?,
        })
    }
}

pub trait ObjectStore: Send + Sync {
    fn put_file(
        &self,
        bucket: &str,
        key: &str,
        path: &Path,
        content_type: Option<&str>,
    ) -> anyhow::Result<()>;
    fn get_file(&self, bucket: &str, key: &str, path: &Path) -> anyhow::Result<u64>;
    fn get_bytes(&self, bucket: &str, key: &str) -> anyhow::Result<Vec<u8>>;
    fn list_keys(&self, bucket: &str, prefix: &str) -> anyhow::Result<Vec<(String, u64)>>;
    fn head_size(&self, bucket: &str, key: &str) -> anyhow::Result<Option<u64>>;
    fn presign_get(&self, bucket: &str, key: &str, expires_s: u64) -> String;
}

/// A non-2xx answer from the object store (status, body, url).
#[derive(Debug, Clone, thiserror::Error)]
#[error("HTTP {status} from {url}: {}", body.chars().take(300).collect::<String>())]
pub struct R2Error {
    pub status: u16,
    pub body: String,
    pub url: String,
}

pub const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
pub const UNSIGNED_PAYLOAD: &str = "UNSIGNED-PAYLOAD";
const RETRY_STATUSES: [u16; 5] = [429, 500, 502, 503, 504];

/// AWS's URI encoding: everything but unreserved characters (botocore's
/// `percent_encode`, used for query keys and values).
pub fn uri_encode(s: &str) -> String {
    quote(s, "")
}

/// `urllib.parse.urlsplit`: (scheme, netloc, path, query).
pub fn urlsplit(url: &str) -> (String, String, String, String) {
    let (scheme, rest) = url.split_once("://").unwrap_or(("", url));
    let (rest, _fragment) = rest.split_once('#').unwrap_or((rest, ""));
    let (rest, query) = rest.split_once('?').unwrap_or((rest, ""));
    let (netloc, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, ""),
    };
    (
        scheme.to_string(),
        netloc.to_string(),
        path.to_string(),
        query.to_string(),
    )
}

/// `Host` as SigV4 signs it: lowercase, no userinfo, no default port
/// (botocore's `_host_from_url`).
pub fn host_header(url: &str) -> String {
    let (scheme, netloc, _, _) = urlsplit(url);
    let hostport = netloc.rsplit('@').next().unwrap_or("");
    let (host, port) = if let Some(end) = hostport.strip_prefix('[').and_then(|h| h.find(']')) {
        let h = &hostport[1..1 + end];
        let p = hostport[1 + end..]
            .strip_prefix(':')
            .and_then(|p| p.parse::<u16>().ok());
        (format!("[{}]", h.to_lowercase()), p)
    } else {
        match hostport.rsplit_once(':') {
            Some((h, p)) if p.parse::<u16>().is_ok() => (h.to_lowercase(), p.parse().ok()),
            _ => (hostport.to_lowercase(), None),
        }
    };
    let default = match scheme.as_str() {
        "http" => Some(80),
        "https" => Some(443),
        _ => None,
    };
    match port {
        Some(p) if Some(p) != default => format!("{host}:{p}"),
        _ => host,
    }
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("hmac key");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

pub fn sha256_hex(data: &[u8]) -> String {
    hex::encode(Sha256::digest(data))
}

/// AWS Signature Version 4 for S3-style requests: header auth for the API
/// calls and query auth for presigned GETs. The canonical request takes the
/// URL path as given (S3 does not normalise it) and the query string's
/// already-encoded `k=v` pairs sorted, like botocore's `S3SigV4Auth`
/// (`worker/tests/test_r2.py` holds the Python twin byte-identical to botocore).
#[derive(Debug, Clone)]
pub struct SigV4 {
    pub access_key_id: String,
    pub secret_access_key: String,
    pub region: String,
    pub service: String,
}

impl SigV4 {
    pub fn new(access_key_id: &str, secret_access_key: &str) -> SigV4 {
        SigV4 {
            access_key_id: access_key_id.into(),
            secret_access_key: secret_access_key.into(),
            region: "auto".into(),
            service: "s3".into(),
        }
    }

    pub fn with_scope(
        access_key_id: &str,
        secret_access_key: &str,
        region: &str,
        service: &str,
    ) -> SigV4 {
        SigV4 {
            access_key_id: access_key_id.into(),
            secret_access_key: secret_access_key.into(),
            region: region.into(),
            service: service.into(),
        }
    }

    pub fn scope(&self, date: &str) -> String {
        format!("{date}/{}/{}/aws4_request", self.region, self.service)
    }

    pub fn signing_key(&self, date: &str) -> Vec<u8> {
        let mut k = hmac_sha256(
            format!("AWS4{}", self.secret_access_key).as_bytes(),
            date.as_bytes(),
        );
        for part in [self.region.as_str(), self.service.as_str(), "aws4_request"] {
            k = hmac_sha256(&k, part.as_bytes());
        }
        k
    }

    pub fn canonical_query(url: &str) -> String {
        let (_, _, _, q) = urlsplit(url);
        if q.is_empty() {
            return String::new();
        }
        let mut pairs: Vec<(String, String)> = q
            .split('&')
            .map(|p| {
                p.split_once('=')
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .unwrap_or((p.to_string(), String::new()))
            })
            .collect();
        pairs.sort();
        pairs
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join("&")
    }

    /// (signed header names, signature) of the canonical request built from
    /// `headers` (every one of them is signed).
    fn signature(
        &self,
        method: &str,
        url: &str,
        headers: &[(String, String)],
        payload_hash: &str,
        amz_date: &str,
    ) -> (String, String) {
        let mut canon: Vec<(String, String)> = headers
            .iter()
            .map(|(k, v)| {
                (
                    k.to_lowercase(),
                    v.split_whitespace().collect::<Vec<_>>().join(" "),
                )
            })
            .collect();
        canon.sort();
        let signed = canon
            .iter()
            .map(|(k, _)| k.as_str())
            .collect::<Vec<_>>()
            .join(";");
        let (_, _, path, _) = urlsplit(url);
        let canonical_request = [
            method.to_uppercase(),
            if path.is_empty() {
                "/".to_string()
            } else {
                path
            },
            Self::canonical_query(url),
            canon
                .iter()
                .map(|(k, v)| format!("{k}:{v}\n"))
                .collect::<String>(),
            signed.clone(),
            payload_hash.to_string(),
        ]
        .join("\n");
        let date = &amz_date[..8];
        let string_to_sign = [
            "AWS4-HMAC-SHA256".to_string(),
            amz_date.to_string(),
            self.scope(date),
            sha256_hex(canonical_request.as_bytes()),
        ]
        .join("\n");
        let sig = hex::encode(hmac_sha256(
            &self.signing_key(date),
            string_to_sign.as_bytes(),
        ));
        (signed, sig)
    }

    /// The request headers with `Host`, `X-Amz-Date`, `X-Amz-Content-SHA256`
    /// and `Authorization` added; every header in the result but
    /// `Authorization` is signed. `now`: Unix seconds (None: the clock).
    pub fn sign_headers(
        &self,
        method: &str,
        url: &str,
        headers: &[(String, String)],
        payload_hash: &str,
        now: Option<i64>,
    ) -> Vec<(String, String)> {
        let amz = amz_date(now.unwrap_or_else(|| crate::now_s() as i64));
        let mut h: Vec<(String, String)> = headers
            .iter()
            .filter(|(k, _)| {
                ![
                    "authorization",
                    "x-amz-date",
                    "x-amz-content-sha256",
                    "host",
                ]
                .contains(&k.to_lowercase().as_str())
            })
            .cloned()
            .collect();
        h.push(("Host".into(), host_header(url)));
        h.push(("X-Amz-Date".into(), amz.clone()));
        h.push(("X-Amz-Content-SHA256".into(), payload_hash.into()));
        let (signed, sig) = self.signature(method, url, &h, payload_hash, &amz);
        h.push((
            "Authorization".into(),
            format!(
                "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={signed}, Signature={sig}",
                self.access_key_id,
                self.scope(&amz[..8])
            ),
        ));
        h
    }

    /// Query-string auth (`X-Amz-Algorithm` … `X-Amz-Signature`) over
    /// `host` only with an unsigned payload, appended to `url`'s own query.
    pub fn presign(&self, method: &str, url: &str, expires_s: u64, now: Option<i64>) -> String {
        let amz = amz_date(now.unwrap_or_else(|| crate::now_s() as i64));
        let auth = [
            ("X-Amz-Algorithm", "AWS4-HMAC-SHA256".to_string()),
            (
                "X-Amz-Credential",
                format!("{}/{}", self.access_key_id, self.scope(&amz[..8])),
            ),
            ("X-Amz-Date", amz.clone()),
            ("X-Amz-Expires", expires_s.to_string()),
            ("X-Amz-SignedHeaders", "host".to_string()),
        ];
        let (scheme, netloc, path, q) = urlsplit(url);
        let mut query = if q.is_empty() {
            String::new()
        } else {
            format!("{q}&")
        };
        query.push_str(
            &auth
                .iter()
                .map(|(k, v)| format!("{}={}", uri_encode(k), uri_encode(v)))
                .collect::<Vec<_>>()
                .join("&"),
        );
        let unsigned = format!("{scheme}://{netloc}{path}?{query}");
        let (_, sig) = self.signature(
            method,
            &unsigned,
            &[("host".to_string(), host_header(url))],
            UNSIGNED_PAYLOAD,
            &amz,
        );
        format!("{unsigned}&X-Amz-Signature={sig}")
    }
}

/// Cloudflare R2 over its S3 API: SigV4 header auth, path-style URLs
/// (`<endpoint>/<bucket>/<key>`), region `auto`. Uploads stream from disk
/// with an unsigned payload (`UNSIGNED-PAYLOAD` over TLS, which R2
/// accepts), downloads stream to a `.part` file; retried on connection
/// errors and 429 / 5xx.
pub struct R2Store {
    pub cfg: R2Config,
    pub endpoint: String,
    pub signer: SigV4,
    pub attempts: u32,
    pub backoff_s: f64,
    pub timeout_s: f64,
    sleep: Box<dyn Fn(f64) + Send + Sync>,
    agent: ureq::Agent,
}

impl R2Store {
    pub fn new(cfg: R2Config) -> R2Store {
        Self::with_sleep(
            cfg,
            Box::new(|s| std::thread::sleep(Duration::from_secs_f64(s))),
        )
    }

    pub fn with_sleep(cfg: R2Config, sleep: Box<dyn Fn(f64) + Send + Sync>) -> R2Store {
        let timeout_s = 120.0;
        R2Store {
            endpoint: cfg.endpoint.trim_end_matches('/').to_string(),
            signer: SigV4::new(&cfg.access_key_id, &cfg.secret_access_key),
            cfg,
            attempts: 4,
            backoff_s: 0.5,
            timeout_s,
            sleep,
            agent: ureq::AgentBuilder::new()
                .timeout_connect(Duration::from_secs(30))
                .timeout_read(Duration::from_secs_f64(timeout_s))
                .timeout_write(Duration::from_secs_f64(timeout_s))
                .build(),
        }
    }

    // ---- requests ------------------------------------------------------------

    pub fn url(&self, bucket: &str, key: &str, query: &[(&str, &str)]) -> String {
        let mut u = format!("{}/{}", self.endpoint, uri_encode(bucket));
        if !key.is_empty() {
            u.push('/');
            u.push_str(&key_url_part(key));
        }
        if !query.is_empty() {
            u.push('?');
            u.push_str(
                &query
                    .iter()
                    .map(|(k, v)| format!("{}={}", uri_encode(k), uri_encode(v)))
                    .collect::<Vec<_>>()
                    .join("&"),
            );
        }
        u
    }

    /// One signed request; the response (2xx) or an error (`R2Error` for a
    /// non-2xx status, the transport's text otherwise).
    fn open(
        &self,
        method: &str,
        url: &str,
        headers: &[(String, String)],
        body: Option<Box<dyn Read + Send>>,
        payload_hash: &str,
        extra: &[(String, String)],
    ) -> anyhow::Result<ureq::Response> {
        let mut req = self.agent.request(method, url);
        for (k, v) in self
            .signer
            .sign_headers(method, url, headers, payload_hash, None)
            .iter()
            .chain(extra.iter())
        {
            req = req.set(k, v); // `extra` is unsigned (Content-Length)
        }
        let r = match body {
            Some(b) => req.send(b),
            None => req.call(),
        };
        match r {
            Ok(resp) => Ok(resp),
            Err(ureq::Error::Status(status, resp)) => {
                let text = if method != "HEAD" {
                    let mut s = String::new();
                    let _ = resp.into_reader().take(1 << 20).read_to_string(&mut s);
                    s
                } else {
                    String::new()
                };
                Err(R2Error {
                    status,
                    body: text,
                    url: url.to_string(),
                }
                .into())
            }
            Err(ureq::Error::Transport(t)) => Err(anyhow::anyhow!("{t}")),
        }
    }

    fn retry<T>(&self, op: &dyn Fn() -> anyhow::Result<T>) -> anyhow::Result<T> {
        let mut last = None;
        for i in 0..self.attempts {
            match op() {
                Ok(v) => return Ok(v),
                Err(e) => {
                    if let Some(r) = e.downcast_ref::<R2Error>() {
                        if !RETRY_STATUSES.contains(&r.status) {
                            return Err(e);
                        }
                    }
                    last = Some(e);
                }
            }
            if i + 1 < self.attempts {
                (self.sleep)(self.backoff_s * 2f64.powi(i as i32));
            }
        }
        Err(last.unwrap())
    }
}

impl ObjectStore for R2Store {
    fn put_file(
        &self,
        bucket: &str,
        key: &str,
        path: &Path,
        content_type: Option<&str>,
    ) -> anyhow::Result<()> {
        let size = std::fs::metadata(path)?.len();
        let url = self.url(bucket, key, &[]);
        let headers = vec![(
            "Content-Type".to_string(),
            content_type
                .map(str::to_string)
                .unwrap_or_else(|| guess_type(path)),
        )];
        let extra = vec![("Content-Length".to_string(), size.to_string())];
        let path = path.to_path_buf();
        self.retry(&|| {
            let f = std::fs::File::open(&path)?;
            let r = self.open(
                "PUT",
                &url,
                &headers,
                Some(Box::new(f)),
                UNSIGNED_PAYLOAD,
                &extra,
            )?;
            let _ = r.into_reader().take(1 << 20).read_to_end(&mut Vec::new());
            Ok(())
        })
    }

    fn get_file(&self, bucket: &str, key: &str, path: &Path) -> anyhow::Result<u64> {
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p)?;
        }
        let tmp = part_path(path);
        let url = self.url(bucket, key, &[]);
        let res = self.retry(&|| {
            let r = self.open("GET", &url, &[], None, EMPTY_SHA256, &[])?;
            let mut f = std::fs::File::create(&tmp)?;
            std::io::copy(&mut r.into_reader(), &mut f)?;
            Ok(())
        });
        if let Err(e) = res {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        std::fs::rename(&tmp, path)?;
        Ok(std::fs::metadata(path)?.len())
    }

    fn get_bytes(&self, bucket: &str, key: &str) -> anyhow::Result<Vec<u8>> {
        let url = self.url(bucket, key, &[]);
        self.retry(&|| {
            let r = self.open("GET", &url, &[], None, EMPTY_SHA256, &[])?;
            let mut v = Vec::new();
            r.into_reader().read_to_end(&mut v)?;
            Ok(v)
        })
    }

    fn list_keys(&self, bucket: &str, prefix: &str) -> anyhow::Result<Vec<(String, u64)>> {
        let mut out = Vec::new();
        let mut token: Option<String> = None;
        loop {
            let mut query: Vec<(&str, &str)> = vec![("list-type", "2"), ("prefix", prefix)];
            if let Some(t) = &token {
                query.push(("continuation-token", t));
            }
            let url = self.url(bucket, "", &query);
            let body = self.retry(&|| {
                let r = self.open("GET", &url, &[], None, EMPTY_SHA256, &[])?;
                let mut v = Vec::new();
                r.into_reader().read_to_end(&mut v)?;
                Ok(v)
            })?;
            let text = String::from_utf8_lossy(&body).into_owned();
            let doc = roxmltree::Document::parse(&text)
                .map_err(|e| anyhow::anyhow!("ListObjectsV2: {e}"))?;
            let mut truncated = false;
            token = None;
            for el in doc.root_element().children().filter(|n| n.is_element()) {
                match el.tag_name().name() {
                    "Contents" => {
                        let mut key = String::new();
                        let mut size = 0u64;
                        for c in el.children().filter(|n| n.is_element()) {
                            match c.tag_name().name() {
                                "Key" => key = c.text().unwrap_or("").to_string(),
                                "Size" => {
                                    size = c.text().unwrap_or("0").trim().parse().unwrap_or(0)
                                }
                                _ => {}
                            }
                        }
                        out.push((key, size));
                    }
                    "IsTruncated" => {
                        truncated = el.text().unwrap_or("").trim().eq_ignore_ascii_case("true")
                    }
                    "NextContinuationToken" => {
                        token = el
                            .text()
                            .map(|t| t.trim().to_string())
                            .filter(|t| !t.is_empty())
                    }
                    _ => {}
                }
            }
            if !(truncated && token.is_some()) {
                return Ok(out);
            }
        }
    }

    fn head_size(&self, bucket: &str, key: &str) -> anyhow::Result<Option<u64>> {
        let url = self.url(bucket, key, &[]);
        self.retry(
            &|| match self.open("HEAD", &url, &[], None, EMPTY_SHA256, &[]) {
                Ok(r) => Ok(r
                    .header("Content-Length")
                    .and_then(|v| v.trim().parse::<u64>().ok())),
                Err(e) => match e.downcast_ref::<R2Error>() {
                    Some(r) if r.status == 404 || r.status == 403 => Ok(None),
                    _ => Err(e),
                },
            },
        )
    }

    fn presign_get(&self, bucket: &str, key: &str, expires_s: u64) -> String {
        self.signer
            .presign("GET", &self.url(bucket, key, &[]), expires_s, None)
    }
}

/// A directory per bucket; presigned URLs are `file://` paths plus a
/// fake query string. For tests and local runs.
pub struct LocalStore {
    pub root: PathBuf,
}

impl LocalStore {
    pub fn new(root: PathBuf) -> LocalStore {
        LocalStore { root }
    }

    fn p(&self, bucket: &str, key: &str) -> PathBuf {
        self.root.join(bucket).join(key)
    }
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.is_file() {
                out.push(p);
            }
        }
    }
}

impl ObjectStore for LocalStore {
    fn put_file(
        &self,
        bucket: &str,
        key: &str,
        path: &Path,
        _content_type: Option<&str>,
    ) -> anyhow::Result<()> {
        let dst = self.p(bucket, key);
        if let Some(p) = dst.parent() {
            std::fs::create_dir_all(p)?;
        }
        std::fs::copy(path, &dst)?;
        Ok(())
    }

    fn get_file(&self, bucket: &str, key: &str, path: &Path) -> anyhow::Result<u64> {
        let src = self.p(bucket, key);
        if !src.is_file() {
            anyhow::bail!("FileNotFoundError: {bucket}/{key}");
        }
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p)?;
        }
        std::fs::copy(&src, path)?;
        Ok(std::fs::metadata(path)?.len())
    }

    fn get_bytes(&self, bucket: &str, key: &str) -> anyhow::Result<Vec<u8>> {
        Ok(std::fs::read(self.p(bucket, key))?)
    }

    fn list_keys(&self, bucket: &str, prefix: &str) -> anyhow::Result<Vec<(String, u64)>> {
        let base = self.root.join(bucket);
        if !base.is_dir() {
            return Ok(Vec::new());
        }
        let mut files = Vec::new();
        walk(&base, &mut files);
        files.sort();
        let mut out = Vec::new();
        for p in files {
            let key = p
                .strip_prefix(&base)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            if key.starts_with(prefix) {
                out.push((key, p.metadata()?.len()));
            }
        }
        Ok(out)
    }

    fn head_size(&self, bucket: &str, key: &str) -> anyhow::Result<Option<u64>> {
        let p = self.p(bucket, key);
        Ok(if p.is_file() {
            Some(p.metadata()?.len())
        } else {
            None
        })
    }

    fn presign_get(&self, bucket: &str, key: &str, expires_s: u64) -> String {
        let exp = crate::now_s() as i64 + expires_s as i64;
        let sig = &sha256_hex(format!("{bucket}/{key}/{exp}").as_bytes())[..16];
        format!(
            "file://{}?X-Expires={exp}&X-Sig={sig}",
            self.p(bucket, key).display()
        )
    }
}

pub fn part_path(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    path.with_file_name(format!("{name}.part"))
}

pub fn guess_type(path: &Path) -> String {
    let name = path.to_string_lossy();
    let t = crate::media::guess_mime(&name);
    if t != "application/octet-stream" {
        return t;
    }
    match path.extension().and_then(|e| e.to_str()) {
        Some("y4m") => "video/x-yuv4mpeg",
        Some("mp4") => "video/mp4",
        Some("wav") => "audio/wav",
        Some("jpg") => "image/jpeg",
        _ => "application/octet-stream",
    }
    .into()
}

pub fn key_url_part(key: &str) -> String {
    quote(key, "/")
}

pub fn read_json(
    store: &dyn ObjectStore,
    bucket: &str,
    key: &str,
) -> anyhow::Result<serde_json::Value> {
    Ok(serde_json::from_slice(&store.get_bytes(bucket, key)?)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: i64 = 1_795_091_696; // 2026-11-19T12:34:56Z
    const ENDPOINT: &str = "https://0123456789abcdef.r2.cloudflarestorage.com";
    const AK: &str = "AKIDEXAMPLE";
    const SK: &str = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";

    fn store() -> R2Store {
        R2Store::with_sleep(
            R2Config {
                endpoint: ENDPOINT.into(),
                access_key_id: AK.into(),
                secret_access_key: SK.into(),
                bucket: "studio".into(),
            },
            Box::new(|_| {}),
        )
    }

    #[test]
    fn signing_key_matches_the_aws_example() {
        // docs.aws.amazon.com/general/latest/gr/signature-v4-examples.html
        let sig = SigV4::with_scope(AK, SK, "us-east-1", "iam");
        assert_eq!(
            hex::encode(sig.signing_key("20150830")),
            "c4afb1cc5771d871763a393e44b703571b55cc28424d1a5e86da6ed3c154a4b9"
        );
    }

    #[test]
    fn aws_style_vector_at_a_fixed_time() {
        // Region / service scoped like the AWS suite's `get-vanilla-query` case,
        // at 2015-08-30T12:36:00Z, with the payload hash header S3 signs.
        let sig = SigV4::with_scope(AK, SK, "us-east-1", "service");
        let h = sig.sign_headers(
            "GET",
            "https://example.amazonaws.com/?Param2=value2&Param1=value1",
            &[],
            EMPTY_SHA256,
            Some(1_440_938_160),
        );
        let auth = h
            .iter()
            .find(|(k, _)| k == "Authorization")
            .unwrap()
            .1
            .clone();
        assert_eq!(auth, "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature=311c7f58b10b06de8540bb5a27f441ee0609f1d5ad7b191e68d7ea87d90e3d6b");
    }

    /// Header auth at a fixed timestamp: the `worker/tests/test_r2.py` cases,
    /// whose signatures the Python signer produces byte-identical to
    /// botocore's `S3SigV4Auth` (recorded from it, 2026-11-19).
    #[test]
    fn header_auth_matches_the_python_signer_and_botocore() {
        let s = store();
        let body_hash = sha256_hex(b"{\"output\": {\"model\": \"fasth3-v2\"}}\n");
        let cases: Vec<(&str, &str, &str, &str, Vec<(&str, &str)>, Vec<(&str, &str)>, &str, &str, &str)> = vec![
            ("get object", "GET", "studio", "jobs/abc-1/video.mp4", vec![], vec![], EMPTY_SHA256,
             "https://0123456789abcdef.r2.cloudflarestorage.com/studio/jobs/abc-1/video.mp4",
             "SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature=b3e762c7463dd5704b2cc9986f5ff797a4602a41340911836ebe4861a7c7f281"),
            ("head object", "HEAD", "fv-weights", "weights/h3-8step/rev1/transformer/model-00001-of-00009.safetensors", vec![], vec![], EMPTY_SHA256,
             "https://0123456789abcdef.r2.cloudflarestorage.com/fv-weights/weights/h3-8step/rev1/transformer/model-00001-of-00009.safetensors",
             "SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature=b89294c3453fae039b35f1ae6763537c2c1000cbaf1c0509c68e6dccd10be20d"),
            ("key with spaces, unicode and reserved characters", "GET", "studio", "uploads/a b/ü (1)+x&y=z#!'*.png", vec![], vec![], EMPTY_SHA256,
             "https://0123456789abcdef.r2.cloudflarestorage.com/studio/uploads/a%20b/%C3%BC%20%281%29%2Bx%26y%3Dz%23%21%27%2A.png",
             "SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature=48cd5add416d4600601da90abe81281c5fb5c1458d4fe098ad7652d6e656b2b4"),
            ("list objects v2", "GET", "fv-weights", "", vec![("list-type", "2"), ("prefix", "manifests/h3 rs/")], vec![], EMPTY_SHA256,
             "https://0123456789abcdef.r2.cloudflarestorage.com/fv-weights?list-type=2&prefix=manifests%2Fh3%20rs%2F",
             "SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature=6a604fdbf091c0168212aa76c4846aaa92d59afb3759e437a0363bc123f2b29f"),
            ("list objects v2, continuation token", "GET", "fv-weights", "", vec![("list-type", "2"), ("prefix", "weights/"), ("continuation-token", "1ueGcxLPRx1Tr/XYExHnhbYLgveDs2J/wm36Hy4vbOwM=")], vec![], EMPTY_SHA256,
             "https://0123456789abcdef.r2.cloudflarestorage.com/fv-weights?list-type=2&prefix=weights%2F&continuation-token=1ueGcxLPRx1Tr%2FXYExHnhbYLgveDs2J%2Fwm36Hy4vbOwM%3D",
             "SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature=17e4aa3d24911c56fdead9e6b883729ebba01e266ff00bcb2c0a2e83cbb042d8"),
            ("put bytes, signed payload", "PUT", "studio", "jobs/abc-1/job.json", vec![], vec![("Content-Type", "application/json")], body_hash.as_str(),
             "https://0123456789abcdef.r2.cloudflarestorage.com/studio/jobs/abc-1/job.json",
             "SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date, Signature=c30f51b0bb950b31d8dd88699fbbb84eaea8a8ef0ab43b0bb7ee11f9234b8c92"),
            ("put file, unsigned payload", "PUT", "studio", "jobs/abc-1/video.mp4", vec![], vec![("Content-Type", "video/mp4")], UNSIGNED_PAYLOAD,
             "https://0123456789abcdef.r2.cloudflarestorage.com/studio/jobs/abc-1/video.mp4",
             "SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date, Signature=01530bc51ceea1dc8dc624adca0bdec808779d628732edd93441c328d8a3af25"),
            ("header value with inner whitespace", "PUT", "studio", "k", vec![], vec![("Content-Type", "text/plain;  charset=utf-8 ")], UNSIGNED_PAYLOAD,
             "https://0123456789abcdef.r2.cloudflarestorage.com/studio/k",
             "SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date, Signature=1662cf1bf7a4a0a2e11c3db01f273988eeac9c3a9a6ce0f7f9eb0a6082a2cd9b"),
        ];
        for (name, method, bucket, key, query, headers, payload, want_url, want_tail) in cases {
            let url = s.url(bucket, key, &query);
            assert_eq!(url, want_url, "{name}");
            let hs: Vec<(String, String)> = headers
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            let ours = s.signer.sign_headers(method, &url, &hs, payload, Some(NOW));
            let get = |k: &str| {
                ours.iter()
                    .find(|(x, _)| x == k)
                    .map(|(_, v)| v.clone())
                    .unwrap()
            };
            assert_eq!(get("X-Amz-Date"), "20261119T123456Z");
            assert_eq!(get("X-Amz-Content-SHA256"), payload);
            assert_eq!(get("Authorization"), format!("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20261119/auto/s3/aws4_request, {want_tail}"), "{name}");
            // Nothing we send is unsigned, apart from Authorization itself.
            let signed: Vec<&str> = want_tail
                .split("SignedHeaders=")
                .nth(1)
                .unwrap()
                .split(',')
                .next()
                .unwrap()
                .split(';')
                .collect();
            let mut names: Vec<String> = ours
                .iter()
                .filter(|(k, _)| k != "Authorization")
                .map(|(k, _)| k.to_lowercase())
                .collect();
            names.sort();
            assert_eq!(names, signed, "{name}");
        }
    }

    /// Presigned GETs at a fixed timestamp, equal to botocore's
    /// `S3SigV4QueryAuth` through the Python twin.
    #[test]
    fn presigned_get_matches_the_python_signer_and_botocore() {
        let s = store();
        for (bucket, key, expires, want) in [
            ("studio", "jobs/abc-1/video.mp4", 86400u64, "https://0123456789abcdef.r2.cloudflarestorage.com/studio/jobs/abc-1/video.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIDEXAMPLE%2F20261119%2Fauto%2Fs3%2Faws4_request&X-Amz-Date=20261119T123456Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=c2fc997d5c2085e38bda6d350bb1acbaccbabf95c4d1bc475dd51e4add742e07"),
            ("studio", "uploads/ü (1)+x&y.png", 600, "https://0123456789abcdef.r2.cloudflarestorage.com/studio/uploads/%C3%BC%20%281%29%2Bx%26y.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIDEXAMPLE%2F20261119%2Fauto%2Fs3%2Faws4_request&X-Amz-Date=20261119T123456Z&X-Amz-Expires=600&X-Amz-SignedHeaders=host&X-Amz-Signature=547721014bb8d7dac7bf985009cc1fc8fccd400f6b9169b65733360dd8fe1ea6"),
            ("fv-weights", "weights/fastwan21-1.3b/rev2/model.safetensors", 1, "https://0123456789abcdef.r2.cloudflarestorage.com/fv-weights/weights/fastwan21-1.3b/rev2/model.safetensors?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIDEXAMPLE%2F20261119%2Fauto%2Fs3%2Faws4_request&X-Amz-Date=20261119T123456Z&X-Amz-Expires=1&X-Amz-SignedHeaders=host&X-Amz-Signature=9d8d841b39a5137ac151b78486df9b6c2e4d1af7815e93dd3f93bbe77030f7e7"),
        ] {
            assert_eq!(s.signer.presign("GET", &s.url(bucket, key, &[]), expires, Some(NOW)), want);
        }
    }

    #[test]
    fn presign_get_is_the_query_form() {
        let s = store();
        let url = s.presign_get("studio", "jobs/x/video.mp4", 3600);
        assert!(url.starts_with(&format!(
            "{ENDPOINT}/studio/jobs/x/video.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential="
        )));
        assert!(url.contains("&X-Amz-Signature=") && url.contains("X-Amz-Expires=3600"));
        let fixed = s.signer.presign(
            "GET",
            &s.url("studio", "jobs/abc-1/video.mp4", &[]),
            86400,
            Some(NOW),
        );
        assert!(fixed.contains("X-Amz-Credential=AKIDEXAMPLE%2F20261119%2Fauto%2Fs3%2Faws4_request&X-Amz-Date=20261119T123456Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature="));
        let sig = fixed.rsplit('=').next().unwrap();
        assert_eq!(sig.len(), 64);
    }

    #[test]
    fn host_header_and_urlsplit() {
        assert_eq!(host_header("https://A.example.com/x"), "a.example.com");
        assert_eq!(host_header("https://a.example.com:443/x"), "a.example.com");
        assert_eq!(host_header("http://127.0.0.1:8080/x?y"), "127.0.0.1:8080");
        assert_eq!(host_header("http://u:p@h.com/x"), "h.com");
        assert_eq!(
            urlsplit("https://h/p/q?a=1&b=2#f"),
            ("https".into(), "h".into(), "/p/q".into(), "a=1&b=2".into())
        );
        assert_eq!(SigV4::canonical_query("https://h/?b=2&a=1"), "a=1&b=2");
        assert_eq!(
            key_url_part("jobs/j1/ü thumb.jpg"),
            "jobs/j1/%C3%BC%20thumb.jpg"
        );
        assert_eq!(
            part_path(Path::new("/a/video.mp4")),
            PathBuf::from("/a/video.mp4.part")
        );
    }

    #[test]
    fn local_store_roundtrip() {
        let dir = crate::tempdir("localstore");
        let st = LocalStore::new(dir.join("r2"));
        std::fs::write(dir.join("a.txt"), b"hello").unwrap();
        st.put_file("b", "x/y/a.txt", &dir.join("a.txt"), None)
            .unwrap();
        st.put_file("b", "x/b.txt", &dir.join("a.txt"), None)
            .unwrap();
        assert_eq!(st.get_bytes("b", "x/y/a.txt").unwrap(), b"hello");
        assert_eq!(st.head_size("b", "x/b.txt").unwrap(), Some(5));
        assert_eq!(st.head_size("b", "nope").unwrap(), None);
        assert_eq!(
            st.list_keys("b", "x/").unwrap(),
            vec![("x/b.txt".to_string(), 5), ("x/y/a.txt".to_string(), 5)]
        );
        assert_eq!(
            st.get_file("b", "x/b.txt", &dir.join("out/b.txt")).unwrap(),
            5
        );
        assert!(st.get_file("b", "nope", &dir.join("out/n")).is_err());
        let u = st.presign_get("b", "x/b.txt", 60);
        assert!(u.starts_with("file://") && u.contains("?X-Expires=") && u.contains("&X-Sig="));
    }
}
