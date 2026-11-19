//! Weights resolver: volume → stage dir → R2 / https fetch, and the symlink
//! root `/workspace/w` every recipe is written against.
//!
//! Catalog `trees` entries (catalog.json `trees.note`):
//! - `weights/<tree>` and `h3-rs/<sub>`: paths relative to the volume
//!   (`/runpod-volume/weights/<tree>`, `/runpod-volume/h3-rs/<sub>`). A path may
//!   name a file inside a tree (`weights/auxiliary/tae/taeh3.safetensors`).
//!   Missing from the volume, a `weights/<tree>` is fetched from the R2 weights
//!   store (`fv-weights`, `weights/<dest>/<rev>/…` + `manifests/<dest>/<rev>.json`).
//! - `r2:<bucket>/<prefix>=<name>`: fetched to the stage dir on first use. A
//!   prefix of the weights store resolves its `<rev>` through the manifest.
//! - `https:<host>/<path>=<name>`: one file, fetched on first use.
//!
//! `stage` entries follow h3-rs `scripts/runpod/make_boot.py`'s `stage TREE
//! [SUB...]`: copy the tree (or only SUB directories, the rest symlinked back
//! to the volume) onto the container disk with 8 parallel `cp --update=none`,
//! verify every file's size, then point `/workspace/w/TREE` at the copy.
//!
//! The volume is add-only and this module never writes under it.

use crate::errors::{JobError, JobResult};
use crate::py::{dumps, realpath};
use crate::r2::ObjectStore;
use crate::{now_s, EnvMap, Log};
use serde_json::{json, Map, Value};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub const STAGE_PARALLEL: usize = 8;
pub const COMPLETE_MARKER: &str = ".complete";
pub const REVISION_MARKER: &str = ".fv-revision";
/// files this worker fetched (subset fetches)
pub const STUDIO_FILES_MARKER: &str = ".studio-files";

#[derive(Debug, Clone, PartialEq)]
pub struct Roots {
    pub volume: PathBuf,
    pub volume_weights: PathBuf,
    pub volume_extras: PathBuf,
    pub stage_dir: PathBuf,
    pub link_root: PathBuf,
}

impl Default for Roots {
    fn default() -> Self {
        Roots {
            volume: "/runpod-volume".into(),
            volume_weights: "/runpod-volume/weights".into(),
            volume_extras: "/runpod-volume/h3-rs".into(),
            stage_dir: "/workspace/stage".into(),
            link_root: "/workspace/w".into(),
        }
    }
}

impl Roots {
    pub fn from_catalog(trees: &Map<String, Value>, env: &EnvMap) -> Roots {
        let pick = |var: &str, key: &str, default: &str| -> PathBuf {
            env.get(var)
                .filter(|v| !v.is_empty())
                .cloned()
                .unwrap_or_else(|| {
                    trees
                        .get(key)
                        .and_then(|v| v.as_str())
                        .unwrap_or(default)
                        .to_string()
                })
                .into()
        };
        Roots {
            volume: pick("STUDIO_VOLUME", "volume", "/runpod-volume"),
            volume_weights: pick(
                "STUDIO_VOLUME_WEIGHTS",
                "volume_weights",
                "/runpod-volume/weights",
            ),
            volume_extras: pick(
                "STUDIO_VOLUME_EXTRAS",
                "volume_extras",
                "/runpod-volume/h3-rs",
            ),
            stage_dir: pick("STUDIO_STAGE_DIR", "stage_dir", "/workspace/stage"),
            link_root: pick("STUDIO_LINK_ROOT", "link_root", "/workspace/w"),
        }
    }
}

/// A parsed catalog `trees` entry.
#[derive(Debug, Clone, PartialEq)]
pub struct TreeEntry {
    /// volume | r2 | https
    pub kind: String,
    /// volume-relative path, `bucket/prefix`, or `host/path`
    pub source: String,
    /// the path under the link root
    pub name: String,
}

impl TreeEntry {
    pub fn parse(entry: &str) -> Result<TreeEntry, String> {
        let t = |k: &str, s: &str, n: &str| TreeEntry {
            kind: k.into(),
            source: s.into(),
            name: n.into(),
        };
        if let Some(rest) = entry.strip_prefix("r2:") {
            let (src, name) = rest
                .split_once('=')
                .ok_or_else(|| format!("unknown tree entry {}", crate::py::py_str_repr(entry)))?;
            return Ok(t("r2", src.trim_matches('/'), name.trim_matches('/')));
        }
        if let Some(rest) = entry.strip_prefix("https:") {
            let (src, name) = rest
                .split_once('=')
                .ok_or_else(|| format!("unknown tree entry {}", crate::py::py_str_repr(entry)))?;
            return Ok(t("https", src.trim_matches('/'), name.trim_matches('/')));
        }
        if let Some(rest) = entry.strip_prefix("weights/") {
            return Ok(t("volume", entry, rest));
        }
        if entry.starts_with("h3-rs/") {
            return Ok(t("volume", entry, entry));
        }
        Err(format!(
            "unknown tree entry {}",
            crate::py::py_str_repr(entry)
        ))
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TreeStatus {
    pub entry: String,
    pub name: String,
    pub volume: bool,
    pub staged: bool,
    pub fetched: bool,
    pub linked: bool,
    pub note: String,
}

impl TreeStatus {
    pub fn to_json(&self) -> Value {
        json!({"entry": self.entry, "name": self.name, "volume": self.volume, "staged": self.staged, "fetched": self.fetched, "linked": self.linked, "note": self.note})
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnsureReport {
    pub stage_s: f64,
    pub fetched: Vec<String>,
    pub staged: Vec<String>,
}

pub struct WeightsResolver {
    pub roots: Roots,
    pub store: Option<Arc<dyn ObjectStore>>,
    pub weights_bucket: String,
    pub log: Log,
    /// tests: never copy
    pub dry_run: bool,
}

fn is_symlink(p: &Path) -> bool {
    std::fs::symlink_metadata(p)
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false)
}

fn exists(p: &Path) -> bool {
    p.exists() // follows symlinks, as `Path.exists()` does
}

fn under(p: &Path, root: &Path) -> bool {
    p.starts_with(root)
}

/// `[p for p in s.rglob("*") if p.is_file()]`: symlinked directories are not followed.
fn files_under(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.is_file() {
            out.push(p);
        } else if p.is_dir() && !is_symlink(&p) {
            files_under(&p, out);
        }
    }
}

impl WeightsResolver {
    pub fn new(
        roots: Roots,
        store: Option<Arc<dyn ObjectStore>>,
        log: Log,
        dry_run: bool,
    ) -> WeightsResolver {
        WeightsResolver {
            roots,
            store,
            weights_bucket: "fv-weights".into(),
            log,
            dry_run,
        }
    }

    // ---- status ------------------------------------------------------------

    pub fn status(&self, entry: &str) -> Result<TreeStatus, String> {
        let t = TreeEntry::parse(entry)?;
        let mut st = TreeStatus {
            entry: entry.into(),
            name: t.name.clone(),
            ..Default::default()
        };
        let vol = self.volume_path(&t);
        st.volume = vol.map(|v| exists(&v)).unwrap_or(false);
        let staged = self.roots.stage_dir.join(&t.name);
        st.staged = if staged.is_dir() {
            exists(&staged) && exists(&staged.join(COMPLETE_MARKER))
        } else {
            staged.is_file()
        };
        st.fetched = exists(&staged) && t.kind != "volume";
        st.linked = exists(&self.roots.link_root.join(&t.name));
        Ok(st)
    }

    // ---- ensure --------------------------------------------------------------

    /// Every tree present and linked, then the stage entries copied.
    pub fn ensure_model(
        &self,
        trees: &[String],
        stage: &[String],
        progress: Option<&dyn Fn(&str)>,
    ) -> JobResult<EnsureReport> {
        let t0 = now_s();
        let mut fetched = Vec::new();
        let mut staged = Vec::new();
        for entry in trees {
            if let Some(p) = progress {
                p(&format!("weights: {entry}"));
            }
            if self.ensure_tree(entry)? {
                fetched.push(entry.clone());
            }
        }
        for s in stage {
            if let Some(p) = progress {
                p(&format!("staging {s}"));
            }
            if self.stage(s)? {
                staged.push(s.clone());
            }
        }
        Ok(EnsureReport {
            stage_s: crate::py::round_to(now_s() - t0, 3),
            fetched,
            staged,
        })
    }

    /// One tree present under the link root; an object-store failure (a
    /// token without read on the weights bucket, a network error) is a
    /// staging failure, not an engine crash.
    pub fn ensure_tree(&self, entry: &str) -> JobResult<bool> {
        self.ensure_tree_inner(entry)
            .map_err(|e| match e.downcast::<JobError>() {
                Ok(j) => j,
                Err(e) => JobError::stage(format!("{entry}: {e:#}")),
            })
    }

    /// Present under the link root. Returns true when it fetched something.
    fn ensure_tree_inner(&self, entry: &str) -> anyhow::Result<bool> {
        let t = TreeEntry::parse(entry).map_err(|e| anyhow::anyhow!("ValueError: {e}"))?;
        let link = self.roots.link_root.join(&t.name);
        if t.kind == "volume" {
            let vol = self.volume_path(&t).expect("volume path");
            if exists(&vol) {
                self.link(&link, &vol)?;
                return Ok(false);
            }
            if let Some(dest) = t.source.strip_prefix("weights/") {
                // Not on the volume: the R2 weights store, by its manifest.
                let fetched = self.fetch_r2_tree(dest, &self.roots.stage_dir.join(&t.name))?;
                self.link(&link, &self.roots.stage_dir.join(&t.name))?;
                return Ok(fetched);
            }
            return Err(JobError::stage(format!(
                "{entry}: not on the volume at {}",
                vol.display()
            ))
            .into());
        }
        let dst = self.roots.stage_dir.join(&t.name);
        if t.kind == "r2" {
            let (bucket, prefix) = t.source.split_once('/').unwrap_or((&t.source, ""));
            let fetched = if bucket == self.weights_bucket && prefix.starts_with("weights/") {
                self.fetch_r2_tree(&prefix["weights/".len()..], &dst)?
            } else {
                self.fetch_r2_prefix(bucket, prefix, &dst)?
            };
            self.link(&link, &dst)?;
            return Ok(fetched);
        }
        if t.kind == "https" {
            let fetched = self.fetch_https(&format!("https://{}", t.source), &dst)?;
            self.link(&link, &dst)?;
            return Ok(fetched);
        }
        Err(JobError::stage(format!("unknown tree kind {}", t.kind)).into())
    }

    fn volume_path(&self, t: &TreeEntry) -> Option<PathBuf> {
        if t.kind != "volume" {
            return None;
        }
        if let Some(rest) = t.source.strip_prefix("weights/") {
            return Some(self.roots.volume_weights.join(rest));
        }
        if let Some(rest) = t.source.strip_prefix("h3-rs/") {
            return Some(self.roots.volume_extras.join(rest));
        }
        None
    }

    /// `link -> target`, never writing through an existing symlinked parent.
    fn link(&self, link: &Path, target: &Path) -> anyhow::Result<()> {
        if exists(link) || is_symlink(link) {
            if is_symlink(link) && realpath(link) == realpath(target) {
                return Ok(());
            }
            if is_symlink(link) {
                std::fs::remove_file(link)?;
            } else if link.is_dir() {
                // A real directory under the link root (parent of nested links):
                // only replace it when it is empty.
                if std::fs::read_dir(link)?.next().is_some() {
                    return Ok(());
                }
                std::fs::remove_dir(link)?;
            } else {
                std::fs::remove_file(link)?;
            }
        }
        let parent = link.parent().map(Path::to_path_buf).unwrap_or_default();
        // Refuse to create inside a symlink that points at the volume (add-only).
        let mut p = Some(parent.as_path());
        while let Some(q) = p {
            if q == self.roots.link_root {
                break;
            }
            if is_symlink(q) {
                let real = realpath(q);
                if under(&real, &self.roots.volume) {
                    // Already reachable through the parent link when the target
                    // is the same volume file; otherwise it cannot be linked.
                    if realpath(link) == realpath(target) {
                        return Ok(());
                    }
                    return Err(JobError::stage(format!(
                        "cannot link {}: its parent {} is a link into the volume",
                        link.display(),
                        q.display()
                    ))
                    .into());
                }
            }
            p = q.parent();
        }
        std::fs::create_dir_all(&parent)?;
        std::os::unix::fs::symlink(target, link)?;
        (self.log)(&format!("link {} -> {}", link.display(), target.display()));
        Ok(())
    }

    // ---- staging (make_boot.py `stage TREE [SUB...]`) ----------------------------

    /// `TREE[:SUB,SUB]`: copy onto the container disk and repoint the link.
    /// Returns true when it copied.
    pub fn stage(&self, spec: &str) -> JobResult<bool> {
        self.stage_inner(spec)
            .map_err(|e| match e.downcast::<JobError>() {
                Ok(j) => j,
                Err(e) => JobError::stage(format!("stage {spec}: {e:#}")),
            })
    }

    fn stage_inner(&self, spec: &str) -> anyhow::Result<bool> {
        let (tree, subs_s) = spec.split_once(':').unwrap_or((spec, ""));
        let mut subs: Vec<String> = subs_s
            .split(',')
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();
        let src = self.resolve_source_dir(tree);
        let dst = self.roots.stage_dir.join(tree);
        let link = self.roots.link_root.join(tree);
        let Some(src) = src else {
            return Err(
                JobError::stage(format!("stage {spec}: {tree} is not on the volume")).into(),
            );
        };
        if realpath(&src) == realpath(&dst) {
            return Ok(false); // already a staged copy (an R2 fetch)
        }
        if self.dry_run {
            self.link(&link, &src)?;
            return Ok(false);
        }
        let t0 = now_s();
        std::fs::create_dir_all(&dst)?;
        if subs.is_empty() {
            let mut names: Vec<String> = std::fs::read_dir(&src)?
                .flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| !n.starts_with(".."))
                .collect();
            names.sort();
            subs = names;
        } else {
            // The rest of the tree: symlinks back to the volume.
            for e in std::fs::read_dir(&src)?.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                let l = dst.join(&name);
                if subs.contains(&name) {
                    continue;
                }
                if !exists(&l) && !is_symlink(&l) {
                    std::os::unix::fs::symlink(e.path(), &l)?;
                }
            }
        }
        let mut copied = 0usize;
        let mut total = 0u64;
        let mut bad: Vec<String> = Vec::new();
        for sub in &subs {
            let s = src.join(sub);
            if !exists(&s) {
                (self.log)(&format!(
                    "stage {tree}: no {sub} in {} (skipped)",
                    src.display()
                ));
                continue;
            }
            let d = dst.join(sub);
            if is_symlink(&d) {
                std::fs::remove_file(&d)?;
            }
            let mut files = Vec::new();
            if s.is_file() {
                files.push(s.clone());
            } else {
                files_under(&s, &mut files);
            }
            let pairs: Vec<(PathBuf, PathBuf)> = files
                .iter()
                .map(|p| (p.clone(), dst.join(p.strip_prefix(&src).unwrap())))
                .collect();
            for (_, q) in &pairs {
                if let Some(parent) = q.parent() {
                    std::fs::create_dir_all(parent)?;
                }
            }
            copied += self.copy_parallel(&pairs)?;
            for (p, q) in &pairs {
                let size = std::fs::metadata(p)?.len();
                total += size;
                if !q.is_file() || std::fs::metadata(q).map(|m| m.len()).unwrap_or(u64::MAX) != size
                {
                    bad.push(p.strip_prefix(&src).unwrap().to_string_lossy().into_owned());
                }
            }
        }
        let dt = (now_s() - t0).max(1e-6);
        if !bad.is_empty() {
            let first: Vec<String> = bad
                .iter()
                .take(3)
                .map(|b| crate::py::py_str_repr(b))
                .collect();
            return Err(JobError::stage(format!(
                "stage {spec}: {} files wrong after the copy ([{}])",
                bad.len(),
                first.join(", ")
            ))
            .into());
        }
        (self.log)(&format!(
            "staged {tree} [{}]: {total} bytes in {dt:.1} s ({:.0} MB/s), {copied} files copied",
            subs.join(" "),
            total as f64 / dt / 1e6
        ));
        std::fs::write(dst.join(COMPLETE_MARKER), b"")?;
        self.link(&link, &dst)?;
        Ok(copied > 0)
    }

    fn resolve_source_dir(&self, tree: &str) -> Option<PathBuf> {
        let p = self.roots.volume_weights.join(tree);
        if p.is_dir() {
            return Some(p);
        }
        if let Some(rest) = tree.strip_prefix("h3-rs/") {
            let p = self.roots.volume_extras.join(rest);
            if p.is_dir() {
                return Some(p);
            }
        }
        // Already fetched to the stage dir (an r2:/https: tree).
        let p = self.roots.stage_dir.join(tree);
        if p.is_dir() {
            return Some(p);
        }
        None
    }

    /// `cp --update=none`: 8 streams; returns how many were missing before.
    fn copy_parallel(&self, pairs: &[(PathBuf, PathBuf)]) -> anyhow::Result<usize> {
        let same = |s: &Path, d: &Path| {
            d.is_file()
                && std::fs::metadata(d).map(|m| m.len()).ok()
                    == std::fs::metadata(s).map(|m| m.len()).ok()
        };
        let todo: Vec<(PathBuf, PathBuf)> =
            pairs.iter().filter(|(s, d)| !same(s, d)).cloned().collect();
        let n = todo.len();
        if n > 0 {
            let queue = Mutex::new(todo.into_iter().collect::<std::collections::VecDeque<_>>());
            let errors: Mutex<Vec<anyhow::Error>> = Mutex::new(Vec::new());
            std::thread::scope(|scope| {
                for _ in 0..STAGE_PARALLEL.min(n) {
                    scope.spawn(|| loop {
                        let Some((s, d)) = queue.lock().unwrap().pop_front() else {
                            break;
                        };
                        let r = std::process::Command::new("cp")
                            .arg("--update=none")
                            .arg(&s)
                            .arg(&d)
                            .output();
                        if !r.map(|o| o.status.success()).unwrap_or(false) {
                            // BusyBox / old coreutils: plain copy to a temp name then rename.
                            let tmp = crate::r2::part_path(&d);
                            if let Err(e) =
                                std::fs::copy(&s, &tmp).and_then(|_| std::fs::rename(&tmp, &d))
                            {
                                errors
                                    .lock()
                                    .unwrap()
                                    .push(anyhow::anyhow!("{}: {e}", s.display()));
                            }
                        }
                    });
                }
            });
            if let Some(e) = errors.into_inner().unwrap().into_iter().next() {
                return Err(e);
            }
        }
        Ok(n)
    }

    // ---- fetches -----------------------------------------------------------------

    /// fastvideo-rs's R2 layout: resolve `<rev>` from the manifest and fetch
    /// `weights/<dest>/<rev>/<path>` with size verification (sha256 skipped:
    /// R2 checked the bodies at publish time). `dest` may name a file or a
    /// directory inside a published tree (`auxiliary/tae/taew2_1.safetensors`):
    /// then only that part is fetched. Add-only, idempotent.
    fn fetch_r2_tree(&self, dest: &str, dst: &Path) -> anyhow::Result<bool> {
        let Some(store) = &self.store else {
            return Err(JobError::stage(format!(
                "{dest}: not on the volume and no R2 credentials to fetch it"
            ))
            .into());
        };
        if dst.is_file() || dst.join(COMPLETE_MARKER).is_file() {
            return Ok(false);
        }
        // The published tree: `dest` or the nearest ancestor with a manifest.
        let mut tree = dest.to_string();
        let mut only: Option<String> = None;
        let mut found = self.find_manifest(store.as_ref(), &tree)?;
        while found.is_none() && tree.contains('/') {
            tree = tree.rsplit_once('/').unwrap().0.to_string();
            only = Some(dest[tree.len() + 1..].to_string());
            found = self.find_manifest(store.as_ref(), &tree)?;
        }
        let Some((manifest, rev)) = found else {
            return Err(JobError::stage(format!(
                "{dest}: not on the volume and no manifest for it in the R2 weights store"
            ))
            .into());
        };
        let tree_dst: PathBuf = match &only {
            None => dst.to_path_buf(),
            Some(o) => {
                let s = dst.to_string_lossy();
                PathBuf::from(&s[..s.len() - o.len() - 1])
            }
        };
        let mut files: Vec<Value> = manifest["files"].as_array().cloned().unwrap_or_default();
        if let Some(o) = &only {
            let pre = format!("{}/", o.trim_end_matches('/'));
            files.retain(|f| {
                f["path"]
                    .as_str()
                    .map(|p| p == o || p.starts_with(&pre))
                    .unwrap_or(false)
            });
            if files.is_empty() {
                return Err(
                    JobError::stage(format!("{dest}: {o} is not in {tree}'s manifest")).into(),
                );
            }
        }
        (self.log)(&format!(
            "fetch {tree}@{} from R2: {} files",
            rev.chars().take(12).collect::<String>(),
            files.len()
        ));
        let prefix = format!("weights/{tree}/{rev}/");
        std::fs::create_dir_all(&tree_dst)?;
        let fetched = Mutex::new(0usize);
        let queue = Mutex::new(
            files
                .iter()
                .cloned()
                .collect::<std::collections::VecDeque<_>>(),
        );
        let errors: Mutex<Vec<anyhow::Error>> = Mutex::new(Vec::new());
        std::thread::scope(|scope| {
            for _ in 0..STAGE_PARALLEL.min(files.len().max(1)) {
                scope.spawn(|| loop {
                    let Some(f) = queue.lock().unwrap().pop_front() else {
                        break;
                    };
                    let path = f["path"].as_str().unwrap_or("").to_string();
                    let size = f["size"].as_f64().unwrap_or(-1.0) as i64;
                    let p = tree_dst.join(&path);
                    if p.is_file()
                        && std::fs::metadata(&p).map(|m| m.len() as i64).unwrap_or(-2) == size
                    {
                        continue;
                    }
                    match store.get_file(&self.weights_bucket, &format!("{prefix}{path}"), &p) {
                        Ok(got) if got as i64 == size => *fetched.lock().unwrap() += 1,
                        Ok(got) => errors.lock().unwrap().push(
                            JobError::stage(format!(
                                "{tree}/{path}: got {got} bytes, manifest says {size}"
                            ))
                            .into(),
                        ),
                        Err(e) => errors.lock().unwrap().push(e),
                    }
                });
            }
        });
        if let Some(e) = errors.into_inner().unwrap().into_iter().next() {
            return Err(e);
        }
        std::fs::write(tree_dst.join(REVISION_MARKER), format!("{rev}\n"))?;
        if only.is_some() {
            let marker = tree_dst.join(STUDIO_FILES_MARKER);
            let mut have: BTreeSet<String> = if marker.is_file() {
                std::fs::read_to_string(&marker)?
                    .split_whitespace()
                    .map(str::to_string)
                    .collect()
            } else {
                BTreeSet::new()
            };
            have.extend(
                files
                    .iter()
                    .filter_map(|f| f["path"].as_str().map(str::to_string)),
            );
            std::fs::write(
                &marker,
                format!("{}\n", have.into_iter().collect::<Vec<_>>().join("\n")),
            )?;
        } else {
            std::fs::write(
                tree_dst.join(COMPLETE_MARKER),
                format!(
                    "{}\n",
                    dumps(
                        &json!({"source": "r2", "dest": tree, "rev": rev}),
                        None,
                        true
                    )
                ),
            )?;
        }
        let n = *fetched.lock().unwrap();
        Ok(n > 0)
    }

    fn find_manifest(
        &self,
        store: &dyn ObjectStore,
        dest: &str,
    ) -> anyhow::Result<Option<(Value, String)>> {
        let prefix = format!("manifests/{dest}/");
        let keys: Vec<String> = store
            .list_keys(&self.weights_bucket, &prefix)?
            .into_iter()
            .map(|(k, _)| k)
            .filter(|k| k.ends_with(".json") && !k[prefix.len()..].contains('/'))
            .collect();
        if keys.is_empty() {
            return Ok(None);
        }
        let mut best: Option<(String, Value)> = None;
        for k in keys {
            let m: Value = serde_json::from_slice(&store.get_bytes(&self.weights_bucket, &k)?)?;
            let created = |v: &Value| {
                v.get("created_at")
                    .map(crate::py::py_str)
                    .unwrap_or_default()
            };
            if best
                .as_ref()
                .map(|(_, b)| created(&m) > created(b))
                .unwrap_or(true)
            {
                best = Some((k, m));
            }
        }
        let (k, m) = best.unwrap();
        let rev = m
            .get("rev")
            .filter(|r| crate::py::truthy(r))
            .map(crate::py::py_str)
            .unwrap_or_else(|| {
                k.rsplit('/')
                    .next()
                    .unwrap()
                    .trim_end_matches(".json")
                    .to_string()
            });
        Ok(Some((m, rev)))
    }

    /// A plain bucket prefix (no manifest): every object, size-checked
    /// against the listing.
    fn fetch_r2_prefix(&self, bucket: &str, prefix: &str, dst: &Path) -> anyhow::Result<bool> {
        let Some(store) = &self.store else {
            return Err(JobError::stage(format!(
                "r2:{bucket}/{prefix}: no R2 credentials to fetch it"
            ))
            .into());
        };
        if dst.join(COMPLETE_MARKER).is_file() {
            return Ok(false);
        }
        let base = prefix.trim_end_matches('/').to_string();
        let objs = store.list_keys(bucket, &format!("{base}/"))?;
        if objs.is_empty() {
            return Err(JobError::stage(format!("r2:{bucket}/{prefix}: nothing there")).into());
        }
        std::fs::create_dir_all(dst)?;
        let fetched = Mutex::new(0usize);
        let queue = Mutex::new(
            objs.iter()
                .cloned()
                .collect::<std::collections::VecDeque<_>>(),
        );
        let errors: Mutex<Vec<anyhow::Error>> = Mutex::new(Vec::new());
        std::thread::scope(|scope| {
            for _ in 0..STAGE_PARALLEL.min(objs.len()) {
                scope.spawn(|| loop {
                    let Some((k, n)) = queue.lock().unwrap().pop_front() else {
                        break;
                    };
                    let rel = &k[base.len() + 1..];
                    let p = dst.join(rel);
                    if p.is_file()
                        && std::fs::metadata(&p).map(|m| m.len()).unwrap_or(u64::MAX) == n
                    {
                        continue;
                    }
                    match store.get_file(bucket, &k, &p) {
                        Ok(got) if got == n => *fetched.lock().unwrap() += 1,
                        Ok(got) => errors.lock().unwrap().push(
                            JobError::stage(format!("{k}: got {got} bytes, listed {n}")).into(),
                        ),
                        Err(e) => errors.lock().unwrap().push(e),
                    }
                });
            }
        });
        if let Some(e) = errors.into_inner().unwrap().into_iter().next() {
            return Err(e);
        }
        std::fs::write(
            dst.join(COMPLETE_MARKER),
            format!(
                "{}\n",
                dumps(
                    &json!({"source": format!("r2:{bucket}/{prefix}")}),
                    None,
                    true
                )
            ),
        )?;
        let n = *fetched.lock().unwrap();
        Ok(n > 0)
    }

    /// One file; the size is checked against Content-Length.
    fn fetch_https(&self, url: &str, dst: &Path) -> anyhow::Result<bool> {
        if dst.is_file() {
            return Ok(false);
        }
        if let Some(p) = dst.parent() {
            std::fs::create_dir_all(p)?;
        }
        let tmp = crate::r2::part_path(dst);
        (self.log)(&format!("fetch {url}"));
        let (got, _, want) =
            crate::httpc::download(url, &tmp, 120.0, None).map_err(|e| anyhow::anyhow!("{e}"))?;
        if let Some(want) = want {
            if got != want {
                let _ = std::fs::remove_file(&tmp);
                return Err(JobError::stage(format!(
                    "{url}: got {got} bytes, Content-Length {want}"
                ))
                .into());
            }
        }
        std::fs::rename(&tmp, dst)?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r2::LocalStore;
    use crate::tempdir;

    fn roots(dir: &Path) -> Roots {
        Roots {
            volume: dir.join("vol"),
            volume_weights: dir.join("vol/weights"),
            volume_extras: dir.join("vol/h3-rs"),
            stage_dir: dir.join("stage"),
            link_root: dir.join("w"),
        }
    }

    /// fastvideo-rs's layout: weights/<dest>/<rev>/<path> + manifests/<dest>/<rev>.json.
    fn publish(store: &LocalStore, dest: &str, rev: &str, files: &[(&str, &[u8])], created: &str) {
        let tmp = store.root.join("_tmp");
        std::fs::create_dir_all(&tmp).unwrap();
        let mut recs = Vec::new();
        for (path, data) in files {
            let p = tmp.join(path.replace('/', "_"));
            std::fs::write(&p, data).unwrap();
            store
                .put_file(
                    "fv-weights",
                    &format!("weights/{dest}/{rev}/{path}"),
                    &p,
                    None,
                )
                .unwrap();
            recs.push(json!({"path": path, "size": data.len(), "sha256": "x"}));
        }
        let m = tmp.join("m.json");
        std::fs::write(
            &m,
            json!({"dest": dest, "rev": rev, "created_at": created, "files": recs}).to_string(),
        )
        .unwrap();
        store
            .put_file(
                "fv-weights",
                &format!("manifests/{dest}/{rev}.json"),
                &m,
                None,
            )
            .unwrap();
    }

    fn resolver(r: Roots, store: Option<Arc<dyn ObjectStore>>) -> WeightsResolver {
        WeightsResolver::new(r, store, crate::null_log(), false)
    }

    #[test]
    fn tree_entry_parsing() {
        assert_eq!(
            TreeEntry::parse("weights/h3-base").unwrap(),
            TreeEntry {
                kind: "volume".into(),
                source: "weights/h3-base".into(),
                name: "h3-base".into()
            }
        );
        assert_eq!(
            TreeEntry::parse("h3-rs/tables").unwrap(),
            TreeEntry {
                kind: "volume".into(),
                source: "h3-rs/tables".into(),
                name: "h3-rs/tables".into()
            }
        );
        assert_eq!(
            TreeEntry::parse("r2:fv-weights/weights/fastwan21-1.3b=fastwan").unwrap(),
            TreeEntry {
                kind: "r2".into(),
                source: "fv-weights/weights/fastwan21-1.3b".into(),
                name: "fastwan".into()
            }
        );
        assert_eq!(
            TreeEntry::parse(
                "https:weights.fastwan.dev/v1/x.safetensors=fastwan-public/x.safetensors"
            )
            .unwrap(),
            TreeEntry {
                kind: "https".into(),
                source: "weights.fastwan.dev/v1/x.safetensors".into(),
                name: "fastwan-public/x.safetensors".into()
            }
        );
        assert!(TreeEntry::parse("nope/x").is_err());
    }

    #[test]
    fn volume_trees_are_linked_not_copied() {
        let dir = tempdir("w-vol");
        let r = roots(&dir);
        std::fs::create_dir_all(r.volume_weights.join("h3-base/transformer")).unwrap();
        std::fs::create_dir_all(r.volume_extras.join("tables")).unwrap();
        std::fs::create_dir_all(r.volume_weights.join("auxiliary/tae")).unwrap();
        std::fs::write(
            r.volume_weights.join("auxiliary/tae/taeh3.safetensors"),
            b"t",
        )
        .unwrap();
        let w = resolver(r.clone(), None);
        assert!(!w.ensure_tree("weights/h3-base").unwrap());
        assert!(!w.ensure_tree("h3-rs/tables").unwrap());
        assert!(!w
            .ensure_tree("weights/auxiliary/tae/taeh3.safetensors")
            .unwrap());
        assert_eq!(
            realpath(&r.link_root.join("h3-base")),
            realpath(&r.volume_weights.join("h3-base"))
        );
        assert!(is_symlink(&r.link_root.join("h3-rs/tables")));
        assert_eq!(
            std::fs::read(r.link_root.join("auxiliary/tae/taeh3.safetensors")).unwrap(),
            b"t"
        );
        assert!(!is_symlink(&r.link_root.join("auxiliary"))); // a real parent dir, no write through a volume link
        let st = w.status("weights/h3-base").unwrap();
        assert!(st.volume && st.linked && !st.staged);
        let e = w.ensure_tree("h3-rs/nope").unwrap_err();
        assert!(e.where_ == "stage" && e.message.contains("not on the volume"));
    }

    #[test]
    fn missing_volume_tree_comes_from_the_r2_store_by_manifest() {
        let dir = tempdir("w-r2");
        let r = roots(&dir);
        std::fs::create_dir_all(&r.volume_weights).unwrap();
        let store = LocalStore::new(dir.join("r2"));
        publish(
            &store,
            "fastwan22-ti2v-5b",
            "3e18704",
            &[
                ("transformer/a.safetensors", &[b'A'; 10]),
                ("config.json", b"{}"),
            ],
            "2026-10-11T00:00:00Z",
        );
        publish(
            &store,
            "fastwan22-ti2v-5b",
            "0000000",
            &[("old.bin", b"o")],
            "2026-01-01T00:00:00Z",
        ); // an older revision loses
        let w = resolver(r.clone(), Some(Arc::new(store)));
        assert!(w.ensure_tree("weights/fastwan22-ti2v-5b").unwrap());
        let dst = r.stage_dir.join("fastwan22-ti2v-5b");
        assert_eq!(
            std::fs::read(dst.join("transformer/a.safetensors")).unwrap(),
            [b'A'; 10]
        );
        assert!(dst.join("config.json").is_file());
        assert_eq!(
            std::fs::read_to_string(dst.join(".fv-revision"))
                .unwrap()
                .trim(),
            "3e18704"
        );
        assert!(dst.join(COMPLETE_MARKER).is_file());
        assert_eq!(
            realpath(&r.link_root.join("fastwan22-ti2v-5b")),
            realpath(&dst)
        );
        assert!(!w.ensure_tree("weights/fastwan22-ti2v-5b").unwrap()); // idempotent
        let st = w.status("weights/fastwan22-ti2v-5b").unwrap();
        assert!(st.staged && st.linked && !st.volume);
    }

    #[test]
    fn file_inside_a_published_tree_fetches_only_that_file() {
        let dir = tempdir("w-file");
        let r = roots(&dir);
        std::fs::create_dir_all(&r.volume_weights).unwrap();
        let store = LocalStore::new(dir.join("r2"));
        publish(
            &store,
            "auxiliary/tae",
            "eu-20261011",
            &[
                ("taew2_1.safetensors", &[b'1'; 5]),
                ("taew2_2.safetensors", &[b'2'; 5]),
                ("taeh3.safetensors", b"3"),
            ],
            "2026-10-11T00:00:00Z",
        );
        let w = resolver(r.clone(), Some(Arc::new(store)));
        assert!(w
            .ensure_tree("weights/auxiliary/tae/taew2_1.safetensors")
            .unwrap());
        let tree = r.stage_dir.join("auxiliary/tae");
        assert_eq!(
            std::fs::read(tree.join("taew2_1.safetensors")).unwrap(),
            [b'1'; 5]
        );
        assert!(!tree.join("taeh3.safetensors").exists());
        assert!(!tree.join(COMPLETE_MARKER).exists());
        assert!(std::fs::read_to_string(tree.join(".studio-files"))
            .unwrap()
            .contains("taew2_1.safetensors"));
        assert_eq!(
            std::fs::read(r.link_root.join("auxiliary/tae/taew2_1.safetensors")).unwrap(),
            [b'1'; 5]
        );
        let e = w.ensure_tree("weights/nothing-here").unwrap_err();
        assert!(e.message.contains("no manifest"), "{}", e.message);
    }

    #[test]
    fn r2_prefix_entry_resolves_the_revision() {
        let dir = tempdir("w-prefix");
        let r = roots(&dir);
        let store = LocalStore::new(dir.join("r2"));
        publish(
            &store,
            "fastwan21-1.3b",
            "25e7ed7",
            &[
                ("transformer/x.safetensors", b"xxx"),
                ("tokenizer/tokenizer.json", b"{}"),
            ],
            "2026-10-11T00:00:00Z",
        );
        let w = resolver(r.clone(), Some(Arc::new(store)));
        assert!(w
            .ensure_tree("r2:fv-weights/weights/fastwan21-1.3b=fastwan")
            .unwrap());
        assert_eq!(
            std::fs::read(r.link_root.join("fastwan/transformer/x.safetensors")).unwrap(),
            b"xxx"
        );
        assert_eq!(
            std::fs::read_to_string(r.stage_dir.join("fastwan/.fv-revision"))
                .unwrap()
                .trim(),
            "25e7ed7"
        );
    }

    #[test]
    fn size_mismatch_is_refused() {
        let dir = tempdir("w-size");
        let r = roots(&dir);
        let store = LocalStore::new(dir.join("r2"));
        publish(
            &store,
            "t",
            "r1",
            &[("a.bin", b"abc")],
            "2026-10-11T00:00:00Z",
        );
        let mp = store.root.join("fv-weights/manifests/t/r1.json");
        let mut m: Value = serde_json::from_slice(&std::fs::read(&mp).unwrap()).unwrap();
        m["files"][0]["size"] = json!(99);
        std::fs::write(&mp, m.to_string()).unwrap();
        let w = resolver(r, Some(Arc::new(store)));
        let e = w.ensure_tree("weights/t").unwrap_err();
        assert!(e.message.contains("manifest says 99"), "{}", e.message);
    }

    #[test]
    fn stage_copies_subdirs_and_links_the_rest() {
        let dir = tempdir("w-stage");
        let r = roots(&dir);
        let src = r.volume_weights.join("h3-8step");
        for (sub, files) in [
            ("transformer", vec!["a.safetensors", "b.safetensors"]),
            ("text_encoder", vec!["m.safetensors"]),
            ("vae", vec!["v.safetensors"]),
        ] {
            std::fs::create_dir_all(src.join(sub)).unwrap();
            for f in files {
                std::fs::write(
                    src.join(sub).join(f),
                    (0..64u8).map(|i| i.wrapping_mul(37)).collect::<Vec<_>>(),
                )
                .unwrap();
            }
        }
        std::fs::write(src.join("fastvideo_inference.json"), "{}").unwrap();
        let w = resolver(r.clone(), None);
        w.ensure_tree("weights/h3-8step").unwrap();
        assert!(w.stage("h3-8step:text_encoder,vae").unwrap());
        let dst = r.stage_dir.join("h3-8step");
        assert!(
            dst.join("text_encoder/m.safetensors").is_file()
                && !is_symlink(&dst.join("text_encoder"))
        );
        assert_eq!(
            std::fs::read(dst.join("vae/v.safetensors")).unwrap(),
            std::fs::read(src.join("vae/v.safetensors")).unwrap()
        );
        assert!(
            is_symlink(&dst.join("transformer"))
                && realpath(&dst.join("transformer")) == realpath(&src.join("transformer"))
        );
        assert!(is_symlink(&dst.join("fastvideo_inference.json")));
        assert_eq!(realpath(&r.link_root.join("h3-8step")), realpath(&dst));
        assert!(dst.join(COMPLETE_MARKER).is_file());
        assert!(!w.stage("h3-8step:text_encoder,vae").unwrap()); // nothing to copy the second time
        assert!(w.stage("h3-8step").unwrap()); // a whole tree
        assert!(
            !is_symlink(&dst.join("transformer"))
                && dst.join("transformer/b.safetensors").is_file()
        );
        let mut names: Vec<String> = std::fs::read_dir(&src)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        assert_eq!(
            names,
            [
                "fastvideo_inference.json",
                "text_encoder",
                "transformer",
                "vae"
            ]
        ); // the volume was never written
    }

    #[test]
    fn stage_skips_a_missing_sub_and_refuses_a_missing_tree() {
        let dir = tempdir("w-skip");
        let r = roots(&dir);
        std::fs::create_dir_all(r.volume_weights.join("h3-base/vae")).unwrap();
        std::fs::write(r.volume_weights.join("h3-base/vae/v"), b"v").unwrap();
        let notes = Arc::new(Mutex::new(Vec::new()));
        let n = notes.clone();
        let w = WeightsResolver::new(
            r,
            None,
            Arc::new(move |s: &str| n.lock().unwrap().push(s.to_string())),
            false,
        );
        w.ensure_tree("weights/h3-base").unwrap();
        assert!(w.stage("h3-base:nope,vae").unwrap());
        assert!(notes.lock().unwrap().iter().any(|n| n.contains("no nope")));
        let e = w.stage("absent").unwrap_err();
        assert!(e.message.contains("not on the volume"));
    }

    #[test]
    fn ensure_model_reports_stage_time_and_dry_run_links() {
        let dir = tempdir("w-model");
        let r = roots(&dir);
        std::fs::create_dir_all(r.volume_weights.join("fastwan")).unwrap();
        let w = resolver(r.clone(), None);
        let out = w
            .ensure_model(&["weights/fastwan".to_string()], &[], None)
            .unwrap();
        assert!(out.fetched.is_empty() && out.staged.is_empty() && out.stage_s >= 0.0);
        let dry = WeightsResolver {
            dry_run: true,
            ..resolver(r.clone(), None)
        };
        assert!(!dry.stage("fastwan").unwrap());
        assert!(!r.stage_dir.join("fastwan").exists() && is_symlink(&r.link_root.join("fastwan")));
    }

    struct Denied;
    impl ObjectStore for Denied {
        fn put_file(&self, _: &str, _: &str, _: &Path, _: Option<&str>) -> anyhow::Result<()> {
            anyhow::bail!("AccessDenied")
        }
        fn get_file(&self, _: &str, _: &str, _: &Path) -> anyhow::Result<u64> {
            anyhow::bail!("AccessDenied")
        }
        fn get_bytes(&self, _: &str, _: &str) -> anyhow::Result<Vec<u8>> {
            anyhow::bail!("AccessDenied")
        }
        fn list_keys(&self, _: &str, _: &str) -> anyhow::Result<Vec<(String, u64)>> {
            anyhow::bail!("AccessDenied: ListObjectsV2")
        }
        fn head_size(&self, _: &str, _: &str) -> anyhow::Result<Option<u64>> {
            anyhow::bail!("AccessDenied")
        }
        fn presign_get(&self, _: &str, _: &str, _: u64) -> String {
            String::new()
        }
    }

    #[test]
    fn object_store_failure_is_a_stage_error() {
        let dir = tempdir("w-denied");
        let w = resolver(roots(&dir), Some(Arc::new(Denied)));
        let e = w
            .ensure_tree("r2:fv-weights/weights/fastwan21-1.3b=fastwan")
            .unwrap_err();
        assert!(
            e.where_ == "stage" && e.message.contains("AccessDenied"),
            "{}",
            e.message
        );
    }
}
