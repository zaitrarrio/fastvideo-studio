"""Weights resolver: volume → stage dir → R2 / https fetch, and the symlink
root `/workspace/w` every recipe is written against.

Catalog `trees` entries (catalog.json `trees.note`):
- `weights/<tree>` and `h3-rs/<sub>`: paths relative to the volume
  (`/runpod-volume/weights/<tree>`, `/runpod-volume/h3-rs/<sub>`). A path may
  name a file inside a tree (`weights/auxiliary/tae/taeh3.safetensors`).
  Missing from the volume, a `weights/<tree>` is fetched from the R2 weights
  store (`fv-weights`, `weights/<dest>/<rev>/…` + `manifests/<dest>/<rev>.json`).
- `r2:<bucket>/<prefix>=<name>`: fetched to the stage dir on first use. A
  prefix of the weights store resolves its `<rev>` through the manifest.
- `https:<host>/<path>=<name>`: one file, fetched on first use.

`stage` entries follow h3-rs `scripts/runpod/make_boot.py`'s `stage TREE
[SUB...]`: copy the tree (or only SUB directories, the rest symlinked back
to the volume) onto the container disk with 8 parallel `cp --update=none`,
verify every file's size, then point `/workspace/w/TREE` at the copy.

The volume is add-only and this module never writes under it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .errors import JobError
from .r2 import ObjectStore

STAGE_PARALLEL = 8
COMPLETE_MARKER = ".complete"
REVISION_MARKER = ".fv-revision"
STUDIO_FILES_MARKER = ".studio-files"  # files this worker fetched (subset fetches)

Log = Callable[[str], None]


@dataclass(frozen=True)
class Roots:
    volume: Path = Path("/runpod-volume")
    volume_weights: Path = Path("/runpod-volume/weights")
    volume_extras: Path = Path("/runpod-volume/h3-rs")
    stage_dir: Path = Path("/workspace/stage")
    link_root: Path = Path("/workspace/w")

    @staticmethod
    def from_catalog(trees: dict, env: dict | None = None) -> "Roots":
        e = os.environ if env is None else env
        return Roots(
            volume=Path(e.get("STUDIO_VOLUME", trees.get("volume", "/runpod-volume"))),
            volume_weights=Path(e.get("STUDIO_VOLUME_WEIGHTS", trees.get("volume_weights", "/runpod-volume/weights"))),
            volume_extras=Path(e.get("STUDIO_VOLUME_EXTRAS", trees.get("volume_extras", "/runpod-volume/h3-rs"))),
            stage_dir=Path(e.get("STUDIO_STAGE_DIR", trees.get("stage_dir", "/workspace/stage"))),
            link_root=Path(e.get("STUDIO_LINK_ROOT", trees.get("link_root", "/workspace/w"))),
        )


@dataclass(frozen=True)
class TreeEntry:
    """A parsed catalog `trees` entry."""

    kind: str  # volume | r2 | https
    source: str  # volume-relative path, `bucket/prefix`, or `host/path`
    name: str  # the path under the link root

    @staticmethod
    def parse(entry: str) -> "TreeEntry":
        if entry.startswith("r2:"):
            src, name = entry[3:].split("=", 1)
            return TreeEntry("r2", src.strip("/"), name.strip("/"))
        if entry.startswith("https:"):
            src, name = entry[6:].split("=", 1)
            return TreeEntry("https", src.strip("/"), name.strip("/"))
        if entry.startswith("weights/"):
            return TreeEntry("volume", entry, entry[len("weights/"):])
        if entry.startswith("h3-rs/"):
            return TreeEntry("volume", entry, entry)
        raise ValueError(f"unknown tree entry {entry!r}")


@dataclass
class TreeStatus:
    entry: str
    name: str
    volume: bool = False
    staged: bool = False
    fetched: bool = False
    linked: bool = False
    note: str = ""

    def to_json(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class WeightsResolver:
    roots: Roots
    store: ObjectStore | None = None
    weights_bucket: str = "fv-weights"
    log: Log = field(default=lambda s: None)
    dry_run: bool = False  # tests: never copy

    # ---- status ------------------------------------------------------------

    def status(self, entry: str) -> TreeStatus:
        t = TreeEntry.parse(entry)
        st = TreeStatus(entry=entry, name=t.name)
        vol = self._volume_path(t)
        st.volume = vol is not None and vol.exists()
        staged = self.roots.stage_dir / t.name
        st.staged = staged.exists() and (staged / COMPLETE_MARKER).exists() if staged.is_dir() else staged.is_file()
        st.fetched = staged.exists() and t.kind != "volume"
        link = self.roots.link_root / t.name
        st.linked = link.exists()
        return st

    # ---- ensure --------------------------------------------------------------

    def ensure_model(self, trees: tuple[str, ...], stage: tuple[str, ...], progress: Callable[[str], None] | None = None) -> dict:
        """Every tree present and linked, then the stage entries copied.
        Returns `{"stage_s": float, "fetched": [...], "staged": [...]}`."""
        t0 = time.time()
        fetched: list[str] = []
        staged: list[str] = []
        for entry in trees:
            if progress:
                progress(f"weights: {entry}")
            if self.ensure_tree(entry):
                fetched.append(entry)
        for s in stage:
            if progress:
                progress(f"staging {s}")
            if self.stage(s):
                staged.append(s)
        return {"stage_s": round(time.time() - t0, 3), "fetched": fetched, "staged": staged}

    def ensure_tree(self, entry: str) -> bool:
        """One tree present under the link root; an object-store failure (a
        token without read on the weights bucket, a network error) is a
        staging failure, not an engine crash."""
        try:
            return self._ensure_tree(entry)
        except JobError:
            raise
        except Exception as e:  # noqa: BLE001
            raise JobError("stage", f"{entry}: {type(e).__name__}: {e}") from None

    def _ensure_tree(self, entry: str) -> bool:
        """Present under the link root. Returns True when it fetched something."""
        t = TreeEntry.parse(entry)
        link = self.roots.link_root / t.name
        if t.kind == "volume":
            vol = self._volume_path(t)
            assert vol is not None
            if vol.exists():
                self._link(link, vol)
                return False
            if t.source.startswith("weights/"):
                # Not on the volume: the R2 weights store, by its manifest.
                fetched = self._fetch_r2_tree(t.source[len("weights/"):], self.roots.stage_dir / t.name)
                self._link(link, self.roots.stage_dir / t.name)
                return fetched
            raise JobError("stage", f"{entry}: not on the volume at {vol}")
        dst = self.roots.stage_dir / t.name
        if t.kind == "r2":
            bucket, _, prefix = t.source.partition("/")
            if bucket == self.weights_bucket and prefix.startswith("weights/"):
                fetched = self._fetch_r2_tree(prefix[len("weights/"):], dst)
            else:
                fetched = self._fetch_r2_prefix(bucket, prefix, dst)
            self._link(link, dst)
            return fetched
        if t.kind == "https":
            fetched = self._fetch_https("https://" + t.source, dst)
            self._link(link, dst)
            return fetched
        raise JobError("stage", f"unknown tree kind {t.kind}")

    def _volume_path(self, t: TreeEntry) -> Path | None:
        if t.kind != "volume":
            return None
        if t.source.startswith("weights/"):
            return self.roots.volume_weights / t.source[len("weights/"):]
        if t.source.startswith("h3-rs/"):
            return self.roots.volume_extras / t.source[len("h3-rs/"):]
        return None

    def _link(self, link: Path, target: Path) -> None:
        """`link -> target`, never writing through an existing symlinked parent."""
        if link.exists() or link.is_symlink():
            try:
                if link.is_symlink() and os.path.realpath(link) == os.path.realpath(target):
                    return
            except OSError:
                pass
            if link.is_symlink():
                link.unlink()
            elif link.is_dir():
                # A real directory under the link root (parent of nested links):
                # only replace it when it is empty.
                if any(link.iterdir()):
                    return
                link.rmdir()
            else:
                link.unlink()
        parent = link.parent
        # Refuse to create inside a symlink that points at the volume (add-only).
        for p in [parent, *parent.parents]:
            if p == self.roots.link_root:
                break
            if p.is_symlink():
                real = Path(os.path.realpath(p))
                if self._under(real, self.roots.volume):
                    # Already reachable through the parent link when the target
                    # is the same volume file; otherwise it cannot be linked.
                    if os.path.realpath(link) == os.path.realpath(target):
                        return
                    raise JobError("stage", f"cannot link {link}: its parent {p} is a link into the volume")
        parent.mkdir(parents=True, exist_ok=True)
        os.symlink(target, link)
        self.log(f"link {link} -> {target}")

    @staticmethod
    def _under(p: Path, root: Path) -> bool:
        try:
            p.relative_to(root)
            return True
        except ValueError:
            return False

    # ---- staging (make_boot.py `stage TREE [SUB...]`) ----------------------------

    def stage(self, spec: str) -> bool:
        """`TREE[:SUB,SUB]`: copy onto the container disk and repoint the link.
        Returns True when it copied."""
        tree, _, subs_s = spec.partition(":")
        subs = [s for s in subs_s.split(",") if s]
        src = self._resolve_source_dir(tree)
        dst = self.roots.stage_dir / tree
        link = self.roots.link_root / tree
        if src is None:
            raise JobError("stage", f"stage {spec}: {tree} is not on the volume")
        if os.path.realpath(src) == os.path.realpath(dst):
            return False  # already a staged copy (an R2 fetch)
        if self.dry_run:
            self._link(link, src)
            return False
        t0 = time.time()
        dst.mkdir(parents=True, exist_ok=True)
        if not subs:
            subs = sorted(e.name for e in src.iterdir() if not e.name.startswith(".."))
        else:
            # The rest of the tree: symlinks back to the volume.
            for e in src.iterdir():
                l = dst / e.name
                if e.name in subs:
                    continue
                if not l.exists() and not l.is_symlink():
                    os.symlink(e, l)
        copied = 0
        total = 0
        bad: list[str] = []
        for sub in subs:
            s = src / sub
            if not s.exists():
                self.log(f"stage {tree}: no {sub} in {src} (skipped)")
                continue
            d = dst / sub
            if d.is_symlink():
                d.unlink()
            files = [s] if s.is_file() else [p for p in s.rglob("*") if p.is_file()]
            for p in files:
                rel = p.relative_to(src)
                (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            copied += self._copy_parallel([(p, dst / p.relative_to(src)) for p in files])
            for p in files:
                size = p.stat().st_size
                total += size
                q = dst / p.relative_to(src)
                if not q.is_file() or q.stat().st_size != size:
                    bad.append(str(p.relative_to(src)))
        dt = max(time.time() - t0, 1e-6)
        if bad:
            raise JobError("stage", f"stage {spec}: {len(bad)} files wrong after the copy ({bad[:3]})")
        self.log(f"staged {tree} [{' '.join(subs)}]: {total} bytes in {dt:.1f} s ({total / dt / 1e6:.0f} MB/s), {copied} files copied")
        (dst / COMPLETE_MARKER).touch()
        self._link(link, dst)
        return copied > 0

    def _resolve_source_dir(self, tree: str) -> Path | None:
        for base in (self.roots.volume_weights, self.roots.volume_extras / tree if tree.startswith("h3-rs/") else None):
            if base is None:
                continue
            p = base / tree if base == self.roots.volume_weights else self.roots.volume_extras / tree[len("h3-rs/"):]
            if p.is_dir():
                return p
        # Already fetched to the stage dir (an r2:/https: tree).
        p = self.roots.stage_dir / tree
        if p.is_dir():
            return p
        return None

    def _copy_parallel(self, pairs: list[tuple[Path, Path]]) -> int:
        """`cp --update=none`: 8 streams; returns how many were missing before."""
        todo = [(s, d) for s, d in pairs if not (d.is_file() and d.stat().st_size == s.stat().st_size)]

        def one(sd: tuple[Path, Path]) -> None:
            s, d = sd
            r = subprocess.run(["cp", "--update=none", str(s), str(d)], capture_output=True, text=True)
            if r.returncode != 0:
                # BusyBox / old coreutils: plain copy to a temp name then rename.
                tmp = d.with_name(d.name + ".part")
                shutil.copyfile(s, tmp)
                os.replace(tmp, d)

        if todo:
            with ThreadPoolExecutor(max_workers=STAGE_PARALLEL) as ex:
                list(ex.map(one, todo))
        return len(todo)

    # ---- fetches -----------------------------------------------------------------

    def _fetch_r2_tree(self, dest: str, dst: Path) -> bool:
        """fastvideo-rs's R2 layout: resolve `<rev>` from the manifest and fetch
        `weights/<dest>/<rev>/<path>` with size verification (sha256 skipped:
        R2 checked the bodies at publish time). `dest` may name a file or a
        directory inside a published tree (`auxiliary/tae/taew2_1.safetensors`):
        then only that part is fetched. Add-only, idempotent."""
        if self.store is None:
            raise JobError("stage", f"{dest}: not on the volume and no R2 credentials to fetch it")
        if dst.is_file() or (dst / COMPLETE_MARKER).is_file():
            return False
        # The published tree: `dest` or the nearest ancestor with a manifest.
        tree, only = dest, None
        manifest, rev = self._find_manifest(tree)
        while manifest is None and "/" in tree:
            tree = tree.rsplit("/", 1)[0]
            only = dest[len(tree) + 1:]
            manifest, rev = self._find_manifest(tree)
        if manifest is None:
            raise JobError("stage", f"{dest}: not on the volume and no manifest for it in the R2 weights store")
        tree_dst = dst if only is None else Path(str(dst)[: -len(only) - 1])
        files = manifest.get("files", [])
        if only:
            files = [f for f in files if f["path"] == only or f["path"].startswith(only.rstrip("/") + "/")]
            if not files:
                raise JobError("stage", f"{dest}: {only} is not in {tree}'s manifest")
        self.log(f"fetch {tree}@{rev[:12]} from R2: {len(files)} files")
        prefix = f"weights/{tree}/{rev}/"
        fetched = 0

        def one(f: dict) -> None:
            nonlocal fetched
            p = tree_dst / f["path"]
            if p.is_file() and p.stat().st_size == int(f["size"]):
                return
            got = self.store.get_file(self.weights_bucket, prefix + f["path"], p)  # type: ignore[union-attr]
            if got != int(f["size"]):
                raise JobError("stage", f"{tree}/{f['path']}: got {got} bytes, manifest says {f['size']}")
            fetched += 1

        tree_dst.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=STAGE_PARALLEL) as ex:
            list(ex.map(one, files))
        (tree_dst / REVISION_MARKER).write_text(rev + "\n")
        if only:
            marker = tree_dst / STUDIO_FILES_MARKER
            have = set(marker.read_text().split()) if marker.is_file() else set()
            have.update(f["path"] for f in files)
            marker.write_text("\n".join(sorted(have)) + "\n")
        else:
            (tree_dst / COMPLETE_MARKER).write_text(json.dumps({"source": "r2", "dest": tree, "rev": rev}) + "\n")
        return fetched > 0

    def _find_manifest(self, dest: str) -> tuple[dict | None, str]:
        assert self.store is not None
        keys = [k for k, _ in self.store.list_keys(self.weights_bucket, f"manifests/{dest}/") if k.endswith(".json")]
        keys = [k for k in keys if k[len(f"manifests/{dest}/"):].count("/") == 0]
        if not keys:
            return None, ""
        best: tuple[str, dict] | None = None
        for k in keys:
            m = json.loads(self.store.get_bytes(self.weights_bucket, k).decode("utf-8"))
            if best is None or str(m.get("created_at", "")) > str(best[1].get("created_at", "")):
                best = (k, m)
        assert best is not None
        rev = best[1].get("rev") or best[0].rsplit("/", 1)[1][: -len(".json")]
        return best[1], rev

    def _fetch_r2_prefix(self, bucket: str, prefix: str, dst: Path) -> bool:
        """A plain bucket prefix (no manifest): every object, size-checked
        against the listing."""
        if self.store is None:
            raise JobError("stage", f"r2:{bucket}/{prefix}: no R2 credentials to fetch it")
        if (dst / COMPLETE_MARKER).is_file():
            return False
        objs = [(k, n) for k, n in self.store.list_keys(bucket, prefix.rstrip("/") + "/")]
        if not objs:
            raise JobError("stage", f"r2:{bucket}/{prefix}: nothing there")
        fetched = 0

        def one(kn: tuple[str, int]) -> None:
            nonlocal fetched
            k, n = kn
            rel = k[len(prefix.rstrip("/")) + 1:]
            p = dst / rel
            if p.is_file() and p.stat().st_size == n:
                return
            got = self.store.get_file(bucket, k, p)  # type: ignore[union-attr]
            if got != n:
                raise JobError("stage", f"{k}: got {got} bytes, listed {n}")
            fetched += 1

        dst.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=STAGE_PARALLEL) as ex:
            list(ex.map(one, objs))
        (dst / COMPLETE_MARKER).write_text(json.dumps({"source": f"r2:{bucket}/{prefix}"}) + "\n")
        return fetched > 0

    def _fetch_https(self, url: str, dst: Path) -> bool:
        """One file; the size is checked against Content-Length."""
        if dst.is_file():
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        self.log(f"fetch {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "fastvideo-studio-worker/1"})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            want = r.headers.get("Content-Length")
            shutil.copyfileobj(r, f, 8 * 1024 * 1024)
        if want is not None and tmp.stat().st_size != int(want):
            tmp.unlink(missing_ok=True)
            raise JobError("stage", f"{url}: got {tmp.stat().st_size if tmp.exists() else 0} bytes, Content-Length {want}")
        os.replace(tmp, dst)
        return True
