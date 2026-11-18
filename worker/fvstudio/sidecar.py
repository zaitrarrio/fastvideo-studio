"""Resident engine processes (`h3 serve`, `fastwan serve`, `fv-serve`) on
127.0.0.1: start, wait for ready, keep the last log lines, stop. One
sidecar per engine, keyed by its argv so a job for another model restarts
that engine's process. Before a start the free VRAM (nvidia-smi) is checked
against the model's `vram_gib` and the other engines are stopped when it
does not fit (design §1)."""

from __future__ import annotations

import collections
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from . import gpu, httpc
from .errors import JobError

Log = Callable[[str], None]
Probe = Callable[[], tuple[bool, str | None]]  # (ready, fatal error)
LOG_KEEP = 400
TAIL = 50


def http_info_probe(url: str) -> Probe:
    """`h3 serve` / `fastwan serve`: `/api/info` → ready / error."""

    def probe() -> tuple[bool, str | None]:
        try:
            info = httpc.get_json(url, timeout=5)
        except Exception:  # noqa: BLE001
            return False, None
        if info.get("error"):
            return False, str(info["error"])
        return bool(info.get("ready")), None

    return probe


def ping_probe(url: str) -> Probe:
    """fv-serve `/ping`: 204 loading, 200 ready, 503 failed."""

    def probe() -> tuple[bool, str | None]:
        try:
            status, body = httpc.get_status(url, timeout=5)
        except Exception:  # noqa: BLE001
            return False, None
        if status == 200:
            return True, None
        if status == 503:
            return False, body.decode("utf-8", "replace")[:300] or "fv-serve reports 503"
        return False, None

    return probe


@dataclass
class Sidecar:
    engine: str
    key: str  # what it was started for (model / recipe); a different key restarts
    argv: list[str]
    env: dict[str, str]
    probe: Probe
    log: Log = field(default=lambda s: None)
    proc: subprocess.Popen | None = None
    lines: collections.deque = field(default_factory=lambda: collections.deque(maxlen=LOG_KEEP))
    started_at: float = 0.0
    ready_at: float | None = None
    _reader: threading.Thread | None = None

    def start(self) -> None:
        self.log(f"[{self.engine}] start: {' '.join(self.argv)}")
        self.proc = subprocess.Popen(
            self.argv,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )
        self.started_at = time.time()
        self.ready_at = None
        self._reader = threading.Thread(target=self._pump, name=f"{self.engine}-log", daemon=True)
        self._reader.start()

    def _pump(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            self.lines.append(line)
            self.log(f"[{self.engine}] {line}")

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def tail(self, n: int = TAIL) -> list[str]:
        return list(self.lines)[-n:]

    def wait_ready(self, timeout_s: float, progress: Callable[[float], None] | None = None) -> float:
        """Seconds until ready; `JobError("engine")` with the log tail on a
        crash, a reported error or the timeout."""
        t0 = time.time()
        last = 0.0
        while True:
            if not self.alive():
                code = self.proc.returncode if self.proc else None
                raise JobError("engine", f"{self.engine} sidecar exited with code {code} while loading", self.tail())
            ready, err = self.probe()
            if err:
                self.stop()
                raise JobError("engine", f"{self.engine} failed to start: {err}", self.tail())
            if ready:
                self.ready_at = time.time()
                return self.ready_at - t0
            now = time.time()
            if now - t0 > timeout_s:
                self.stop()
                raise JobError("engine", f"{self.engine} not ready after {timeout_s:.0f} s", self.tail())
            if progress and now - last >= 5:
                last = now
                progress(now - t0)
            time.sleep(0.5)

    def stop(self, grace_s: float = 15.0) -> None:
        p = self.proc
        if p is None:
            return
        if p.poll() is None:
            self.log(f"[{self.engine}] stop (pid {p.pid})")
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                p.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait(timeout=10)
        self.proc = None


@dataclass
class SidecarManager:
    log: Log = field(default=lambda s: None)
    sidecars: dict[str, Sidecar] = field(default_factory=dict)
    load_timeout_s: float = 1800.0
    vram_margin_gib: float = 1.0
    gpu_query: Callable[[], gpu.GpuInfo | None] = gpu.query

    def resident(self) -> dict[str, dict]:
        return {
            e: {"key": s.key, "alive": s.alive(), "pid": s.proc.pid if s.proc else None,
                "ready_s": None if s.ready_at is None else round(s.ready_at - s.started_at, 1)}
            for e, s in self.sidecars.items()
        }

    def get(self, engine: str) -> Sidecar | None:
        return self.sidecars.get(engine)

    def ensure(self, engine: str, key: str, make: Callable[[], Sidecar], vram_gib: float = 0.0,
               progress: Callable[[str, float], None] | None = None) -> tuple[Sidecar, float]:
        """The engine's sidecar, started for `key` (restarted when it was
        running for another key or died). Returns it and the load seconds (0
        when it was already up)."""
        cur = self.sidecars.get(engine)
        if cur is not None and cur.alive() and cur.key == key:
            ready, err = cur.probe()
            if ready and not err:
                return cur, 0.0
            if err:
                self.log(f"[{engine}] resident sidecar reports an error, restarting: {err}")
        if cur is not None:
            if not cur.alive() and cur.proc is not None:
                self.log(f"[{engine}] sidecar had exited (code {cur.proc.returncode}); last lines: {' | '.join(cur.tail(5))}")
            cur.stop()
            del self.sidecars[engine]
        self.make_room(engine, vram_gib)
        sc = make()
        sc.log = self.log
        if progress:
            progress("loading", 0.0)
        sc.start()
        self.sidecars[engine] = sc
        load_s = sc.wait_ready(self.load_timeout_s, (lambda t: progress("loading", t)) if progress else None)
        return sc, load_s

    def make_room(self, engine: str, vram_gib: float) -> None:
        if vram_gib <= 0:
            return
        g = self.gpu_query()
        if g is None:
            return
        if g.free_gib >= vram_gib + self.vram_margin_gib:
            return
        others = [e for e in self.sidecars if e != engine]
        if not others:
            self.log(f"[{engine}] only {g.free_gib:.1f} GiB free for a {vram_gib:.0f} GiB model and nothing else to stop")
            return
        self.log(f"[{engine}] {g.free_gib:.1f} GiB free < {vram_gib:.0f} GiB: stopping {', '.join(others)}")
        for e in others:
            self.sidecars[e].stop()
            del self.sidecars[e]
        for _ in range(20):
            g = self.gpu_query()
            if g is None or g.free_gib >= vram_gib + self.vram_margin_gib:
                return
            time.sleep(0.5)
        g = self.gpu_query()
        if g is not None:
            self.log(f"[{engine}] still only {g.free_gib:.1f} GiB free; starting anyway")

    def stop_all(self) -> None:
        for e in list(self.sidecars):
            self.sidecars[e].stop()
            del self.sidecars[e]
