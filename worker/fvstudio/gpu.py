"""nvidia-smi queries. Everything degrades to `None` without a GPU."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuInfo:
    name: str
    driver: str
    memory_total_mib: int
    memory_free_mib: int
    compute_cap: str | None = None

    @property
    def free_gib(self) -> float:
        return self.memory_free_mib / 1024.0

    @property
    def sm(self) -> int | None:
        if not self.compute_cap:
            return None
        try:
            major, minor = self.compute_cap.split(".")
            return int(major) * 10 + int(minor)
        except ValueError:
            return None


def query(index: int = 0) -> GpuInfo | None:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    fields = "name,driver_version,memory.total,memory.free"
    r = subprocess.run([smi, f"--id={index}", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=20)
    if r.returncode != 0 or not r.stdout.strip():
        return None
    parts = [p.strip() for p in r.stdout.strip().splitlines()[0].split(",")]
    if len(parts) < 4:
        return None
    cap = None
    r2 = subprocess.run([smi, f"--id={index}", "--query-gpu=compute_cap", "--format=csv,noheader"],
                        capture_output=True, text=True, timeout=20)
    if r2.returncode == 0 and r2.stdout.strip():
        cap = r2.stdout.strip().splitlines()[0].strip()
    try:
        return GpuInfo(parts[0], parts[1], int(float(parts[2])), int(float(parts[3])), cap)
    except ValueError:
        return None
