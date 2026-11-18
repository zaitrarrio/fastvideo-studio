#!/usr/bin/env python3
"""RunPod serverless handler for fastvideo-studio (docs/design.md §1,
docs/jobs.md). One worker process, concurrency 1, three engines as
sidecars; `make_handler()` is what the tests drive with a fake `runpod`."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fvstudio.worker import Worker  # noqa: E402


def make_handler(worker: Worker | None = None, progress_update=None):
    w = worker or Worker.from_env()

    def handler(job: dict) -> dict:
        def progress(rec: dict) -> None:
            if progress_update is not None:
                progress_update(job, rec)

        return w.handle(job, progress)

    handler.worker = w  # type: ignore[attr-defined]
    return handler


def main() -> None:
    import runpod  # the RunPod SDK: the job loop

    handler = make_handler(progress_update=lambda job, rec: runpod.serverless.progress_update(job, rec))
    w = handler.worker  # type: ignore[attr-defined]
    w.log(f"fastvideo-studio worker: catalog {w.cfg.catalog_path} ({len(w.catalog.models)} models), image {w.cfg.image or '?'}")
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
