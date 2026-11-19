#!/usr/bin/env python3
"""RunPod serverless handler for fastvideo-studio (docs/design.md §1,
docs/jobs.md). One worker process, concurrency 1, three engines as
sidecars. The job loop is `fvstudio.rp` (RunPod's protocol over `urllib`,
no SDK); `make_handler()` is what the tests drive with a recording
`progress_update`."""

from __future__ import annotations

import argparse
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fvstudio import rp  # noqa: E402
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="fastvideo-studio RunPod worker")
    ap.add_argument("--test-input", "--test_input", metavar="FILE",
                    help="run this one job ({\"input\": …}) locally and exit, as the SDK's --test_input did")
    args = ap.parse_args(argv)

    env = rp.RunpodEnv.from_env()
    if args.test_input or env is None:
        handler = make_handler(progress_update=lambda job, rec: rp._stderr(f"progress {job['id']}: {rec}"))
        return rp.run_local(handler, args.test_input, log=handler.worker.log)  # type: ignore[attr-defined]

    client = rp.RunpodClient(env)
    handler = make_handler(progress_update=client.progress_update)
    w = handler.worker  # type: ignore[attr-defined]
    client.log = w.log
    w.log(f"fastvideo-studio worker: catalog {w.cfg.catalog_path} ({len(w.catalog.models)} models), image {w.cfg.image or '?'}")
    stop = threading.Event()
    rp.install_signal_handlers(stop, log=w.log)
    rp.serve(handler, client, stop=stop, log=w.log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
