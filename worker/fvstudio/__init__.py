"""fastvideo-studio worker: the RunPod handler's library.

Pure Python 3.10+, stdlib only: `rp` speaks RunPod's serverless protocol
(no `runpod` SDK), `r2` signs R2's S3 API (SigV4, no `boto3`).
The contract is `docs/jobs.md` and `catalog/catalog.json`; `docs/design.md`
has the reasoning.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
