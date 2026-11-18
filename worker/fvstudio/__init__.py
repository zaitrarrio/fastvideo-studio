"""fastvideo-studio worker: the RunPod handler's library.

Pure Python 3.10+, stdlib plus `runpod` (the job loop) and `boto3` (R2).
The contract is `docs/jobs.md` and `catalog/catalog.json`; `docs/design.md`
has the reasoning.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
