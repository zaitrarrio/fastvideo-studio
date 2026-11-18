"""The one error type a job can fail with: a message and `where`."""

from __future__ import annotations

Where = str  # "validate" | "stage" | "engine" | "mux" | "upload"


class JobError(Exception):
    """A refused or failed job: `{"error": str(e), "where": e.where}`."""

    def __init__(self, where: Where, message: str, log_tail: list[str] | None = None):
        super().__init__(message)
        self.where = where
        self.message = message
        self.log_tail = log_tail or []

    def to_output(self) -> dict:
        out: dict = {"error": self.message, "where": self.where}
        if self.log_tail:
            out["log_tail"] = self.log_tail
        return out
