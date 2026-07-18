"""
rpa_registry.py — thread-safe, in-process tracking of every IKDD form-filler
(RPA) job the orchestrator has kicked off. Mirrors orchestrator/registry.py's
role for AEGIS extraction runs, but kept as its own registry rather than
reused, because the two jobs track genuinely different things:
RunRecord's input_links_path/track_keywords fields are extraction-pipeline
concepts that don't apply here, and this job's meaningful "result" is
submission counts (submitted/skipped/failed), not a scraped-links path.
"""
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


def _key(conference: str, year: str) -> str:
    return f"{conference.strip().upper()}::{str(year).strip()}"


@dataclass
class RPAJobRecord:
    conference: str
    year: str
    venue: str = ""
    month: str = ""
    state: str = "not_started"   # not_started | queued | running | completed | failed
    total_candidates: int = 0
    duplicates_skipped: int = 0
    submitted: int = 0
    failed: int = 0
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    duplicate_details: list = field(default_factory=list)
    details: list = field(default_factory=list)  # per-paper submit/fail results from the last run

    def to_dict(self) -> dict:
        return asdict(self)


class RPARegistry:
    """One instance shared by tools.py and rpa_runner.py for the process's lifetime."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: dict[str, RPAJobRecord] = {}

    def enqueue(self, conference: str, year: str, venue: str, month: str) -> RPAJobRecord:
        with self._lock:
            rec = RPAJobRecord(
                conference=conference,
                year=str(year),
                venue=venue,
                month=month,
                state="queued",
            )
            self._jobs[_key(conference, year)] = rec
            return rec

    def mark_running(self, conference: str, year: str) -> None:
        with self._lock:
            rec = self._jobs.get(_key(conference, year))
            if rec:
                rec.state = "running"
                rec.started_at = datetime.now().isoformat()

    def mark_completed(self, conference: str, year: str, result: dict) -> None:
        with self._lock:
            rec = self._jobs.get(_key(conference, year))
            if rec is None:
                rec = RPAJobRecord(conference=conference, year=str(year))
                self._jobs[_key(conference, year)] = rec
            rec.state = "completed"
            rec.total_candidates = result.get("total_candidates", 0)
            rec.duplicates_skipped = result.get("duplicates_skipped", 0)
            rec.submitted = result.get("submitted", 0)
            rec.failed = result.get("failed", 0)
            rec.duplicate_details = result.get("duplicate_details", [])
            rec.details = result.get("details", [])
            rec.error = None
            rec.finished_at = datetime.now().isoformat()

    def mark_failed(self, conference: str, year: str, error: str) -> None:
        with self._lock:
            rec = self._jobs.get(_key(conference, year))
            if rec is None:
                rec = RPAJobRecord(conference=conference, year=str(year))
                self._jobs[_key(conference, year)] = rec
            rec.state = "failed"
            rec.error = error
            rec.finished_at = datetime.now().isoformat()

    def get(self, conference: str, year: str) -> Optional[RPAJobRecord]:
        with self._lock:
            return self._jobs.get(_key(conference, year))

    def all(self) -> list:
        with self._lock:
            return list(self._jobs.values())


# Process-wide singleton.
RPA_REGISTRY = RPARegistry()
