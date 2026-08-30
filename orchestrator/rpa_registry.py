"""
rpa_registry.py — thread-safe, in-process tracking of every IKDD form-filler (or any other suitable venue)
(RPA) job the orchestrator has kicked off. 
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
    state: str = "not_started"   
    total_candidates: int = 0
    duplicates_skipped: int = 0
    submitted: int = 0
    failed: int = 0
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    duplicate_details: list = field(default_factory=list)
    details: list = field(default_factory=list) 

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

    def update_progress(self, conference: str, year: str, submitted: int, failed: int, total_candidates: int, duplicates_skipped: int = 0) -> None:
        """
        Called after each paper during a running job — this is what makes
        get_rpa_status actually move while the job is still in progress,
        instead of showing a frozen 0/0/0 for the entire run (which for
        ~80+ papers, several multi-author, can genuinely take over an
        hour — see process_papers_with_selenium's per-author/per-paper
        sleeps). Does not touch state/started_at/finished_at — only
        mark_completed/mark_failed own those transitions.
        """
        with self._lock:
            rec = self._jobs.get(_key(conference, year))
            if rec:
                rec.submitted = submitted
                rec.failed = failed
                rec.total_candidates = total_candidates
                rec.duplicates_skipped = duplicates_skipped

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
