"""
summary_registry.py — thread-safe, in-process tracking of every email-summary
job the orchestrator has kicked off. Mirrors orchestrator/rpa_registry.py's
role for IKDD form-filler runs, kept as its own registry for the same reason
rpa_registry.py is separate from registry.py: this job's meaningful
"result" (subject/body/paper_count) and progress shape (papers_scraped/
total_papers) are specific to it.
"""
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


def _key(conference: str, year: str) -> str:
    return f"{conference.strip().upper()}::{str(year).strip()}"


@dataclass
class SummaryJobRecord:
    conference: str
    year: str
    state: str = "not_started"   # not_started | queued | running | completed | failed
    stage: str = ""              # "" | "fetching_abstracts" | "summarizing" | "done"
    total_papers: int = 0
    papers_scraped: int = 0      # live progress during the fetching_abstracts stage
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[dict] = field(default_factory=dict)  # build_email() output once completed

    def to_dict(self) -> dict:
        return asdict(self)


class SummaryRegistry:
    """One instance shared by tools.py and summary_runner.py for the process's lifetime."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: dict[str, SummaryJobRecord] = {}

    def enqueue(self, conference: str, year: str) -> SummaryJobRecord:
        with self._lock:
            rec = SummaryJobRecord(conference=conference, year=str(year), state="queued")
            self._jobs[_key(conference, year)] = rec
            return rec

    def mark_running(self, conference: str, year: str, total_papers: int) -> None:
        with self._lock:
            rec = self._jobs.get(_key(conference, year))
            if rec:
                rec.state = "running"
                rec.stage = "fetching_abstracts"
                rec.total_papers = total_papers
                rec.started_at = datetime.now().isoformat()

    def update_progress(self, conference: str, year: str, papers_scraped: int) -> None:
        with self._lock:
            rec = self._jobs.get(_key(conference, year))
            if rec:
                rec.papers_scraped = papers_scraped

    def mark_summarizing(self, conference: str, year: str) -> None:
        with self._lock:
            rec = self._jobs.get(_key(conference, year))
            if rec:
                rec.stage = "summarizing"

    def mark_completed(self, conference: str, year: str, result: dict) -> None:
        with self._lock:
            rec = self._jobs.get(_key(conference, year))
            if rec is None:
                rec = SummaryJobRecord(conference=conference, year=str(year))
                self._jobs[_key(conference, year)] = rec
            rec.state = "completed"
            rec.stage = "done"
            rec.result = result
            rec.error = None
            rec.finished_at = datetime.now().isoformat()

    def mark_failed(self, conference: str, year: str, error: str) -> None:
        with self._lock:
            rec = self._jobs.get(_key(conference, year))
            if rec is None:
                rec = SummaryJobRecord(conference=conference, year=str(year))
                self._jobs[_key(conference, year)] = rec
            rec.state = "failed"
            rec.error = error
            rec.finished_at = datetime.now().isoformat()

    def get(self, conference: str, year: str) -> Optional[SummaryJobRecord]:
        with self._lock:
            return self._jobs.get(_key(conference, year))

    def all(self) -> list:
        with self._lock:
            return list(self._jobs.values())


# Process-wide singleton.
SUMMARY_REGISTRY = SummaryRegistry()
