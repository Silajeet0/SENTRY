"""
registry.py — thread-safe, in-process tracking of every conference/year run
the orchestrator has kicked off.

"""
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


def _key(conference: str, year: str) -> str:
    return f"{conference.strip().upper()}::{str(year).strip()}"


@dataclass
class RunRecord:
    conference: str
    year: str
    proceeding_url: str = ""
    mode: str = "scraped"        
    venue_id: Optional[str] = None
    skip_venue_keywords: list = field(default_factory=list)
    include_only_venue_keywords: list = field(default_factory=list)
    state: str = "not_started"     
    stage: str = ""              
    input_links_path: Optional[str] = None
    track_skip_keywords: list = field(default_factory=list)
    track_include_keywords: list = field(default_factory=list)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class RunRegistry:
    """One instance shared by tools.py and runner.py for the process's lifetime."""

    def __init__(self):
        self._lock = threading.Lock()
        self._runs: dict[str, RunRecord] = {}

    def enqueue(
        self,
        conference: str,
        year: str,
        proceeding_url: str,
        track_skip_keywords=None,
        track_include_keywords=None,
    ) -> RunRecord:
        """Record a run as queued — not yet actually executing. Runs execute
        one at a time (see runner.py's single worker thread) because SENTRY's
        ACM/OpenReview fetchers share a single on-disk cookie file per
        handler type; running two of the same handler concurrently would
        race on that file."""
        with self._lock:
            rec = RunRecord(
                conference=conference,
                year=str(year),
                proceeding_url=proceeding_url,
                mode="scraped",
                state="queued",
                stage="queued",
                track_skip_keywords=list(track_skip_keywords or []),
                track_include_keywords=list(track_include_keywords or []),
            )
            self._runs[_key(conference, year)] = rec
            return rec

    def enqueue_openreview(
        self,
        conference: str,
        year: str,
        venue_id: str,
        skip_venue_keywords=None,
        include_only_venue_keywords=None,
    ) -> RunRecord:
        with self._lock:
            rec = RunRecord(
                conference=conference,
                year=str(year),
                mode="openreview_api",
                venue_id=venue_id,
                skip_venue_keywords=list(skip_venue_keywords or []),
                include_only_venue_keywords=list(include_only_venue_keywords or []),
                state="queued",
                stage="queued",
            )
            self._runs[_key(conference, year)] = rec
            return rec

    def mark_running(self, conference: str, year: str, stage: str = "link_extraction") -> None:
        with self._lock:
            rec = self._runs.get(_key(conference, year))
            if rec:
                rec.state = "running"
                rec.stage = stage
                rec.started_at = datetime.now().isoformat()

    def mark_links_ready(self, conference: str, year: str, input_links_path: str) -> None:
        with self._lock:
            rec = self._runs.get(_key(conference, year))
            if rec:
                rec.input_links_path = input_links_path
                rec.stage = "paper_processing"

    def mark_completed(self, conference: str, year: str) -> None:
        with self._lock:
            rec = self._runs.get(_key(conference, year))
            if rec:
                rec.state = "completed"
                rec.stage = "done"
                rec.finished_at = datetime.now().isoformat()

    def mark_failed(self, conference: str, year: str, error: str) -> None:
        with self._lock:
            rec = self._runs.get(_key(conference, year))
            if rec:
                rec.state = "failed"
                rec.error = error
                rec.finished_at = datetime.now().isoformat()
            else:
                self._runs[_key(conference, year)] = RunRecord(
                    conference=conference,
                    year=str(year),
                    state="failed",
                    error=error,
                    finished_at=datetime.now().isoformat(),
                )

    def mark_blocked(self, conference: str, year: str, message: str) -> None:
        """Distinct from mark_failed — the run stopped itself deliberately
        because a domain's circuit breaker tripped (pipeline.RunBlockedError),
        not because of an unexpected crash."""
        with self._lock:
            rec = self._runs.get(_key(conference, year))
            if rec is None:
                rec = RunRecord(conference=conference, year=str(year))
                self._runs[_key(conference, year)] = rec
            rec.state = "blocked"
            rec.error = message
            rec.finished_at = datetime.now().isoformat()

    def mark_retry_queued(self, conference: str, year: str) -> RunRecord:
        with self._lock:
            rec = self._runs.get(_key(conference, year))
            if rec is None:
                rec = RunRecord(conference=conference, year=str(year))
                self._runs[_key(conference, year)] = rec
            rec.state = "queued"
            rec.stage = "queued"
            rec.retry_count += 1
            rec.error = None
            rec.finished_at = None
            return rec

    def get(self, conference: str, year: str) -> Optional[RunRecord]:
        with self._lock:
            return self._runs.get(_key(conference, year))

    def all(self) -> list:
        with self._lock:
            return list(self._runs.values())


# Process-wide singleton.
REGISTRY = RunRegistry()
