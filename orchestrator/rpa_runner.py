"""
rpa_runner.py — executes IKDD form-filler (RPA) jobs one at a time on a
single background worker thread, keeping orchestrator.rpa_registry's
RPA_REGISTRY updated as they progress. Mirrors runner.py's job-queue
pattern for the AEGIS extraction pipeline, kept as a separate queue/worker
because it's a genuinely separate resource: Form_filler drives a single
real Chrome/Selenium session per job, and running two of those concurrently
on one machine risks window-focus/resource contention between them the
same way two concurrent scraping runs would race on a shared cookie file.
Jobs queue and run strictly one after another either way.

start_form_filler returns immediately with status "queued" (or
"already_running"/"already_queued"); get_status reflects the queued ->
running -> completed/failed transitions.

RPA_REGISTRY is purely in-memory and only knows about jobs actually started
via start_form_filler *in this process's lifetime* — it has no idea whether
data/final_output/<conference>/<year>/indian_papers_structured.json exists
on disk. That file is written independently by the AEGIS extraction
pipeline (pipeline.py) and is exactly what run_form_filler needs to submit
anything, so a conference/year can be fully extracted and ready to submit
while RPA_REGISTRY still reports "not_started" — e.g. right after a process
restart, or if extraction ran in an earlier process. get_status/list_runs
below cross-check the filesystem so callers (the orchestrator agent, and
the UI on top of it) see readiness that isn't tied to registry state.
"""
import json
import queue
import threading
import traceback
from pathlib import Path
from typing import Optional

from orchestrator.rpa_registry import RPA_REGISTRY

_JOB_QUEUE: "queue.Queue" = queue.Queue()
_WORKER_STARTED = False
_WORKER_LOCK = threading.Lock()

REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_OUTPUT_DIR = REPO_ROOT / "data" / "final_output"


def _structured_json_path(conference: str, year: str) -> Path:
    return FINAL_OUTPUT_DIR / conference / str(year) / "indian_papers_structured.json"


def _extracted_candidate_count(json_path: Path) -> Optional[int]:
    """Best-effort paper count from an indian_papers_structured.json. None if unreadable."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return len(data) if isinstance(data, list) else None
    except Exception:
        return None


def _scan_extracted_conferences() -> dict:
    """
    Walks data/final_output/*/*/indian_papers_structured.json directly,
    independent of RPA_REGISTRY. Returns {(conference, year): candidate_count}.
    """
    found: dict = {}
    if not FINAL_OUTPUT_DIR.exists():
        return found
    for conf_dir in FINAL_OUTPUT_DIR.iterdir():
        if not conf_dir.is_dir():
            continue
        for year_dir in conf_dir.iterdir():
            if not year_dir.is_dir():
                continue
            json_path = year_dir / "indian_papers_structured.json"
            if json_path.exists():
                found[(conf_dir.name, year_dir.name)] = _extracted_candidate_count(json_path)
    return found


def _ensure_worker_started() -> None:
    global _WORKER_STARTED
    with _WORKER_LOCK:
        if not _WORKER_STARTED:
            threading.Thread(target=_worker_loop, daemon=True, name="aegis-rpa-worker").start()
            _WORKER_STARTED = True


def _worker_loop() -> None:
    while True:
        job = _JOB_QUEUE.get()
        try:
            job()
        except Exception:  # noqa: BLE001 — one job's bug must not kill the worker
            traceback.print_exc()
        finally:
            _JOB_QUEUE.task_done()


def _job(conference, year, month, venue, form_url, refresh_dedup_cache) -> None:
    RPA_REGISTRY.mark_running(conference, year)

    try:
        # Imported lazily (not at module top) so that environments without
        # Selenium/Chrome/chromedriver configured can still import and use
        # the rest of the orchestrator (extraction, dedup, etc.) without
        # this module's import failing at process startup.
        from Form_filler.run_selenium_filler import run_form_filler
    except ImportError as e:
        RPA_REGISTRY.mark_failed(
            conference, year,
            f"Could not import the form filler ({type(e).__name__}: {e}). "
            "Is selenium installed, and is a matching chromedriver on PATH?",
        )
        return

    try:
        result = run_form_filler(
            conference=conference,
            year=str(year),
            month=month,
            venue=venue,
            form_url=form_url,
            refresh_dedup_cache=refresh_dedup_cache,
        )
        RPA_REGISTRY.mark_completed(conference, year, result)
    except Exception as e:  # noqa: BLE001
        RPA_REGISTRY.mark_failed(
            conference, year, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}"
        )


def start_form_filler(
    conference: str,
    year: str,
    month: str,
    venue: str,
    form_url: Optional[str] = None,
    refresh_dedup_cache: bool = True,
) -> dict:
    existing = RPA_REGISTRY.get(conference, year)
    if existing and existing.state in ("running", "queued"):
        return {
            "status": f"already_{existing.state}",
            "conference": conference,
            "year": year,
            "message": (
                f"A form-filler run for {conference} {year} is already "
                f"{existing.state} — call get_rpa_status instead of "
                "starting another one."
            ),
        }

    RPA_REGISTRY.enqueue(conference, year, venue, month)
    _ensure_worker_started()
    _JOB_QUEUE.put(
        lambda: _job(conference, year, month, venue, form_url, refresh_dedup_cache)
    )

    ahead = _JOB_QUEUE.qsize()  # rough — includes this job if not yet picked up
    return {
        "status": "queued",
        "conference": conference,
        "year": year,
        "venue": venue,
        "month": month,
        "queue_position": ahead,
        "message": (
            f"Queued the IKDD form-filler for {conference} {year} "
            f"(venue='{venue}', month='{month}') — position {ahead} (RPA "
            "jobs run one at a time, same reasoning as the extraction "
            "queue: a single real Chrome session per job). It will first "
            "refresh the IKDD dedup cache (New + Approved) and skip "
            "anything already there, then submit only genuinely new "
            "papers. Poll get_rpa_status for progress."
        ),
    }


def get_status(conference: str, year: str) -> dict:
    rec = RPA_REGISTRY.get(conference, year)
    json_path = _structured_json_path(conference, year)
    has_extracted_data = json_path.exists()
    extracted_candidates = _extracted_candidate_count(json_path) if has_extracted_data else None

    if rec is None:
        if has_extracted_data:
            return {
                "conference": conference,
                "year": year,
                "state": "ready_to_submit",
                "has_extracted_data": True,
                "extracted_candidates": extracted_candidates,
                "message": (
                    f"No form-filler run has been started yet for {conference} {year}, "
                    f"but {json_path} exists with "
                    f"{extracted_candidates if extracted_candidates is not None else 'an unknown number of'} "
                    "candidate paper(s) — call initiate_form_filler to submit them."
                ),
            }
        return {
            "conference": conference,
            "year": year,
            "state": "not_started",
            "has_extracted_data": False,
            "message": (
                "No form-filler run found for this conference/year, and no "
                "extracted data on disk either — run_pipeline hasn't "
                "completed (or hasn't been started) for it yet."
            ),
        }

    result = rec.to_dict()
    # Even when the registry has a record, surface on-disk state too — e.g.
    # a "completed" run from earlier plus fresh data re-extracted since.
    result["has_extracted_data"] = has_extracted_data
    result["extracted_candidates"] = extracted_candidates
    return result


def list_runs() -> dict:
    tracked_by_key = {(r.conference, r.year): r for r in RPA_REGISTRY.all()}
    runs = []

    for (conference, year), rec in tracked_by_key.items():
        json_path = _structured_json_path(conference, year)
        d = rec.to_dict()
        d["has_extracted_data"] = json_path.exists()
        d["extracted_candidates"] = _extracted_candidate_count(json_path) if d["has_extracted_data"] else None
        runs.append(d)

    # Conferences/years with extracted data on disk but no registry entry —
    # i.e. never (yet) submitted this process — surface as "ready_to_submit"
    # so they aren't invisible just because RPA_REGISTRY never saw them.
    for (conference, year), count in _scan_extracted_conferences().items():
        if (conference, year) in tracked_by_key:
            continue
        runs.append({
            "conference": conference,
            "year": year,
            "venue": "",
            "month": "",
            "state": "ready_to_submit",
            "total_candidates": 0,
            "duplicates_skipped": 0,
            "submitted": 0,
            "failed": 0,
            "started_at": None,
            "finished_at": None,
            "error": None,
            "duplicate_details": [],
            "details": [],
            "has_extracted_data": True,
            "extracted_candidates": count,
        })

    return {"runs": runs}
