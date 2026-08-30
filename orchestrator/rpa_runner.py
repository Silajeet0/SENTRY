"""
rpa_runner.py — executes IKDD form-filler (RPA) jobs one at a time on a
single background worker thread, keeping orchestrator.rpa_registry's
RPA_REGISTRY updated as they progress.
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
    """Paper count from an indian_papers_structured.json. None if unreadable."""
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
        except Exception: 
            traceback.print_exc()
        finally:
            _JOB_QUEUE.task_done()


def _job(conference, year, month, venue, form_url, refresh_dedup_cache) -> None:
    RPA_REGISTRY.mark_running(conference, year)

    try:
        from Form_filler.run_selenium_filler import run_form_filler
    except ImportError as e:
        RPA_REGISTRY.mark_failed(
            conference, year,
            f"Could not import the form filler ({type(e).__name__}: {e}). "
            "Is selenium installed, and is a matching chromedriver on PATH?",
        )
        return

    try:
        def _report_progress(results_so_far, total_candidates=0, duplicates_skipped=0):
            submitted = sum(1 for d in results_so_far if d["status"] == "submitted")
            failed = sum(1 for d in results_so_far if d["status"] == "failed")
            RPA_REGISTRY.update_progress(
                conference, year,
                submitted=submitted, failed=failed,
                total_candidates=total_candidates, duplicates_skipped=duplicates_skipped,
            )

        result = run_form_filler(
            conference=conference,
            year=str(year),
            month=month,
            venue=venue,
            form_url=form_url,
            refresh_dedup_cache=refresh_dedup_cache,
            on_progress=_report_progress,
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

    ahead = _JOB_QUEUE.qsize()  
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
