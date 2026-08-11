"""
runner.py — executes SENTRY runs (fresh or retry-only) one at a time on a
single background worker thread, keeping orchestrator.registry.REGISTRY
updated as they progress.


"""
import json
import queue
import shutil
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import main_driver
import pipeline as paper_pipeline
from pipeline import RunBlockedError
from orchestrator.output_routing import route_output_to_file
from orchestrator.registry import REGISTRY

_JOB_QUEUE: "queue.Queue" = queue.Queue()
_WORKER_STARTED = False
_WORKER_LOCK = threading.Lock()


def _ensure_worker_started() -> None:
    global _WORKER_STARTED
    with _WORKER_LOCK:
        if not _WORKER_STARTED:
            threading.Thread(target=_worker_loop, daemon=True, name="sentry-run-worker").start()
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


def _log_path(conference: str, year: str) -> str:
    return f"data/orchestrator_logs/{conference}_{year}.log"


def _run_job(conference, year, proceeding_url, delay, track_skip_keywords, track_include_keywords):
    def _on_links_ready(path: str):
        REGISTRY.mark_links_ready(conference, year, path)

    REGISTRY.mark_running(conference, year, stage="link_extraction")
    with route_output_to_file(_log_path(conference, year)):
        try:
            main_driver.run_pipeline(
                proceeding_url=proceeding_url,
                conference=conference,
                year=str(year),
                max_papers=None,
                resume_from=0,
                delay=delay,
                interactive=False,
                track_skip_keywords=track_skip_keywords,
                track_include_keywords=track_include_keywords,
                on_links_ready=_on_links_ready,
            )
            REGISTRY.mark_completed(conference, year)
        except RunBlockedError as e:
            REGISTRY.mark_blocked(conference, year, str(e))
        except Exception as e:  
            REGISTRY.mark_failed(
                conference, year, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}"
            )


def start_run(
    conference: str,
    year: str,
    proceeding_url: str,
    delay: int = 10,
    track_skip_keywords: Optional[list] = None,
    track_include_keywords: Optional[list] = None,
) -> dict:
    existing = REGISTRY.get(conference, year)
    if existing and existing.state in ("running", "queued"):
        return {
            "status": f"already_{existing.state}",
            "conference": conference,
            "year": year,
            "stage": existing.stage,
            "message": (
                f"{conference} {year} is already {existing.state} "
                f"(stage: {existing.stage}) — call get_run_status instead of "
                "starting another one."
            ),
        }

    REGISTRY.enqueue(conference, year, proceeding_url, track_skip_keywords, track_include_keywords)
    _ensure_worker_started()
    _JOB_QUEUE.put(
        lambda: _run_job(
            conference, year, proceeding_url, delay, track_skip_keywords, track_include_keywords
        )
    )

    ahead = _JOB_QUEUE.qsize() 
    return {
        "status": "queued",
        "conference": conference,
        "year": year,
        "queue_position": ahead,
        "log_file": _log_path(conference, year),
        "message": (
            f"Queued {conference} {year} (position {ahead} — runs execute one "
            "at a time to avoid ACM/OpenReview session-cookie conflicts "
            "between concurrent runs). A full run over hundreds of papers "
            "can take a while — poll get_run_status to check progress, or "
            f"tail {_log_path(conference, year)} for the raw log."
        ),
    }

def _run_job_openreview(conference, year, venue_id, delay, skip_venue_keywords, include_only_venue_keywords):
    REGISTRY.mark_running(conference, year, stage="paper_processing")
    with route_output_to_file(_log_path(conference, year)):
        try:
            paper_pipeline.run_pipeline(
                conference=conference,
                year=str(year),
                venue_id=venue_id,
                skip_venue_keywords=skip_venue_keywords,
                include_only_venue_keywords=include_only_venue_keywords,
                max_papers=None,
                resume_from=0,
                delay=delay,
            )
            REGISTRY.mark_completed(conference, year)
        except RunBlockedError as e:
            REGISTRY.mark_blocked(conference, year, str(e))
        except Exception as e: 
            REGISTRY.mark_failed(
                conference, year, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}"
            )


def start_run_openreview(
    conference: str,
    year: str,
    venue_id: str,
    delay: int = 10,
    skip_venue_keywords: Optional[list] = None,
    include_only_venue_keywords: Optional[list] = None,
) -> dict:
    existing = REGISTRY.get(conference, year)
    if existing and existing.state in ("running", "queued"):
        return {
            "status": f"already_{existing.state}",
            "conference": conference,
            "year": year,
            "stage": existing.stage,
            "message": (
                f"{conference} {year} is already {existing.state} "
                f"(stage: {existing.stage}) — call get_run_status instead of "
                "starting another one."
            ),
        }

    REGISTRY.enqueue_openreview(
        conference, year, venue_id, skip_venue_keywords, include_only_venue_keywords
    )
    _ensure_worker_started()
    _JOB_QUEUE.put(
        lambda: _run_job_openreview(
            conference, year, venue_id, delay, skip_venue_keywords, include_only_venue_keywords
        )
    )

    ahead = _JOB_QUEUE.qsize()
    return {
        "status": "queued",
        "conference": conference,
        "year": year,
        "queue_position": ahead,
        "log_file": _log_path(conference, year),
        "message": (
            f"Queued {conference} {year} via OpenReview's API (venue_id="
            f"{venue_id}) — position {ahead} in the shared run queue. No "
            "scraping or browser involved for this mode; poll "
            "get_run_status for progress."
        ),
    }

def _guess_links_path(conference: str, year: str) -> Optional[str]:
    """
    Fallback for when the registry doesn't know input_links_path (e.g. the
    orchestrator process was restarted between the original run and the
    retry request). Tries the two locations SENTRY ever writes a final
    per-paper links.json to.
    """
    candidates = [
        Path(f"data/links_raw/{conference}/selected_links/{conference.lower()}/{year}/links.json"),
        Path(f"data/links_raw/{conference}/{year}/links.json"),
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return None


def _retry_job(conference, year, input_links_path, delay):
    REGISTRY.mark_running(conference, year, stage="retrying_errors")
    with route_output_to_file(_log_path(conference, year)):
        try:
            paper_pipeline.run_pipeline(
                conference=conference,
                year=str(year),
                input_links_path=input_links_path,
                max_papers=None,
                resume_from=0,
                delay=delay,
            )
            REGISTRY.mark_completed(conference, year)
        except RunBlockedError as e:
            REGISTRY.mark_blocked(conference, year, str(e))
        except Exception as e:  
            REGISTRY.mark_failed(
                conference, year, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}"
            )


def _retry_job_openreview(conference, year, venue_id, delay, skip_venue_keywords, include_only_venue_keywords):
    REGISTRY.mark_running(conference, year, stage="retrying_errors")
    with route_output_to_file(_log_path(conference, year)):
        try:
            paper_pipeline.run_pipeline(
                conference=conference,
                year=str(year),
                venue_id=venue_id,
                skip_venue_keywords=skip_venue_keywords,
                include_only_venue_keywords=include_only_venue_keywords,
                max_papers=None,
                resume_from=0,
                delay=delay,
            )
            REGISTRY.mark_completed(conference, year)
        except RunBlockedError as e:
            REGISTRY.mark_blocked(conference, year, str(e))
        except Exception as e: 
            REGISTRY.mark_failed(
                conference, year, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}"
            )


def start_retry(conference: str, year: str, delay: int = 10) -> dict:
    out_dir = Path(f"data/final_output/{conference}/{year}")
    processed_file = out_dir / "processed_papers.json"

    if not processed_file.exists():
        return {
            "status": "no_run_found",
            "conference": conference,
            "year": year,
            "message": f"No prior run found for {conference} {year} — nothing to retry.",
        }

    existing = REGISTRY.get(conference, year)
    if existing and existing.state in ("running", "queued"):
        return {
            "status": f"already_{existing.state}",
            "conference": conference,
            "year": year,
            "stage": existing.stage,
            "message": f"{conference} {year} is already {existing.state} (stage: {existing.stage}).",
        }

    processed_records = json.loads(processed_file.read_text(encoding="utf-8"))
    error_records = [r for r in processed_records if r.get("status") == "error"]

    if not error_records:
        return {
            "status": "no_errors",
            "conference": conference,
            "year": year,
            "message": f"No errored papers to retry for {conference} {year}.",
        }

    is_openreview = bool(existing and existing.mode == "openreview_api")

    if is_openreview:
        if not existing.venue_id:
            return {
                "status": "no_venue_id",
                "conference": conference,
                "year": year,
                "message": (
                    f"Found {len(error_records)} errored paper(s) for {conference} "
                    f"{year} from an OpenReview-API run, but this orchestrator "
                    "process has no record of the venue_id used (likely "
                    "restarted since). Re-run run_pipeline with venue_id "
                    "explicitly instead — it will skip papers already "
                    "completed and only reprocess the rest."
                ),
            }
    else:
        input_links_path = existing.input_links_path if existing else None
        if not input_links_path:
            input_links_path = _guess_links_path(conference, year)
        if not input_links_path:
            return {
                "status": "no_links_file",
                "conference": conference,
                "year": year,
                "message": (
                    f"Found {len(error_records)} errored paper(s) for {conference} "
                    f"{year}, but couldn't locate the original links.json to retry "
                    "against. Re-run run_pipeline for this conference instead."
                ),
            }

    # Un-mark the errored URLs as "processed" so pipeline.run_pipeline's
    # resume-skip logic will attempt them again on the next call.
    backup_path = processed_file.with_suffix(
        f".json.bak-{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    )
    shutil.copy2(processed_file, backup_path)

    kept_records = [r for r in processed_records if r.get("status") != "error"]
    processed_file.write_text(json.dumps(kept_records, indent=2), encoding="utf-8")

    REGISTRY.mark_retry_queued(conference, year)
    _ensure_worker_started()

    if is_openreview:
        _JOB_QUEUE.put(
            lambda: _retry_job_openreview(
                conference, year, existing.venue_id, delay,
                existing.skip_venue_keywords, existing.include_only_venue_keywords,
            )
        )
    else:
        REGISTRY.mark_links_ready(conference, year, input_links_path)
        _JOB_QUEUE.put(lambda: _retry_job(conference, year, input_links_path, delay))

    return {
        "status": "queued",
        "conference": conference,
        "year": year,
        "retrying_count": len(error_records),
        "log_file": _log_path(conference, year),
        "message": (
            f"Queued a retry of {len(error_records)} previously-failed "
            f"paper(s) for {conference} {year}."
        ),
    }
