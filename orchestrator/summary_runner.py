"""
summary_runner.py — executes email-summary jobs one at a time on a single
background worker thread, keeping orchestrator.summary_registry's
SUMMARY_REGISTRY updated as they progress. Mirrors rpa_runner.py's
job-queue pattern.

"""
import json
import queue
import threading
import traceback
from pathlib import Path
from typing import Optional

from orchestrator.summary_registry import SUMMARY_REGISTRY

_JOB_QUEUE: "queue.Queue" = queue.Queue()
_WORKER_STARTED = False
_WORKER_LOCK = threading.Lock()

REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_OUTPUT_DIR = REPO_ROOT / "data" / "final_output"


def _structured_json_path(conference: str, year: str) -> Path:
    return FINAL_OUTPUT_DIR / conference / str(year) / "indian_papers_structured.json"


def _email_summary_path(conference: str, year: str) -> Path:
    return FINAL_OUTPUT_DIR / conference / str(year) / "email_summary.json"


def _load_papers(conference: str, year: str) -> Optional[list]:
    p = _structured_json_path(conference, year)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else None
    except Exception:
        return None


def _ensure_worker_started() -> None:
    global _WORKER_STARTED
    with _WORKER_LOCK:
        if not _WORKER_STARTED:
            threading.Thread(target=_worker_loop, daemon=True, name="sentry-summary-worker").start()
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


def _job(conference: str, year: str, refresh_cache: bool, delay_seconds: float) -> None:
    from summarizer.abstract_fetcher import fetch_abstracts_for_papers
    from summarizer.email_summarizer import SummaryLLM, build_email


    _probe = SummaryLLM()
    print(
        f"[summary_runner] {conference} {year}: summarizer resolved to "
        f"model={_probe.model!r} base_url={_probe.base_url!r} "
        f"temperature={_probe.temperature} — if this looks like your MAIN "
        f"orchestrator model rather than a distinct summarizer model, "
        f"SUMMARY_LLM_* likely isn't set (or this process needs restarting "
        f"to pick up a recent .env change)."
    )

    papers = _load_papers(conference, year)
    if papers is None:
        SUMMARY_REGISTRY.mark_failed(
            conference, year,
            f"No indian_papers_structured.json found for {conference} {year} — "
            "run extraction (run_pipeline) for this conference/year first.",
        )
        return

    if not papers:
        SUMMARY_REGISTRY.mark_running(conference, year, total_papers=0)
        result = {
            "subject": f"Summary: Indian-Authored Papers at {conference} {year} (0 papers)",
            "body": f"No Indian-affiliated papers were found for {conference} {year}.",
            "paper_count": 0,
            "papers_included": [],
            "papers_skipped": [],
        }
        _email_summary_path(conference, year).write_text(json.dumps(result, indent=2), encoding="utf-8")
        SUMMARY_REGISTRY.mark_completed(conference, year, result)
        return

    SUMMARY_REGISTRY.mark_running(conference, year, total_papers=len(papers))

    try:
        def _on_progress(done: int, total: int) -> None:
            SUMMARY_REGISTRY.update_progress(conference, year, done)

        papers_with_abstracts = fetch_abstracts_for_papers(
            papers, conference, year,
            refresh_cache=refresh_cache,
            delay_seconds=delay_seconds,
            on_progress=_on_progress,
        )

        SUMMARY_REGISTRY.mark_summarizing(conference, year)
        summaries_by_index = SummaryLLM().summarize_all(papers_with_abstracts, conference, year)
        result = build_email(conference, year, papers_with_abstracts, summaries_by_index)

        out_path = _email_summary_path(conference, year)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        SUMMARY_REGISTRY.mark_completed(conference, year, result)
    except Exception as e:  # noqa: BLE001
        SUMMARY_REGISTRY.mark_failed(
            conference, year, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}"
        )


def check_existing_summary(conference: str = None, year: str = None) -> dict:
    """
    Read-only check for an already-generated email_summary.json on disk —
    never enqueues or affects anything. Two modes:

      - conference AND year given: checks just that pair and returns the
        cached result (subject/paper_count/generated_at) if present, so a
        caller can decide whether summarize_indian_authors is even needed.
      - neither given: scans data/final_output/ for EVERY conference/year
        that already has a summary, e.g. to answer "which conferences have
        already been summarized?" directly, without generating anything.
    """
    if conference and year:
        path = _email_summary_path(conference, year)
        if not path.exists():
            return {
                "conference": conference,
                "year": str(year),
                "exists": False,
                "message": f"No existing summary on disk for {conference} {year}.",
            }
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            result = None
        return {
            "conference": conference,
            "year": str(year),
            "exists": True,
            "paper_count": (result or {}).get("paper_count"),
            "subject": (result or {}).get("subject"),
            "generated_at": (result or {}).get("generated_at"),
            "result": result,
            "message": (
                f"A summary for {conference} {year} already exists on disk — "
                "use it directly (get_summary_status / this result's 'result' "
                "field) instead of calling summarize_indian_authors again."
            ),
        }

    summarized = []
    if FINAL_OUTPUT_DIR.exists():
        for conf_dir in sorted(FINAL_OUTPUT_DIR.iterdir()):
            if not conf_dir.is_dir():
                continue
            for year_dir in sorted(conf_dir.iterdir()):
                if not year_dir.is_dir():
                    continue
                summary_path = year_dir / "email_summary.json"
                if not summary_path.exists():
                    continue
                try:
                    result = json.loads(summary_path.read_text(encoding="utf-8"))
                except Exception:
                    result = {}
                summarized.append({
                    "conference": conf_dir.name,
                    "year": year_dir.name,
                    "paper_count": result.get("paper_count"),
                    "subject": result.get("subject"),
                    "generated_at": result.get("generated_at"),
                })

    return {
        "exists": bool(summarized),
        "count": len(summarized),
        "summarized": summarized,
        "message": (
            f"{len(summarized)} conference/year(s) already have a generated "
            "summary on disk." if summarized else
            "No conference/year has a generated summary on disk yet."
        ),
    }


def start_summary(
    conference: str,
    year: str,
    refresh_cache: bool = False,
    delay_seconds: float = 3,
    force: bool = False,
) -> dict:
    existing = SUMMARY_REGISTRY.get(conference, year)
    if existing and existing.state in ("running", "queued"):
        return {
            "status": f"already_{existing.state}",
            "conference": conference,
            "year": year,
            "stage": existing.stage,
            "message": (
                f"A summary run for {conference} {year} is already "
                f"{existing.state} — call get_summary_status instead of "
                "starting another one."
            ),
        }

    if not force:
        existing_summary_path = _email_summary_path(conference, year)
        if existing_summary_path.exists():
            try:
                result = json.loads(existing_summary_path.read_text(encoding="utf-8"))
            except Exception:
                result = None
            return {
                "status": "already_summarized",
                "conference": conference,
                "year": year,
                "result": result,
                "message": (
                    f"{conference} {year} already has a generated summary on "
                    "disk — not starting a new run. Fetching abstracts and "
                    "re-running the summarizer LLM on every paper again is "
                    "expensive and unnecessary if this cached summary is what "
                    "was actually wanted; use the 'result' field above (or "
                    "get_summary_status) directly for the digest instead. "
                    "Only call summarize_indian_authors again with force=True "
                    "if a genuinely fresh summary is needed (e.g. abstracts "
                    "changed, or refresh_cache is specifically wanted)."
                ),
            }

    if _load_papers(conference, year) is None:
        return {
            "status": "no_extracted_data",
            "conference": conference,
            "year": year,
            "message": (
                f"No indian_papers_structured.json found for {conference} {year} "
                "— run extraction (run_pipeline) for this conference/year first, "
                "then summarize it."
            ),
        }

    SUMMARY_REGISTRY.enqueue(conference, year)
    _ensure_worker_started()
    _JOB_QUEUE.put(lambda: _job(conference, year, refresh_cache, delay_seconds))

    ahead = _JOB_QUEUE.qsize()
    return {
        "status": "queued",
        "conference": conference,
        "year": year,
        "queue_position": ahead,
        "message": (
            f"Queued the email-summary run for {conference} {year} — position "
            f"{ahead} (summary jobs share the same scraper resources as "
            "extraction runs, so they queue one at a time). It will fetch each "
            "paper's abstract"
            + ("" if not refresh_cache else " (ignoring any cached abstracts and re-fetching all of them)")
            + ", then write a cited email-body summary. Poll get_summary_status for progress."
        ),
    }


def get_status(conference: str, year: str) -> dict:
    rec = SUMMARY_REGISTRY.get(conference, year)
    has_extracted_data = _structured_json_path(conference, year).exists()
    existing_summary_path = _email_summary_path(conference, year)

    if rec is None:
        if existing_summary_path.exists():
            try:
                result = json.loads(existing_summary_path.read_text(encoding="utf-8"))
            except Exception:
                result = None
            if result is not None:
                return {
                    "conference": conference,
                    "year": year,
                    "state": "completed",
                    "stage": "done",
                    "has_extracted_data": has_extracted_data,
                    "result": result,
                    "message": (
                        f"Found a previously-generated summary on disk for {conference} "
                        f"{year} (from an earlier process — this session's registry has "
                        "no record of running it). Call summarize_indian_authors again "
                        "if you want a fresh one."
                    ),
                }
        if has_extracted_data:
            return {
                "conference": conference,
                "year": year,
                "state": "ready_to_summarize",
                "has_extracted_data": True,
                "message": (
                    f"No summary run has been started yet for {conference} {year}, but "
                    "extraction has already completed — call summarize_indian_authors."
                ),
            }
        return {
            "conference": conference,
            "year": year,
            "state": "not_started",
            "has_extracted_data": False,
            "message": (
                "No summary run found for this conference/year, and no extracted data "
                "on disk either — run_pipeline hasn't completed (or hasn't been "
                "started) for it yet."
            ),
        }

    result = rec.to_dict()
    result["has_extracted_data"] = has_extracted_data
    return result


def list_runs() -> dict:
    tracked_by_key = {(r.conference, r.year): r for r in SUMMARY_REGISTRY.all()}
    runs = []

    for (conference, year), rec in tracked_by_key.items():
        d = rec.to_dict()
        d["has_extracted_data"] = _structured_json_path(conference, year).exists()
        runs.append(d)

    if FINAL_OUTPUT_DIR.exists():
        for conf_dir in FINAL_OUTPUT_DIR.iterdir():
            if not conf_dir.is_dir():
                continue
            for year_dir in conf_dir.iterdir():
                if not year_dir.is_dir():
                    continue
                key = (conf_dir.name, year_dir.name)
                if key in tracked_by_key:
                    continue
                if not (year_dir / "indian_papers_structured.json").exists():
                    continue
                summary_path = year_dir / "email_summary.json"
                if summary_path.exists():
                    state = "completed"
                else:
                    state = "ready_to_summarize"
                runs.append({
                    "conference": conf_dir.name,
                    "year": year_dir.name,
                    "state": state,
                    "stage": "done" if state == "completed" else "",
                    "has_extracted_data": True,
                })

    return {"runs": runs}
