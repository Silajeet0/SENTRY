"""
summary_runner.py — executes email-summary jobs one at a time on a single
background worker thread, keeping orchestrator.summary_registry's
SUMMARY_REGISTRY updated as they progress. Mirrors rpa_runner.py's
job-queue pattern.

Why one worker instead of one thread per job: fetch_abstracts_for_papers
falls through to pipeline.TIERS for non-OpenReview papers, which includes
BrowserScraper's single persistent Playwright browser/ACM cookie jar — the
exact same shared-resource reason runner.py and rpa_runner.py each use a
single serial queue instead of true concurrency. Summary jobs queue behind
each other (and implicitly behind whatever else is using pipeline.TIERS,
since it's the same shared scraper instances) rather than racing.

start_summary returns immediately with status "queued" (or
"already_running"/"already_queued"/"no_extracted_data"); get_status reflects
the queued -> running (fetching_abstracts -> summarizing) -> completed/failed
transitions, with live papers_scraped/total_papers progress during the
fetching_abstracts stage.
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
            threading.Thread(target=_worker_loop, daemon=True, name="aegis-summary-worker").start()
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


def _job(conference: str, year: str, refresh_cache: bool, delay_seconds: float) -> None:
    # Imported lazily so importing this module doesn't require the full
    # scraping stack (Playwright etc.) to be installed just to check status
    # or list runs elsewhere in the orchestrator.
    from summarizer.abstract_fetcher import fetch_abstracts_for_papers
    from summarizer.email_summarizer import SummaryLLM, build_email

    # Print the ACTUALLY-RESOLVED config for this run, not what .env
    # currently says on disk. load_dotenv() only runs once, at process
    # start — if orchestrator_api.py has been running since before
    # SUMMARY_LLM_* was added/changed in .env, os.getenv() here will
    # silently fall back to LLM_* (the SAME model the orchestrator's own
    # tool-calling loop uses), and the two compete for one quota. This line
    # makes that visible immediately in the server's own log instead of only
    # being inferable from which model name shows up in the HTTP request
    # logs — if this doesn't say what you expect, restart orchestrator_api.py
    # so it picks up the current .env.
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


def start_summary(
    conference: str,
    year: str,
    refresh_cache: bool = False,
    delay_seconds: float = 3,
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
