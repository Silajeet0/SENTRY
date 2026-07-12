"""
tools.py — the tool surface exposed to the LLM-driven orchestrator agent.

Every tool is a plain, synchronous, JSON-in/JSON-out Python function.
agent.py is what turns TOOL_SCHEMAS into an OpenAI-style `tools=[...]` list
and dispatches calls through TOOL_FUNCTIONS — this module has no idea an LLM
is involved, which keeps it independently testable and callable from plain
Python (e.g. from a Jupyter cell) with no agent loop at all.

Two design choices worth calling out:

- run_pipeline / retry_errors return immediately (they start a background
  thread — see runner.py) rather than blocking. A single AEGIS run over
  hundreds of papers at `delay` seconds apart routinely takes minutes to
  hours; holding a tool-call turn open that long isn't workable. Progress is
  polled afterwards via get_run_status, which just reads the same files
  pipeline.py already saves after every single paper.

- resolve_conference_url never guesses at ACM/IEEE proceeding URLs. See
  conference_catalog.py's docstring for why.
"""
import json
from pathlib import Path

from orchestrator import conference_catalog, runner
from orchestrator.registry import REGISTRY


def resolve_conference_url(conference: str, year: str) -> dict:
    return conference_catalog.resolve_conference_url(conference, year)


def validate_url(url: str) -> dict:
    return conference_catalog.validate_url(url)


def detect_structure(conference: str, proceeding_url: str = "") -> dict:
    return conference_catalog.detect_structure(conference, proceeding_url)


def run_pipeline(
    conference: str,
    year: str,
    proceeding_url: str,
    delay: int = 10,
    skip_track_keywords: list = None,
    include_track_keywords: list = None,
) -> dict:
    return runner.start_run(
        conference=conference,
        year=year,
        proceeding_url=proceeding_url,
        delay=delay,
        track_skip_keywords=skip_track_keywords,
        track_include_keywords=include_track_keywords,
    )


def get_run_status(conference: str, year: str) -> dict:
    out_dir = Path(f"data/final_output/{conference}/{year}")
    rec = REGISTRY.get(conference, year)

    result = {
        "conference": conference,
        "year": str(year),
        "orchestrator_state": rec.state if rec else "unknown",
        "stage": rec.stage if rec else "",
        "run_error": rec.error if rec else None,
        "retry_count": rec.retry_count if rec else 0,
    }

    summary_file = out_dir / "summary.json"
    errors_file = out_dir / "errors.json"
    processed_file = out_dir / "processed_papers.json"

    if not processed_file.exists() and not summary_file.exists():
        result["found_output"] = False
        result["message"] = (
            "No output on disk yet for this conference/year — it may still "
            "be extracting links, or hasn't started."
        )
        return result

    result["found_output"] = True

    # Detect stale data: if this run is still queued/extracting links, it
    # hasn't written its own summary.json yet — anything currently on disk
    # is leftover from a PREVIOUS run of this same conference/year, not live
    # progress. _save() stamps summary.json with the writing run's own
    # started_at, so comparing that against the registry's started_at for
    # *this* run tells them apart precisely.
    is_stale = False
    if rec and rec.stage in ("queued", "link_extraction") and rec.started_at:
        if summary_file.exists():
            file_started_at = json.loads(summary_file.read_text(encoding="utf-8")).get("started_at")
            if file_started_at and file_started_at < rec.started_at:
                is_stale = True

    if is_stale:
        result["stale_data"] = True
        result["message"] = (
            "The numbers below are leftover from a PREVIOUS run of this "
            "conference/year — not live progress. The current run is still "
            "extracting/selecting links and hasn't processed any papers yet, "
            "so it hasn't written its own output. Check back once stage "
            "moves past 'link_extraction'."
        )

    if summary_file.exists():
        result["summary"] = json.loads(summary_file.read_text(encoding="utf-8"))

    if processed_file.exists():
        processed = json.loads(processed_file.read_text(encoding="utf-8"))
        result["papers_attempted"] = len(processed)
        breakdown: dict = {}
        for p in processed:
            s = p.get("status", "unknown")
            breakdown[s] = breakdown.get(s, 0) + 1
        result["status_breakdown"] = breakdown

    if errors_file.exists():
        errors = json.loads(errors_file.read_text(encoding="utf-8"))
        result["current_error_count"] = len(errors)
        result["sample_errors"] = errors[:5]

    if not is_stale:
        total_input = result.get("summary", {}).get("total_input_links")
        attempted = result.get("papers_attempted")
        if total_input and attempted is not None:
            result["progress_pct"] = round(100 * attempted / total_input, 1)

    return result


def retry_errors(conference: str, year: str, delay: int = 10) -> dict:
    return runner.start_retry(conference=conference, year=year, delay=delay)


def list_runs() -> dict:
    return {"runs": [r.to_dict() for r in REGISTRY.all()]}


# ---------------------------------------------------------------------------
# Dispatch table + OpenAI-style function-calling schemas
# ---------------------------------------------------------------------------
TOOL_FUNCTIONS = {
    "resolve_conference_url": resolve_conference_url,
    "validate_url": validate_url,
    "detect_structure": detect_structure,
    "run_pipeline": run_pipeline,
    "get_run_status": get_run_status,
    "retry_errors": retry_errors,
    "list_runs": list_runs,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "resolve_conference_url",
            "description": (
                "Resolve a conference name + year to its proceedings URL. "
                "Never guesses ACM/IEEE proceeding URLs (per-instance DOIs "
                "with no stable pattern across years) — returns "
                "resolved=false and asks for the URL directly in that case."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "conference": {
                        "type": "string",
                        "description": "Conference name, e.g. 'NeurIPS', 'ICML', 'ACL', 'ACM_KDD', 'IEEE-ICDM'.",
                    },
                    "year": {"type": "string", "description": "Conference year, e.g. '2025'."},
                },
                "required": ["conference", "year"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_url",
            "description": "Check that a proceedings URL is reachable before spending time scraping it.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_structure",
            "description": (
                "Determine which scraping path a conference will take: "
                "'flat' (NeurIPS/IEEE-style — one link per paper) or "
                "'grouped' (ACL/ACM/OpenReview-style — papers organized "
                "under tracks/sessions that get filtered before scraping)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "conference": {"type": "string"},
                    "proceeding_url": {
                        "type": "string",
                        "description": "Optional — needed to distinguish ACM DL URLs from other grouped sources.",
                    },
                },
                "required": ["conference"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_pipeline",
            "description": (
                "Start the AEGIS extraction pipeline for one conference/year "
                "in the background and return immediately — a full run over "
                "hundreds of papers can take a long time, so poll "
                "get_run_status afterwards rather than waiting here. For "
                "grouped/track-based conferences, optionally filter which "
                "tracks get scraped by keyword."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "conference": {"type": "string"},
                    "year": {"type": "string"},
                    "proceeding_url": {"type": "string"},
                    "delay": {
                        "type": "integer",
                        "description": "Seconds to wait between papers. Default 10.",
                    },
                    "skip_track_keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Track/session titles containing any of these "
                            "(case-insensitive) are excluded from scraping, "
                            "e.g. ['workshop', 'tutorial']. Only affects "
                            "grouped conferences."
                        ),
                    },
                    "include_track_keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "If given, ONLY tracks containing one of these are included.",
                    },
                },
                "required": ["conference", "year", "proceeding_url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_run_status",
            "description": (
                "Get current progress, results, and errors for a "
                "conference/year run, whether it's still running or finished."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "conference": {"type": "string"},
                    "year": {"type": "string"},
                },
                "required": ["conference", "year"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retry_errors",
            "description": (
                "Re-run only the papers that previously failed for a "
                "conference/year, in the background. No-ops cleanly if "
                "there's nothing to retry or no prior run is found."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "conference": {"type": "string"},
                    "year": {"type": "string"},
                    "delay": {
                        "type": "integer",
                        "description": "Seconds to wait between papers. Default 10.",
                    },
                },
                "required": ["conference", "year"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_runs",
            "description": (
                "List every conference/year run this orchestrator session "
                "knows about, with current state. Useful when the person "
                "gives a vague follow-up like 'retry the errors' without "
                "naming a conference — call this first to see which runs "
                "actually have errors."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
