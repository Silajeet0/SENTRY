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

from orchestrator import conference_catalog, ikdd_form_catalog, runner, rpa_runner, summary_runner
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
    proceeding_url: str = None,
    venue_id: str = None,
    delay: int = 10,
    skip_track_keywords: list = None,
    include_track_keywords: list = None,
    skip_venue_keywords: list = None,
    include_only_venue_keywords: list = None,
) -> dict:
    """
    Exactly one of proceeding_url / venue_id should be given — use venue_id
    for OpenReview-hosted conferences (ICML/ICLR/...), which resolve_conference_url
    returns a venue_id for rather than a proceeding_url. venue_id runs bypass
    scraping entirely (OpenReview's API + ground-truth affiliation data) and
    use skip_venue_keywords/include_only_venue_keywords instead of the
    track-keyword params, which only apply to scraped/grouped conferences.
    """
    if venue_id:
        return runner.start_run_openreview(
            conference=conference,
            year=year,
            venue_id=venue_id,
            delay=delay,
            skip_venue_keywords=skip_venue_keywords,
            include_only_venue_keywords=include_only_venue_keywords,
        )
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
# RPA / IKDD form-filler tools
#
# These are deliberately separate from run_pipeline/get_run_status/list_runs
# above: run_pipeline extracts and classifies papers (scraping + LLM), while
# initiate_form_filler is a downstream, independent step that takes papers
# ALREADY extracted (data/final_output/<conference>/<year>/indian_papers_
# structured.json must already exist) and submits the new ones to IKDD via
# Selenium. A person can ask for one without the other, e.g. re-running the
# form filler after fixing a rejected submission without re-scraping.
# ---------------------------------------------------------------------------
def resolve_ikdd_form_metadata(conference: str, year: str) -> dict:
    return ikdd_form_catalog.resolve(conference, year)


def initiate_form_filler(
    conference: str,
    year: str,
    venue: str = None,
    month: str = None,
    form_url: str = None,
    refresh_dedup_cache: bool = True,
) -> dict:
    """
    Starts the IKDD Selenium form-filler (RPA) for one conference/year in
    the background and returns immediately — hundreds of papers at several
    seconds each adds up, so poll get_rpa_status afterwards rather than
    waiting here, same pattern as run_pipeline/get_run_status.

    venue/month must be the EXACT text of the form's dropdown options. If
    either is omitted, this looks them up via resolve_ikdd_form_metadata
    first and only proceeds if that resolves cleanly — never guesses, since
    a wrong dropdown value fails Selenium hard mid-run instead of cleanly
    upfront.
    """
    if not venue or not month:
        meta = ikdd_form_catalog.resolve(conference, year)
        if not meta["resolved"]:
            return {
                "status": "needs_input",
                "conference": conference,
                "year": year,
                "message": meta["message"],
            }
        venue = venue or meta["venue"]
        month = month or meta["month"]

    return rpa_runner.start_form_filler(
        conference=conference,
        year=year,
        month=month,
        venue=venue,
        form_url=form_url,
        refresh_dedup_cache=refresh_dedup_cache,
    )


def get_rpa_status(conference: str, year: str) -> dict:
    return rpa_runner.get_status(conference, year)


def list_rpa_runs() -> dict:
    return rpa_runner.list_runs()


# ---------------------------------------------------------------------------
# Email-summary tools
#
# A THIRD downstream, independent step alongside initiate_form_filler —
# same relationship to run_pipeline: it reads papers ALREADY extracted
# (data/final_output/<conference>/<year>/indian_papers_structured.json must
# already exist) rather than re-scraping/re-classifying anything. Fetches
# each paper's abstract (OpenReview API directly for ICML/ICLR/..., the
# same four-tier scraper as run_pipeline for everything else — see
# summarizer/abstract_fetcher.py), then writes a cited email-body summary
# via a SEPARATE summarization LLM (see summarizer/email_summarizer.py for
# why — a different model/temperature than the main extraction/orchestrator
# LLM). Runs in the background like run_pipeline/initiate_form_filler —
# poll get_summary_status rather than waiting here.
# ---------------------------------------------------------------------------
def summarize_indian_authors(conference: str, year: str, refresh_cache: bool = False) -> dict:
    return summary_runner.start_summary(conference=conference, year=year, refresh_cache=refresh_cache)


def get_summary_status(conference: str, year: str) -> dict:
    return summary_runner.get_status(conference, year)


def list_summary_runs() -> dict:
    return summary_runner.list_runs()


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
    "resolve_ikdd_form_metadata": resolve_ikdd_form_metadata,
    "initiate_form_filler": initiate_form_filler,
    "get_rpa_status": get_rpa_status,
    "list_rpa_runs": list_rpa_runs,
    "summarize_indian_authors": summarize_indian_authors,
    "get_summary_status": get_summary_status,
    "list_summary_runs": list_summary_runs,
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
                "get_run_status afterwards rather than waiting here. "
                "IMPORTANT: for OpenReview-hosted conferences (ICML, ICLR, "
                "and their oral/spotlight variants — resolve_conference_url "
                "returns mode='openreview_api' with a venue_id for these), "
                "pass venue_id instead of proceeding_url — that mode calls "
                "OpenReview's API directly with no scraping/browser involved "
                "at all, and uses skip_venue_keywords/include_only_venue_keywords "
                "instead of the track-keyword params below. For every other "
                "conference, pass proceeding_url and optionally filter which "
                "tracks get scraped by keyword."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "conference": {"type": "string"},
                    "year": {"type": "string"},
                    "proceeding_url": {
                        "type": "string",
                        "description": "For scraped-mode conferences only — omit when using venue_id.",
                    },
                    "venue_id": {
                        "type": "string",
                        "description": (
                            "For OpenReview-hosted conferences only, e.g. "
                            "'ICML.cc/2025/Conference'. Omit when using proceeding_url."
                        ),
                    },
                    "delay": {
                        "type": "integer",
                        "description": "Seconds to wait between papers. Default 10.",
                    },
                    "skip_track_keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Scraped/grouped conferences only. Track/session "
                            "titles containing any of these (case-insensitive) "
                            "are excluded from scraping, e.g. ['workshop', 'tutorial']."
                        ),
                    },
                    "include_track_keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Scraped/grouped conferences only. If given, ONLY tracks containing one of these are included.",
                    },
                    "skip_venue_keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "venue_id/OpenReview mode only. Same idea as "
                            "skip_track_keywords but filters OpenReview venue "
                            "groups instead. Defaults to ['Workshop', 'Tutorial'] "
                            "if not given."
                        ),
                    },
                    "include_only_venue_keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "venue_id/OpenReview mode only. If given, ONLY venue groups containing one of these are included.",
                    },
                },
                "required": ["conference", "year"],
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
                "List every EXTRACTION (scraping/classification) conference/"
                "year run this orchestrator session knows about, with "
                "current state. Useful when the person gives a vague "
                "follow-up like 'retry the errors' without naming a "
                "conference — call this first to see which runs actually "
                "have errors. For IKDD form-filler/RPA submission runs, use "
                "list_rpa_runs instead — the two are tracked separately."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_ikdd_form_metadata",
            "description": (
                "Resolve a conference name + year to the EXACT 'venue' and "
                "'month' dropdown text the IKDD submission form expects. "
                "Never guesses — returns resolved=false and asks for the "
                "exact text directly if this conference isn't on file, "
                "since a wrong dropdown value fails Selenium hard mid-run "
                "rather than cleanly upfront. Call this before "
                "initiate_form_filler if you don't already have venue/month "
                "from the person."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "conference": {"type": "string", "description": "e.g. 'NeurIPS', 'IEEE-ICDM'."},
                    "year": {"type": "string", "description": "e.g. '2025'."},
                },
                "required": ["conference", "year"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "initiate_form_filler",
            "description": (
                "Start the IKDD Selenium form-filler (RPA) for one "
                "conference/year in the background and return immediately — "
                "poll get_rpa_status afterwards rather than waiting here. "
                "Requires that run_pipeline has ALREADY completed for this "
                "conference/year (it reads data/final_output/<conference>/"
                "<year>/indian_papers_structured.json). Before submitting "
                "anything, this dedup-checks every candidate paper against "
                "IKDD's current New + Approved lists and skips anything "
                "already present — only genuinely new papers get submitted. "
                "If venue/month are omitted, they're looked up via "
                "resolve_ikdd_form_metadata; if that can't resolve them "
                "either, this returns status='needs_input' asking the "
                "person for the exact dropdown text instead of guessing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "conference": {"type": "string"},
                    "year": {"type": "string"},
                    "venue": {
                        "type": "string",
                        "description": "Exact IKDD form dropdown text, e.g. 'ICDM'. Omit to auto-resolve via resolve_ikdd_form_metadata.",
                    },
                    "month": {
                        "type": "string",
                        "description": "Exact IKDD form dropdown text, e.g. 'Nov'. Omit to auto-resolve via resolve_ikdd_form_metadata.",
                    },
                    "form_url": {
                        "type": "string",
                        "description": "Override the default IKDD submission form URL. Omit unless the person gives a different one.",
                    },
                    "refresh_dedup_cache": {
                        "type": "boolean",
                        "description": "Re-scrape IKDD's New+Approved lists before checking (default true). Set false only to reuse a recently-refreshed local cache for speed.",
                    },
                },
                "required": ["conference", "year"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_rpa_status",
            "description": (
                "Get current progress/results for an IKDD form-filler (RPA) "
                "run — submitted/skipped/failed counts — whether it's still "
                "running or finished. Also checks disk for this conference/"
                "year's data/final_output/.../indian_papers_structured.json "
                "independent of run history: if no RPA run has ever been "
                "started but that file exists, state comes back "
                "'ready_to_submit' (with extracted_candidates count) instead "
                "of a misleading 'not_started'. has_extracted_data/"
                "extracted_candidates are included alongside any tracked "
                "run's own state too. Separate from get_run_status, which "
                "covers extraction runs."
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
            "name": "list_rpa_runs",
            "description": (
                "List every IKDD form-filler (RPA) run this orchestrator "
                "session knows about, with current state — PLUS every "
                "conference/year on disk under data/final_output/ that has "
                "an indian_papers_structured.json but was never submitted "
                "in this session (state 'ready_to_submit', with an "
                "extracted_candidates count). Use this to discover what's "
                "available to submit, not just what's already been run. "
                "Separate from list_runs, which covers extraction runs."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_indian_authors",
            "description": (
                "Summarize the work of Indian-affiliated authors at a "
                "conference/year as a cited email body — e.g. 'summarize "
                "the works of the Indian authors in ICML 2025'. Requires "
                "that run_pipeline has ALREADY completed for this "
                "conference/year (reads data/final_output/<conference>/"
                "<year>/indian_papers_structured.json) — if it hasn't, this "
                "returns status='no_extracted_data' and extraction should "
                "be run first, NOT this tool. Fetches each paper's abstract "
                "(from OpenReview's API directly for OpenReview-hosted "
                "conferences, or by scraping otherwise) and writes a short "
                "plain-English summary of each one, with the paper title, "
                "Indian author names/institutions, and a link — the summary "
                "text itself is model-written but every citation detail "
                "(title/authors/institutions/link) comes straight from the "
                "already-verified extraction data, not from the summarizing "
                "model. Runs in the BACKGROUND and returns immediately — "
                "poll get_summary_status for progress and the final "
                "subject/body once done, the same pattern as run_pipeline/"
                "get_run_status."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "conference": {"type": "string"},
                    "year": {"type": "string"},
                    "refresh_cache": {
                        "type": "boolean",
                        "description": (
                            "Re-fetch every paper's abstract instead of reusing "
                            "the cached ones from a previous summary run for this "
                            "conference/year. Default false — abstracts don't "
                            "change once a paper is published, so there's "
                            "normally no reason to re-scrape."
                        ),
                    },
                },
                "required": ["conference", "year"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_summary_status",
            "description": (
                "Get progress/results for an email-summary run — whether "
                "it's still fetching abstracts, writing the summary, or "
                "done. Once state is 'completed', the result field has the "
                "full subject/body ready to hand back to the person (or "
                "pass to a 'send this' step) — read it from here rather "
                "than assuming success just because summarize_indian_authors "
                "returned 'queued'. Also checks disk independent of run "
                "history, the same way get_rpa_status does for form-filler "
                "runs: if no summary run has been started this session but "
                "email_summary.json already exists (from an earlier process) "
                "or indian_papers_structured.json exists but no summary has "
                "been generated yet, state comes back 'completed' or "
                "'ready_to_summarize' respectively instead of a misleading "
                "'not_started'."
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
            "name": "list_summary_runs",
            "description": (
                "List every email-summary run this orchestrator session "
                "knows about, with current state — PLUS every conference/"
                "year on disk under data/final_output/ that has an "
                "indian_papers_structured.json but no summary generated yet "
                "in this session (state 'ready_to_summarize'). Use this for "
                "a vague 'summarize what we've got' request with no "
                "conference named, the same way list_runs/list_rpa_runs "
                "resolve vague follow-ups for their own steps."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
