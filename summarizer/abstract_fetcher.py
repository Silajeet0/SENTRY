"""
abstract_fetcher.py — deterministic per-paper abstract retrieval for the
email-summary feature. No LLM calls here.

Two paths, mirroring the exact split pipeline.py / run_single_paper.py
already use elsewhere in AEGIS:

    OpenReview URLs ("openreview.net" in paper_url — ICML, ICLR, ...)
        -> workflows.link_extractors.openreview_api_fetcher.fetch_single_openreview_paper
           The abstract is already a field on the OpenReview note. No
           scraping, no browser, no regex needed.

    Everything else (ACL/NeurIPS PDFs, IEEE/ACM via the browser tier, ...)
        -> reuse pipeline.TIERS (the exact same HTMLScraper/BrowserScraper/
           PDFScraper/APIScraper instances the main pipeline uses — same
           persistent Playwright browser, same ACM cookie jar, no separate
           network stack to keep warm) to get raw page/PDF-first-page text,
           then a regex heuristic pulls just the Abstract window out of it.

Title / authors / institutions are deliberately NOT re-derived from scraped
text here — indian_papers_structured.json already has them from the main
pipeline's extraction LLM call, and that's what the email-summary feature
treats as ground truth for citations.
"""
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Same per-request politeness delay philosophy as pipeline.py's
# INTER_PAPER_DELAY — only applied to the tiered-scraper path below, since
# the OpenReview path is one authenticated API call, not a scrape against a
# server that might rate-limit/block us.
DEFAULT_SCRAPE_DELAY_SECONDS = 3

# Section headings that mark the end of an abstract in scraped page/PDF
# text. Checked case-insensitively; order doesn't matter, the EARLIEST match
# in the text wins. Deliberately broad — a false-positive cut (stopping too
# early) still yields a usable, if shorter, abstract; missing every marker
# and running the abstract into the next section is the worse failure mode.
ABSTRACT_STOP_MARKERS = [
    "1 introduction", "1. introduction", "i. introduction", "introduction",
    "keywords", "index terms", "ccs concepts",
    "1 related work", "1. related work", "related work",
    "acm reference format", "categories and subject descriptors",
    "table of contents", "permission to make digital",
    "1 background", "1. background",
]

# Hard cap on abstract length fed downstream — keeps the aggregate
# summarization prompt (dozens of papers at once) within a small local
# model's context window regardless of how verbose a given abstract or
# scraped page is.
MAX_ABSTRACT_CHARS = 2500

# If no "Abstract" heading is found at all, fall back to a leading window of
# the cleaned content rather than failing outright — degraded but usable.
FALLBACK_WINDOW_CHARS = 1500


def is_openreview_url(url: str) -> bool:
    return "openreview.net" in (url or "")


def extract_abstract_from_content(content: str) -> tuple[str, bool]:
    """
    Returns (abstract_text, found_heading). found_heading=False means the
    FALLBACK_WINDOW_CHARS leading-text fallback was used instead of a real
    'Abstract' heading match — callers surface this so a degraded abstract
    is visible rather than silently indistinguishable from a clean one.

    Pages (IEEE Xplore in particular) often contain the word "Abstract"
    twice: once as a tab/nav link near the top of the page ("...Full Text
    Views Abstract Document Sections 1. Introduction...") and once as the
    real heading further down ("Abstract: Deep learning models..."). The
    nav occurrence is immediately followed by a stop marker, so matching
    only the FIRST occurrence silently grabs the nav link instead of the
    real abstract. Evaluate every occurrence and keep whichever yields the
    longest text before its nearest stop marker — the real heading is the
    one with an actual paragraph after it.
    """
    if not content:
        return "", False

    matches = list(re.finditer(r"\babstract\b\s*[:\-—]?\s*", content, re.IGNORECASE))
    if not matches:
        fallback = re.sub(r"\s+", " ", content[:FALLBACK_WINDOW_CHARS]).strip()
        return fallback, False

    best_abstract = ""
    for match in matches:
        after = content[match.end():]
        lower_after = after.lower()

        cut = len(after)
        for marker in ABSTRACT_STOP_MARKERS:
            idx = lower_after.find(marker)
            if idx != -1:
                cut = min(cut, idx)

        candidate = re.sub(r"\s+", " ", after[:cut]).strip()
        if len(candidate) > len(best_abstract):
            best_abstract = candidate

    # A heading match with almost nothing after it before the next marker
    # (e.g. "Abstract" as a nav link, immediately followed by "Keywords")
    # isn't a real abstract — fall back to the leading-window heuristic
    # instead of returning a near-empty string.
    if len(best_abstract) < 40:
        fallback = re.sub(r"\s+", " ", content[:FALLBACK_WINDOW_CHARS]).strip()
        return fallback, False

    return best_abstract, True


def fetch_abstract_for_paper(paper: dict, delay_seconds: float = DEFAULT_SCRAPE_DELAY_SECONDS) -> dict:
    """
    paper: one entry from indian_papers_structured.json (has paper_url,
    paper_title, all_authors, authors_with_indian_affiliations, source, ...).

    Returns a dict:
        {
          "paper_url": str,
          "abstract": str,          # "" on failure
          "abstract_source": str,   # "openreview_api" | "html" | "browser" |
                                     # "pdf" | "neurips_html_fallback" |
                                     # "api" | "" (on failure)
          "heading_found": bool,    # False = degraded fallback-window abstract
          "error": str,             # "" on success
        }
    Never raises — failures go into "error", matching BaseScraper's contract.
    """
    url = paper.get("paper_url", "")
    out = {"paper_url": url, "abstract": "", "abstract_source": "", "heading_found": False, "error": ""}

    if not url:
        out["error"] = "Paper record has no paper_url"
        return out

    if is_openreview_url(url):
        try:
            from workflows.link_extractors.openreview_api_fetcher import fetch_single_openreview_paper
            record = fetch_single_openreview_paper(url)
            abstract = (record.get("abstract") or "").strip()
            if not abstract:
                out["error"] = "OpenReview note has no abstract field"
                return out
            out["abstract"] = abstract[:MAX_ABSTRACT_CHARS]
            out["abstract_source"] = "openreview_api"
            out["heading_found"] = True  # structured field, not a heuristic match
            return out
        except Exception as e:  # noqa: BLE001
            out["error"] = f"{type(e).__name__}: {e}"
            return out

    # Non-OpenReview: reuse the exact tier instances pipeline.py uses (same
    # persistent Playwright browser / ACM cookie jar as the main run, rather
    # than spinning up a second one).
    from pipeline import TIERS

    content = ""
    content_source = ""
    tier_errors = []
    for tier in TIERS:
        if not tier.can_handle(url):
            continue
        result = tier.scrape(url)
        if result.success:
            content = result.content
            content_source = result.source
            break
        tier_errors.append(f"{tier.__class__.__name__}: {result.error}")

    if delay_seconds:
        time.sleep(delay_seconds)

    if not content:
        out["error"] = "All tiers failed — " + " | ".join(tier_errors) if tier_errors else "No tier could handle this URL"
        return out

    abstract, heading_found = extract_abstract_from_content(content)
    if not abstract:
        out["error"] = "Scraped content but found no usable abstract text"
        return out

    out["abstract"] = abstract[:MAX_ABSTRACT_CHARS]
    out["abstract_source"] = content_source
    out["heading_found"] = heading_found
    return out


# ---------------------------------------------------------------------------
# Crash-safe, resumable cache — mirrors pipeline.py's "save after every item"
# philosophy so a long scraping pass across dozens of papers survives a
# restart, and so asking for the same conference/year's summary again later
# doesn't re-scrape everything from scratch.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]


def cache_path(conference: str, year: str) -> Path:
    return REPO_ROOT / "data" / "final_output" / conference / str(year) / "abstracts_cache.json"


def load_cache(conference: str, year: str) -> dict:
    p = cache_path(conference, year)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — a corrupt cache shouldn't crash the run
        log.warning(f"Could not parse existing abstracts cache at {p} — starting fresh")
        return {}


def save_cache(conference: str, year: str, cache: dict) -> None:
    p = cache_path(conference, year)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def fetch_abstracts_for_papers(
    papers: list[dict],
    conference: str,
    year: str,
    refresh_cache: bool = False,
    delay_seconds: float = DEFAULT_SCRAPE_DELAY_SECONDS,
    on_progress=None,
) -> list[dict]:
    """
    Fetches (or reuses cached) abstracts for every paper, saving the cache
    after each newly-fetched paper. Returns one dict per input paper, each
    merging the original paper record with the fetch_abstract_for_paper()
    result (so callers get paper_title/authors/institutions/url + abstract
    all in one place).

    on_progress(done, total), if given, is called after every paper —
    used by summary_runner.py to keep orchestrator.summary_registry's live
    progress counters current the same way pipeline.py's per-paper _save()
    keeps get_run_status current.
    """
    cache = {} if refresh_cache else load_cache(conference, year)
    total = len(papers)
    results = []

    for i, paper in enumerate(papers, start=1):
        url = paper.get("paper_url", "")
        cached = cache.get(url)

        if cached is not None:
            fetch_result = cached
        else:
            fetch_result = fetch_abstract_for_paper(paper, delay_seconds=delay_seconds)
            cache[url] = fetch_result
            save_cache(conference, year, cache)

        merged = dict(paper)
        merged.update(fetch_result)
        results.append(merged)

        if on_progress:
            on_progress(i, total)

    return results
