"""
Main pipeline: drop-in replacement for agent_caller.py.

Two input modes, one output contract:

    A) input_links_path given  → original tiered-scraper flow
       URL → Tier1(HTML) → Tier2(Browser) → Tier3(PDF) → Tier4(API)
                                                               ↓
                                                       LLM Extractor
                                                               ↓
                                                       PaperInfo

    B) venue_id given          → OpenReview API flow (no scraping tiers)
       fetch_openreview_papers() → ground-truth affiliation check
                                        ↓
                              only positive papers → LLM (area_of_research)
                                        ↓
                                    PaperInfo

    "ambiguous" (a collision-prone acronym like IIT/ISI/NIT/VIT with no
    stronger corroboration — see evaluation/india_rules.py) is treated the
    same as "negative" everywhere in this file: it never counts as an
    Indian affiliation in the output, and never triggers an LLM call on
    its own.

Exactly one of input_links_path / venue_id must be provided to run_pipeline.
Both modes write the identical indian_papers_structured.json / errors.json /
processed_papers.json / summary.json shape via the shared _save() — nothing
downstream needs to know which mode produced a given conference's output.
"""
import json
import random
import time
import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from scrapers.html_scraper import HTMLScraper
from scrapers.browser_scraper import BrowserScraper
from scrapers.pdf_scraper import PDFScraper
from scrapers.api_scraper import APIScraper
from extractors.llm_extractor import LLMExtractor, PaperInfo
from evaluation.india_rules import classify_affiliation, combine_author_decisions
from workflows.link_extractors.openreview_api_fetcher import fetch_openreview_papers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# Ordered tiers — pipeline tries each in sequence until one succeeds
TIERS = [
    HTMLScraper(),
    BrowserScraper(),
    PDFScraper(),
    APIScraper(),       # always last — API fallback works for any URL
]

PDF_PREFILTER_DOMAINS = {
    "papers.nips.cc",
    "aclanthology.org",
}

EXTRACTOR = LLMExtractor()

# Delay between papers (seconds) — be polite to servers
INTER_PAPER_DELAY = 3

# ---------------------------------------------------------------------------
# Block detection / circuit breaker
# ---------------------------------------------------------------------------
# Substrings that indicate the *server* is refusing us (bot challenge, IP
# rate-limit, outright block) as opposed to a normal per-paper failure
# (malformed content, LLM hiccup, missing DOI). Only the former should ever
# trip the breaker — a run shouldn't stop just because a few individual
# papers had bad metadata. Only relevant to the tiered-scraper path
# (input_links_path) — the OpenReview API path's errors won't match these
# and the breaker simply never trips there, which is correct: there's no
# per-domain HTTP scraping to be blocked from in that mode.
BLOCK_SIGNAL_SUBSTRINGS = [
    "bot challenge detected",
    "just a moment",
    "access denied",
    "unusual traffic",
    "temporarily blocked",
    "forbidden",
    "429",
    "too many requests",
]

# Consecutive block-looking failures *from the same domain* before the run
# stops itself rather than continuing to hammer a domain that's actively
# blocking this IP — burning through the rest of the queue against a dead
# connection wastes hours and can extend the block.
MAX_CONSECUTIVE_BLOCK_SIGNALS = 4


class RunBlockedError(Exception):
    """Raised when a domain trips the circuit breaker — the run stopped
    itself early rather than continuing to hit a domain that's blocking
    this IP. Distinct from a generic per-paper error."""

    def __init__(self, domain: str, consecutive_signals: int, papers_attempted: int):
        self.domain = domain
        self.consecutive_signals = consecutive_signals
        self.papers_attempted = papers_attempted
        super().__init__(
            f"Stopped after {consecutive_signals} consecutive block-like "
            f"failures from {domain} ({papers_attempted} papers attempted "
            "this run before stopping)."
        )


def _looks_like_block(error: str) -> bool:
    if not error:
        return False
    e = error.lower()
    return any(sig in e for sig in BLOCK_SIGNAL_SUBSTRINGS)


# ─────────────────────────────────────────────────────────────────────────
# Tiered-scraper path (HTML/IEEE/ACM/ACL/etc via a flat list of URLs)
# ─────────────────────────────────────────────────────────────────────────
def process_paper(url: str) -> PaperInfo:
    """
    Process a single paper URL through the tiered pipeline.
    Returns PaperInfo regardless of outcome — check .error for failures.
    """
    content = ""
    content_source = "none"
    tier_errors = []

    for tier in TIERS:
        if not tier.can_handle(url):
            log.debug(f"  [{tier.__class__.__name__}] skipped (can_handle=False)")
            continue

        log.debug(f"  [{tier.__class__.__name__}] trying...")
        result = tier.scrape(url)

        if result.success:
            content = result.content
            content_source = result.source
            log.debug(f"  [{tier.__class__.__name__}] ✓ got {len(content)} chars")
            break
        else:
            log.debug(f"  [{tier.__class__.__name__}] ✗ {result.error}")
            tier_errors.append(f"{tier.__class__.__name__}: {result.error}")

    if not content:
        return PaperInfo(
            paper_url=url,
            error="All tiers failed — no content retrieved | " + " | ".join(tier_errors),
            raw_content_source="none"
        )

    if _should_skip_llm_by_prefilter(url, content, content_source):
        return PaperInfo(paper_url=url, raw_content_source=content_source, source=content_source)

    # Single LLM extraction call
    info = EXTRACTOR.extract(content, url, content_source)
    info.source = content_source
    return info


def _should_skip_llm_by_prefilter(url, content, content_source) -> bool:
    '''skip llm calls for pdf's that  definitely aren't Indian, saves compute and time'''
    if content_source != "pdf":
        return False

    domain = urlparse(url).netloc

    if domain not in PDF_PREFILTER_DOMAINS:
        return False

    return classify_affiliation(content).label != "positive"


# ─────────────────────────────────────────────────────────────────────────
# OpenReview API path (no scraping tiers — data already structured)
# ─────────────────────────────────────────────────────────────────────────
def _build_openreview_llm_content(paper: dict) -> str:
    """
    Synthetic content string in the same "Title / Abstract / Author |
    Affiliation" shape the LLM extractor already expects from scraped
    ACM/IEEE pages — lets us reuse EXTRACTION_PROMPT unchanged instead of
    writing a second prompt/parser just for OpenReview.
    """
    lines = [f"Title: {paper['title']}", "", f"Abstract: {paper['abstract']}", ""]
    for author in paper["authors"]:
        lines.append(f"Author: {author['name']} | Affiliation: {author['affiliation']}")
    return "\n".join(lines)


def process_openreview_paper(paper: dict) -> PaperInfo:
    """
    Ground-truth-first version of process_paper for OpenReview papers.
    No scraping tiers — `paper` already has title/abstract/authors/
    affiliations from the API. classify_affiliation runs on real
    institution/country strings before any LLM call is made.
    """
    decisions = [classify_affiliation(a["affiliation"]) for a in paper["authors"]]
    overall_label = combine_author_decisions(decisions)

    if overall_label in ("negative", "ambiguous"):
        # No LLM call at all — same shape as the tiered path's prefilter
        # skip (empty area_of_research, empty Indian-affiliation fields).
        # "ambiguous" (e.g. an author's affiliation only matches a
        # collision-prone acronym like "IIT"/"ISI"/"NIT"/"VIT" with no
        # stronger corroboration) is treated identically to "negative"
        # here — it will never produce an Indian-affiliated author below,
        # so there's no reason to spend an LLM call on it.
        return PaperInfo(
            paper_url=paper["paper_url"],
            raw_content_source="openreview_api",
            source="openreview_api",
        )

    # At least one author flagged positive — one LLM call, mainly for
    # area_of_research (and a second opinion on the author list, though our
    # own ground-truth data below takes precedence over its guess).
    content = _build_openreview_llm_content(paper)
    info = EXTRACTOR.extract(content, paper["paper_url"], content_source="openreview_api")
    info.source = "openreview_api"

    if info.error:
        return info

    info.total_authors = len(paper["authors"])
    info.all_authors = paper["authors"]
    if not info.paper_title:
        info.paper_title = paper["title"]

    # Only "positive" decisions count as Indian-affiliated — "ambiguous"
    # authors on an otherwise-positive paper must NOT be pulled in just
    # because the paper as a whole made the cut. A collision-prone acronym
    # match on one co-author's affiliation is not evidence about them.
    positive_authors = [
        a["name"] for a, d in zip(paper["authors"], decisions)
        if d.label == "positive"
    ]
    positive_institutions = [
        a["affiliation"] for a, d in zip(paper["authors"], decisions)
        if d.label == "positive"
    ]
    # Overwrite rather than merge with the LLM's own guess — ground-truth
    # profile data is what we trust here, and the LLM's version can differ
    # in formatting (e.g. "Adamas University" vs "Adamas University, India"
    # for the same author), which set-union would keep as two entries.
    info.authors_with_indian_affiliations = sorted(set(positive_authors))
    info.indian_institutions = sorted(set(positive_institutions))

    return info


# ─────────────────────────────────────────────────────────────────────────
# Shared main loop — same accumulation/save logic regardless of source
# ─────────────────────────────────────────────────────────────────────────
def run_pipeline(
    conference: str,
    year: str,
    input_links_path: str = None,
    venue_id: str = None,
    skip_venue_keywords: list = None,
    include_only_venue_keywords: list = None,
    max_papers: int = None,
    resume_from: int = 0,
    delay: int = INTER_PAPER_DELAY
):
    """
    Exactly one of input_links_path / venue_id must be given:
        input_links_path — path to a flat links.json (tiered-scraper path)
        venue_id         — OpenReview venueid, e.g. "ICLR.cc/2026/Conference"
                            (API path; skip_venue_keywords/include_only_venue_keywords
                            filter tracks the same way select_tracks_programmatic does,
                            defaulting to skipping Workshop/Tutorial)
    """
    if (input_links_path is None) == (venue_id is None):
        raise ValueError(
            "run_pipeline requires exactly one of input_links_path or venue_id "
            f"(got input_links_path={input_links_path!r}, venue_id={venue_id!r})"
        )

    if venue_id is not None:
        papers = fetch_openreview_papers(venue_id, skip_venue_keywords, include_only_venue_keywords)
        all_urls = [p["paper_url"] for p in papers]
        work_items = papers[resume_from:]
        if max_papers:
            work_items = work_items[:max_papers]
        item_url = lambda item: item["paper_url"]
        processor = process_openreview_paper
    else:
        with open(input_links_path, "r", encoding="utf-8") as f:
            all_urls = json.load(f)
        work_items = all_urls[resume_from:]
        if max_papers:
            work_items = work_items[:max_papers]
        item_url = lambda item: item
        processor = process_paper

    log.info(f"Starting {conference.upper()} {year} — {len(work_items)} papers")
    start = datetime.now()

    out_dir = Path(f"data/final_output/{conference}/{year}")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / "indian_papers_structured.json"
    errors_file = out_dir / "errors.json"
    processed_file = out_dir / "processed_papers.json"

    # Resume support
    results = []
    errors = []
    processed_urls = set()

    if output_file.exists():
        with open(output_file) as f:
            results = json.load(f)
        processed_urls = {r["paper_url"] for r in results}
        log.info(f"Resuming — {len(results)} already processed")

    processed_records = []
    if processed_file.exists():
        with open(processed_file) as f:
            processed_records = json.load(f)
        processed_urls.update(r["paper_url"] for r in processed_records)
        log.info(f"Processed checkpoint — {len(processed_records)} total papers already attempted")

    # Consecutive block-like failures per domain — see RunBlockedError above.
    # Stays empty (never trips) for the venue_id/OpenReview-API path, since
    # that path makes no per-domain scraping requests to be blocked from.
    consecutive_block_signals: dict[str, int] = {}

    for i, item in enumerate(work_items, start=resume_from + 1):
        url = item_url(item)
        if url in processed_urls:
            log.info(f"[{i}] ⏩ skip (already done): {url}")
            continue

        log.info(f"[{i}/{resume_from + len(work_items)}] {url}")

        info = processor(item)
        domain = urlparse(url).netloc

        if info.error:
            log.warning(f"  ✗ {info.error}")
            errors.append({"url": url, "error": info.error, "index": i})
            processed_records.append({
                "paper_number": i,
                "paper_url": url,
                "status": "error",
                "error": info.error,
                "processed_at": datetime.now().isoformat(),
            })

            if _looks_like_block(info.error):
                consecutive_block_signals[domain] = consecutive_block_signals.get(domain, 0) + 1
                if consecutive_block_signals[domain] >= MAX_CONSECUTIVE_BLOCK_SIGNALS:
                    log.error(
                        f"🛑 {consecutive_block_signals[domain]} consecutive "
                        f"block-like failures from {domain} — stopping this "
                        "run now instead of continuing to hit a domain "
                        "that's likely blocking this IP. Save your progress "
                        "and wait before retrying (retry_errors will pick up "
                        "exactly where this left off)."
                    )
                    _save(output_file, errors_file, processed_file, results, errors,
                          processed_records, all_urls, start, resume_from, max_papers)
                    raise RunBlockedError(domain, consecutive_block_signals[domain], i)
            else:
                # A non-block error (bad metadata, LLM hiccup, etc.) doesn't
                # indicate the domain is blocking us — don't let it count
                # toward the breaker.
                consecutive_block_signals[domain] = 0
        elif info.authors_with_indian_affiliations:
            consecutive_block_signals[domain] = 0
            result_dict = {
                "paper_number": i,
                "paper_url": info.paper_url,
                "paper_title": info.paper_title,
                "area_of_research": info.area_of_research,
                "total_authors": info.total_authors,
                "all_authors": info.all_authors,
                "authors_with_indian_affiliations": info.authors_with_indian_affiliations,
                "indian_institutions": info.indian_institutions,
                "source": info.source,
                "processed_at": datetime.now().isoformat(),
            }
            results.append(result_dict)
            processed_records.append({
                "paper_number": i,
                "paper_url": url,
                "status": "indian_affiliated",
                "processed_at": result_dict["processed_at"],
            })
            log.info(f"  ✅ Indian authors: {info.authors_with_indian_affiliations}")
        else:
            consecutive_block_signals[domain] = 0
            processed_records.append({
                "paper_number": i,
                "paper_url": url,
                "status": "no_indian_affiliation",
                "source": info.source or info.raw_content_source,
                "processed_at": datetime.now().isoformat(),
            })
            log.info(f"  — No Indian affiliations")

        # Save after every paper — crash-safe
        _save(output_file, errors_file, processed_file, results, errors, processed_records,
              all_urls, start, resume_from, max_papers)

        # For OpenReview's negative (no-LLM) path there's no external
        # service call to be polite to, so skip the delay for those —
        # applies to the majority of papers in that mode.
        if venue_id is None or info.source != "openreview_api" or info.paper_title or info.error:
            # Jittered delay — a perfectly regular request interval is
            # itself a bot signal on top of raw volume; +/-30%
            # randomization breaks that pattern without meaningfully
            # slowing the run down.
            time.sleep(delay + random.uniform(-0.3 * delay, 0.3 * delay))

    log.info(f"Done — {len(results)} Indian-affiliated papers found")
    log.info(f"Output: {output_file.resolve()}")


def _save(output_file, errors_file, processed_file, results, errors, processed_records,
          all_urls, start, resume_from, max_papers):
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    with open(errors_file, "w") as f:
        json.dump(errors, f, indent=2)
    with open(processed_file, "w") as f:
        json.dump(processed_records, f, indent=2)
    summary = output_file.parent / "summary.json"
    with open(summary, "w") as f:
        json.dump({
            "total_input_links": len(all_urls),
            "resume_from": resume_from,
            "max_papers": max_papers,
            "indian_papers_found": len(results),
            "errors": len(errors),
            "started_at": start.isoformat(),
            "last_updated": datetime.now().isoformat(),
        }, f, indent=2)
