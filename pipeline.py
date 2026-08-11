"""
Main pipeline
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

# Block detection

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
    """Raised when a domain trips the circuit breaker ."""

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

# Tiered-scraper path (HTML/IEEE/ACM/ACL/etc via a flat list of URLs)

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

    return not _has_positive_pdf_affiliation_evidence(content)


def _has_positive_pdf_affiliation_evidence(content: str) -> bool:
    """
    PDF text is a page-level blob, not a single affiliation. Checking the
    whole blob can accidentally veto valid local evidence such as "IIT
    Kanpur" because another author on the same first page has "Singapore" in
    their affiliation. Prefer local lines/windows, with the original whole-
    content check kept as a fast positive path.
    """
    if classify_affiliation(content).label == "positive":
        return True

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    for line in lines:
        if classify_affiliation(line).label == "positive":
            return True

    for window_size in (2, 3):
        for start in range(len(lines) - window_size + 1):
            window = " ".join(lines[start:start + window_size])
            if classify_affiliation(window).label == "positive":
                return True

    return False


# OpenReview API path (no scraping tiers — data already structured)

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
    decisions = [classify_affiliation(a["affiliation"]) for a in paper["authors"]]
    overall_label = combine_author_decisions(decisions)

    if overall_label in ("negative", "ambiguous"):
        return PaperInfo(
            paper_url=paper["paper_url"],
            raw_content_source="openreview_api",
            source="openreview_api",
        )

    content = _build_openreview_llm_content(paper)
    info = EXTRACTOR.extract(content, paper["paper_url"], content_source="openreview_api")
    info.source = "openreview_api"

    if info.error:
        return info

    info.total_authors = len(paper["authors"])
    info.all_authors = paper["authors"]
    if not info.paper_title:
        info.paper_title = paper["title"]

    positive_authors = [
        a["name"] for a, d in zip(paper["authors"], decisions)
        if d.label == "positive"
    ]
    positive_institutions = [
        a["affiliation"] for a, d in zip(paper["authors"], decisions)
        if d.label == "positive"
    ]

    info.authors_with_indian_affiliations = sorted(set(positive_authors))
    info.indian_institutions = sorted(set(positive_institutions))

    return info


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
                "area_of_research_other": info.area_of_research_other,
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

        if venue_id is None or info.source != "openreview_api" or info.paper_title or info.error:
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
