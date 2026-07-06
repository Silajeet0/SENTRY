"""
Main pipeline: drop-in replacement for agent_caller.py.

Architecture:
    URL → Tier1(HTML) → Tier2(Browser) → Tier3(PDF) → Tier4(API)
                                                            ↓
                                                    LLM Extractor (single call)
                                                            ↓
                                                    PaperInfo (structured JSON)

No agent loop. No planning. Fully deterministic.
Each tier tries once and either succeeds or passes to the next.
LLM is called exactly once per paper.
"""
import json
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
from evaluation.india_rules import classify_affiliation

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


def run_pipeline(
    conference: str,
    year: str,
    input_links_path: str,
    max_papers: int = None,
    resume_from: int = 0,
    delay: int = INTER_PAPER_DELAY
):
    """
    Drop-in replacement for the original run_agent_analysis().
    Same signature, same output structure, no Agent-E dependency.
    """
    with open(input_links_path, "r", encoding="utf-8") as f:
        all_urls = json.load(f)

    paper_urls = all_urls[resume_from:]
    if max_papers:
        paper_urls = paper_urls[:max_papers]

    log.info(f"Starting {conference.upper()} {year} — {len(paper_urls)} papers")
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

    for i, url in enumerate(paper_urls, start=resume_from + 1):
        if url in processed_urls:
            log.info(f"[{i}] ⏩ skip (already done): {url}")
            continue

        log.info(f"[{i}/{resume_from + len(paper_urls)}] {url}")

        info = process_paper(url)

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
        elif info.authors_with_indian_affiliations:
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

        time.sleep(delay)

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
