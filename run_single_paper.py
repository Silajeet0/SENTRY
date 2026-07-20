"""
Run the extraction pipeline for one paper URL, or run a synthetic Indian-affiliation
smoke test through the same LLM extractor.
"""
import argparse
import json
from dataclasses import asdict

from dotenv import load_dotenv

load_dotenv()

from extractors.llm_extractor import LLMExtractor
from pipeline import process_paper, process_openreview_paper
from workflows.link_extractors.openreview_api_fetcher import fetch_single_openreview_paper


SYNTHETIC_INDIAN_CONTENT = """
Title: Contrastive Retrieval Augmented In-Context Learning for Medical Classification Tasks with Imbalanced Data
Author: Swarnika Joshi | Affiliation: Scottish High International School
Author: Kshitij Jadhav | Affiliation: Indian Institute of Technology Bombay, Mumbai, India

Abstract:
This paper studies retrieval augmented in-context learning for medical classification tasks with imbalanced data.
"""


def main():
    parser = argparse.ArgumentParser(description="Run AEGIS on a single paper.")
    parser.add_argument("url", nargs="?", help="Paper URL to process, e.g. an IEEE Xplore document URL.")
    parser.add_argument(
        "--synthetic-indian",
        action="store_true",
        help="Run a no-network smoke test with a known IIT Bombay affiliation.",
    )
    args = parser.parse_args()

    if args.synthetic_indian:
        info = LLMExtractor().extract(
            SYNTHETIC_INDIAN_CONTENT,
            "synthetic://icdm-2025-indian-affiliation-smoke-test",
            "synthetic",
        )
    else:
        if not args.url:
            parser.error("provide a paper URL or use --synthetic-indian")

        # OpenReview URLs go through the ground-truth API path (fetch note +
        # author profiles, classify affiliation from real institution/country
        # data, LLM only called for area_of_research on positive/ambiguous
        # papers) — NOT process_paper()'s tiered scrapers. browser_scraper.py
        # (Tier 2) no longer even claims openreview.net URLs at all; this
        # branch is the only correct way to process one.
        if "openreview.net" in args.url:
            paper = fetch_single_openreview_paper(args.url)
            info = process_openreview_paper(paper)

        else:
            # ACM DL cookie freshness is no longer pre-checked here on a
            # wall-clock timer — process_paper() below (via
            # scrapers/browser_scraper.py's BrowserScraper) tries with
            # whatever session cookies are on disk first, and only refreshes
            # them itself, reactively, if it actually gets a Cloudflare
            # challenge back. That's strictly cheaper when the existing
            # cookies are still good, and no less correct when they aren't.

            # Process the paper via the tiered scrapers — runs for every URL
            # that isn't OpenReview (ACM, IEEE, NeurIPS, ACL, etc). Kept
            # outside the ACM-only `if` above (not nested inside it) so
            # non-ACM URLs actually reach this line — nesting it there was
            # the original unbound-`info` bug.
            info = process_paper(args.url)

    print(json.dumps(asdict(info), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
