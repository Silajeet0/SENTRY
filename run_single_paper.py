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
from pipeline import process_paper


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

        # ACM DL needs a fresh Cloudflare session before scraping — warm up
        # cookies here if they're missing or stale. This only affects ACM
        # URLs; all other URLs (OpenReview, IEEE, NeurIPS, etc.) skip this
        # block entirely and go straight to process_paper() below.
        if "dl.acm.org" in args.url:
            from pathlib import Path
            import time

            cookie_path = Path("data/acm_session_cookies.json")
            cookies_fresh = (
                cookie_path.exists()
                and (time.time() - cookie_path.stat().st_mtime) / 60 < 30
            )

            if not cookies_fresh:
                print("[INFO] No fresh ACM cookies — warming up via proceedings page...")
                import re
                match = re.search(r"10\.1145/(\d+)\.\d+", args.url)
                if match:
                    from workflows.link_extractors.acm_api_fetcher import warmup_acm_cookies
                    proceedings_url = f"https://dl.acm.org/doi/proceedings/10.1145/{match.group(1)}"
                    try:
                        warmup_acm_cookies(proceedings_url)
                        print("[INFO] ACM cookies refreshed.")
                    except Exception as e:
                        print(f"[WARN] Cookie warmup failed: {e} — scraping may fall through to SS")
            else:
                print("[INFO] Using existing ACM cookies.")

        # Process the paper — runs for every URL, ACM or otherwise.
        # This was previously nested inside the ACM-only `if` block above,
        # which meant non-ACM URLs (including OpenReview) never reached this
        # line and `info` was left unbound, causing the crash at print() time.
        info = process_paper(args.url)

    print(json.dumps(asdict(info), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
