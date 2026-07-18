# run_selenium_filler.py
"""
Loads a conference/year's `indian_papers_structured.json`, dedup-checks
every candidate title against IKDD (New + Approved), and Selenium-submits
only the genuinely new ones.

Two ways to use this module:

1. Programmatically — `run_form_filler(...)` is the entry point the
   orchestrator's RPA tool calls (see orchestrator/rpa_runner.py). It takes
   conference/year/venue/month as plain arguments and returns a summary
   dict rather than just printing, so a caller can report real numbers back
   to whoever asked for the run.

2. As a script — `python Form_filler/run_selenium_filler.py --conference
   NeurIPS --year 2025 --month Dec --venue NeurIPS` for a manual/local run.
   All four are required on the CLI; there is no hardcoded default
   conference/year/month/venue anymore; venue/month must match the form's
   dropdown text exactly, or Selenium's select_by_visible_text will fail
   loudly on the mismatch, so it can't be guessed here.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]

# Make sure `utils` (which lives at REPO_ROOT/utils) is importable regardless
# of the current working directory this script is launched from, and
# regardless of whether this file is imported as `Form_filler.run_selenium_filler`
# (by the orchestrator) or run directly as a script.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# selenium_filler lives alongside this file. When the orchestrator imports
# this module as `Form_filler.run_selenium_filler`, __package__ is set
# ("Form_filler") and the package-relative import is what works; when this
# file is run directly as a script (`python Form_filler/run_selenium_filler.py`),
# __package__ is empty and Python puts Form_filler/ itself on sys.path[0]
# instead, so the bare top-level import is what works there.
#
# Branching on __package__ (rather than try/except ImportError on the first
# form) matters: selenium_filler.py itself does `from selenium import
# webdriver` at module load time, which also raises ImportError if selenium
# isn't installed — catching ImportError broadly here would misreport a
# genuine "selenium isn't installed" failure as an unrelated fallback
# import error instead of surfacing the real, actionable one.
if __package__:
    from .selenium_filler import process_papers_with_selenium
else:
    from selenium_filler import process_papers_with_selenium

from utils.ikdd_dedup import IKDDDeduplicator, refresh_cache, load_cache

DEFAULT_FORM_URL = "https://ikdd.hosting.acm.org/ds-papers-form.php"


def get_dedup_cache(refresh: bool = True) -> Optional[dict]:
    """
    Returns a fresh (or cached) IKDD approved-titles cache, with a sensible
    fallback if a live refresh isn't possible.
    """
    if refresh:
        try:
            print("🔍 Refreshing IKDD approved-papers cache (scraping latest list)...")
            return refresh_cache()
        except Exception as e:
            print(f"⚠️  Could not refresh IKDD cache live: {e}")
            print("   Falling back to local cache, if one exists...")

    try:
        return load_cache()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return None


def run_form_filler(
    conference: str,
    year: str,
    month: str,
    venue: str,
    form_url: str = None,
    refresh_dedup_cache: bool = True,
) -> dict:
    """
    Programmatic entry point — dedup-checks `conference`/`year`'s extracted
    papers against IKDD, then Selenium-submits only the new ones.

    Args:
        conference: e.g. "NeurIPS" — must match the folder name under
            data/final_output/<conference>/<year>/.
        year: e.g. "2025".
        month: EXACT text of the form's month dropdown option, e.g. "Nov".
        venue: EXACT text of the form's venue dropdown option, e.g. "ICDM".
        form_url: overrides DEFAULT_FORM_URL if given.
        refresh_dedup_cache: if True (default), re-scrapes IKDD's New +
            Approved lists before checking. If False, uses whatever is
            already cached on disk (faster, but may be stale).

    Returns a summary dict:
        {
            "conference": ..., "year": ...,
            "total_candidates": int,
            "duplicates_skipped": int,
            "submitted": int,
            "failed": int,
            "duplicate_details": [...],
            "details": [...],   # per-paper submit/fail results
        }
    Raises FileNotFoundError if the conference/year's structured JSON
    doesn't exist, and RuntimeError if no dedup cache is available at all
    (neither a live refresh nor a local cache) — both are treated as hard
    stops rather than silently skipping the dedup check, since submitting
    without it risks duplicate entries on IKDD.
    """
    year = str(year)
    form_url = form_url or DEFAULT_FORM_URL

    print("--- Starting Selenium Form Filling Process ---")

    # 1. Locate and load the structured papers JSON for this conference/year
    json_path = REPO_ROOT / f"data/final_output/{conference}/{year}/indian_papers_structured.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Could not find extracted-papers JSON for {conference} {year} "
            f"(checked {json_path}). Run the AEGIS extraction pipeline for "
            "this conference/year first."
        )

    print(f"✅ Found data file: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        papers_data = json.load(f)

    if not papers_data:
        print("ℹ️ The JSON file is empty. Nothing to process.")
        return {
            "conference": conference, "year": year,
            "total_candidates": 0, "duplicates_skipped": 0,
            "submitted": 0, "failed": 0,
            "duplicate_details": [], "details": [],
        }

    # 2. Dedup step — check candidates against IKDD's New + Approved lists
    #    BEFORE opening a browser or touching the form at all.
    cache = get_dedup_cache(refresh=refresh_dedup_cache)
    if cache is None:
        raise RuntimeError(
            "No IKDD approved-titles cache available (live refresh failed "
            "and no local cache exists). Run "
            "`python -m utils.ikdd_dedup --refresh` first, or fix "
            "IKDD_USERNAME/IKDD_PASSWORD in .env."
        )

    dedup = IKDDDeduplicator(cache=cache)
    new_papers, duplicates = dedup.filter_new(papers_data)

    print(f"\n📊 Dedup results: {len(papers_data)} total | "
          f"{len(new_papers)} new | {len(duplicates)} already in IKDD")

    duplicate_details = [
        {
            "paper_title": p.get("paper_title", ""),
            "score": p.get("_dedup_score", 0),
            "matched_title": p.get("_dedup_matched", ""),
        }
        for p in duplicates
    ]
    if duplicates:
        print("   Skipping the following (already present in IKDD):")
        for d in duplicate_details:
            print(f"     [{d['score']:.2f}] {d['paper_title']} → matched '{d['matched_title']}'")

    if not new_papers:
        print("\n✅ Nothing to submit — every candidate paper is already in IKDD.")
        return {
            "conference": conference, "year": year,
            "total_candidates": len(papers_data),
            "duplicates_skipped": len(duplicates),
            "submitted": 0, "failed": 0,
            "duplicate_details": duplicate_details, "details": [],
        }

    # 3. Fill and submit only the genuinely new papers.
    form_config = {"form_url": form_url, "venue": venue, "year": year, "month": month}
    print(f"\n🚀 Proceeding to fill and submit {len(new_papers)} new paper(s)...")
    details = process_papers_with_selenium(new_papers, form_config)

    submitted = sum(1 for d in details if d["status"] == "submitted")
    failed = sum(1 for d in details if d["status"] == "failed")

    return {
        "conference": conference, "year": year,
        "total_candidates": len(papers_data),
        "duplicates_skipped": len(duplicates),
        "submitted": submitted, "failed": failed,
        "duplicate_details": duplicate_details, "details": details,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Dedup-check and Selenium-submit a conference/year's extracted papers to IKDD."
    )
    parser.add_argument("--conference", required=True, help="e.g. NeurIPS, IEEE-ICDM")
    parser.add_argument("--year", required=True, help="e.g. 2025")
    parser.add_argument("--month", required=True, help="Exact form dropdown text, e.g. Nov")
    parser.add_argument("--venue", required=True, help="Exact form dropdown text, e.g. ICDM")
    parser.add_argument("--form-url", default=DEFAULT_FORM_URL)
    parser.add_argument(
        "--no-refresh", action="store_true",
        help="Use the local dedup cache instead of re-scraping IKDD live.",
    )
    args = parser.parse_args()

    result = run_form_filler(
        conference=args.conference,
        year=args.year,
        month=args.month,
        venue=args.venue,
        form_url=args.form_url,
        refresh_dedup_cache=not args.no_refresh,
    )

    print(f"\n{'─'*50}")
    print(f"  Total candidates   : {result['total_candidates']}")
    print(f"  Skipped (dupes)    : {result['duplicates_skipped']}")
    print(f"  Submitted          : {result['submitted']}")
    print(f"  Failed             : {result['failed']}")
    print(f"{'─'*50}")


if __name__ == "__main__":
    main()
