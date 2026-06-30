# main_driver.py
import time
from pathlib import Path
from workflows.html_fetcher import fetch_and_save_html
from workflows.track_detector import is_track_grouped
from workflows.link_extractors.flat_link_extractor import extract_flat_links_with_base
from workflows.link_extractors.grouped_link_extractor import extract_grouped_links
from workflows.link_extractors.acm_api_fetcher import fetch_acm_links, COOKIE_PATH
from workflows.link_extractors.openreview_fetcher import (
    fetch_openreview_links,
    warmup_openreview_cookies,
    OPENREVIEW_COOKIE_PATH,
    OPENREVIEW_CONFERENCE_CONFIG,
)
from utils.track_selector_cli import select_tracks_cli
from pipeline import run_pipeline as run_paper_pipeline

ACM_COOKIE_MAX_AGE_MINUTES = 30
OPENREVIEW_COOKIE_MAX_AGE_MINUTES = 30

# Conferences hosted on OpenReview — detected by conference name prefix.
# Link extraction loads the group page in Playwright (the API itself returns
# a ChallengeRequiredError for unauthenticated programmatic access).
OPENREVIEW_CONFERENCES = {
    "ICLR", "ICML", "NEURIPS", "ICLR_ORAL", "ICLR_SPOTLIGHT",
    "ICML_ORAL", "ICML_SPOTLIGHT",
}


def run_pipeline(
    proceeding_url: str,
    conference: str,
    year: str,
    max_papers: int = None,
    resume_from: int = 0,
    delay: int = 10
):
    print(f"\n🚀 Running pipeline for {conference.upper()} {year}")

    # ------------------------------------------------------------------
    # OpenReview-hosted conferences (ICLR, ICML, NeurIPS on OpenReview)
    # Loads the group page in Playwright (headless=False — the challenge
    # wall blocks headless Chromium). Session cookies are saved during link
    # extraction and reused by browser_scraper.py for per-paper PDF downloads.
    #
    # Same staleness check as ACM: if cached links exist but cookies are
    # stale, re-warm the session before per-paper scraping starts, so a
    # resumed run doesn't hit the challenge wall cold.
    #
    # To use: run_pipeline(
    #     proceeding_url="",   # unused for OpenReview
    #     conference="ICLR",
    #     year="2025",
    # )
    # ------------------------------------------------------------------
    if conference.upper().split("_")[0] in OPENREVIEW_CONFERENCES:
        print(f"[🔎] OpenReview conference detected — using Playwright.")

        links_json_path = Path(f"data/links_raw/{conference}/{year}/grouped_links.json")
        links_already_exist = links_json_path.exists()

        cookies_are_stale = True
        if OPENREVIEW_COOKIE_PATH.exists():
            age_minutes = (time.time() - OPENREVIEW_COOKIE_PATH.stat().st_mtime) / 60
            cookies_are_stale = age_minutes > OPENREVIEW_COOKIE_MAX_AGE_MINUTES
            if not cookies_are_stale:
                print(f"[✅] OpenReview session cookies fresh ({age_minutes:.0f}m old).")

        if not links_already_exist:
            grouped_json_path = fetch_openreview_links(conference, year)
        elif cookies_are_stale:
            print(
                "[⚠️] OpenReview session cookies stale — re-visiting group "
                "page to refresh challenge clearance."
            )
            key = f"{conference.upper()}_{year}"
            venue_id = OPENREVIEW_CONFERENCE_CONFIG.get(key, {}).get(
                "venue_id", f"{conference}.cc/{year}/Conference"
            )
            warmup_openreview_cookies(venue_id)
            grouped_json_path = str(links_json_path.resolve())
        else:
            print("[✅] Using cached OpenReview links.")
            grouped_json_path = str(links_json_path.resolve())

        links_json_path = select_tracks_cli(grouped_json_path)

    # ------------------------------------------------------------------
    # ACM DL conferences — uses Playwright to bypass Cloudflare.
    # The proceedings page visit clears Cloudflare and saves session
    # cookies so per-paper scraping doesn't re-challenge.
    #
    # Cookie freshness check: if cookies are stale (>30min), re-run
    # link extraction to refresh them before per-paper scraping starts.
    # This ensures resumed runs don't hit Cloudflare cold.
    #
    # Covers all A* ACM conferences:
    # KDD, SIGMOD, SIGIR, SIGCOMM, STOC, FOCS, SOSP, OSDI, CCS,
    # SIGGRAPH, CHI, PLDI, ASPLOS, ICSE, FSE, WWW, CSCW, UIST, PODC
    # ------------------------------------------------------------------
    elif "dl.acm.org" in proceeding_url:
        print("[🔎] ACM DL proceedings detected — using Playwright.")

        links_json_path = Path(f"data/links_raw/{conference}/{year}/grouped_links.json")
        links_already_exist = links_json_path.exists()

        cookies_are_stale = True
        if COOKIE_PATH.exists():
            age_minutes = (time.time() - COOKIE_PATH.stat().st_mtime) / 60
            cookies_are_stale = age_minutes > ACM_COOKIE_MAX_AGE_MINUTES
            if not cookies_are_stale:
                print(f"[✅] ACM session cookies fresh ({age_minutes:.0f}m old).")

        if not links_already_exist or cookies_are_stale:
            if cookies_are_stale and links_already_exist:
                print(
                    f"[⚠️] ACM session cookies stale — re-visiting proceedings "
                    f"page to refresh Cloudflare clearance."
                )
            grouped_json_path = fetch_acm_links(proceeding_url, conference, year)
        else:
            print("[✅] Using cached ACM links.")
            grouped_json_path = str(links_json_path.resolve())

        links_json_path = select_tracks_cli(grouped_json_path)

    # ------------------------------------------------------------------
    # All other conferences — existing html_fetcher flow
    # (IEEE, ACL, EMNLP, NAACL, ICML via MLR Press, etc.)
    # ------------------------------------------------------------------
    else:
        html_path = fetch_and_save_html(proceeding_url, conference, year)

        if is_track_grouped(conference):
            print("[🔎] Track-based structure detected.")
            grouped_json_path = extract_grouped_links(html_path, conference, year)
            links_json_path = select_tracks_cli(grouped_json_path)
        else:
            print("[🔎] Flat structure detected.")
            extract_flat_links_with_base(conference, html_path, year)
            links_json_path = Path(f"data/links_raw/{conference}/{year}/links.json")

    # ------------------------------------------------------------------
    # Per-paper processing — same for all conference types
    # ------------------------------------------------------------------
    run_paper_pipeline(
        conference=conference,
        year=year,
        input_links_path=str(links_json_path),
        max_papers=max_papers,
        resume_from=resume_from,
        delay=delay
    )

    print(f"✅ Pipeline completed for {conference.upper()} {year}\n")
