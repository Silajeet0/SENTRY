# main_driver.py
import time
from pathlib import Path
from typing import Callable, Optional
from workflows.html_fetcher import fetch_and_save_html
from workflows.track_detector import is_track_grouped
from workflows.link_extractors.flat_link_extractor import extract_flat_links_with_base
from workflows.link_extractors.grouped_link_extractor import extract_grouped_links
from workflows.link_extractors.acm_api_fetcher import fetch_acm_links, COOKIE_PATH

from utils.track_selector_cli import select_tracks_cli
from utils.track_selector_auto import select_tracks_auto
from pipeline import run_pipeline as run_paper_pipeline

ACM_COOKIE_MAX_AGE_MINUTES = 30
OPENREVIEW_COOKIE_MAX_AGE_MINUTES = 30

# NOTE: kept only for backward compatibility with any old code still
# importing this — OpenReview conferences (ICLR/ICML) are no longer routed
# through this function at all. They go straight to pipeline.run_pipeline's
# venue_id/API path (see orchestrator/conference_catalog.py), which needs
# no proceeding_url, no HTML fetch, no browser. Nothing below this line
# branches on this set — do not rely on it for routing.
OPENREVIEW_CONFERENCES = {
    "ICLR", "ICML", "ICLR_ORAL", "ICLR_SPOTLIGHT",
    "ICML_ORAL", "ICML_SPOTLIGHT",
}


def run_pipeline(
    proceeding_url: str,
    conference: str,
    year: str,
    max_papers: int = None,
    resume_from: int = 0,
    delay: int = 10,
    interactive: bool = True,
    track_skip_keywords: Optional[list[str]] = None,
    track_include_keywords: Optional[list[str]] = None,
    on_links_ready: Optional[Callable[[str], None]] = None,
) -> str:
    """
    interactive: when True (default — unchanged from before), track
        selection for grouped conferences blocks on a CLI prompt exactly
        like it always has. Set False to select tracks automatically via
        track_skip_keywords / track_include_keywords instead — this is what
        the agentic orchestrator uses so a run never stalls waiting on stdin
        that's never coming.
    on_links_ready: optional callback invoked with the final links.json path
        as soon as it's known, before the (long) per-paper loop starts. Lets
        a caller record the path for later use (e.g. retrying just the
        failed papers) even if the per-paper loop is later interrupted.

    Returns the resolved links.json path that was fed into per-paper
    processing.
    """
    print(f"\n🚀 Running pipeline for {conference.upper()} {year}")

    def _select_tracks(grouped_path: str) -> str:
        if interactive:
            return select_tracks_cli(grouped_path)
        result = select_tracks_auto(
            grouped_path,
            skip_keywords=track_skip_keywords,
            include_keywords=track_include_keywords,
        )
        print(
            f"[INFO] 🤖 Auto-selected {len(result['selected_tracks'])}/"
            f"{result['total_tracks']} tracks ({result['total_links']} links)"
            f" — skipped: {result['skipped_tracks'] or 'none'}"
        )
        return result["path"]


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
    if "dl.acm.org" in proceeding_url:
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

        links_json_path = _select_tracks(grouped_json_path)

    # ------------------------------------------------------------------
    # All other conferences — existing html_fetcher flow
    # (IEEE, ACL, EMNLP, NAACL, etc. — NOT ICML/ICLR, which bypass this
    # function entirely via pipeline.run_pipeline's venue_id path)
    # ------------------------------------------------------------------
    else:
        html_path = fetch_and_save_html(proceeding_url, conference, year)

        if is_track_grouped(conference):
            print("[🔎] Track-based structure detected.")
            grouped_json_path = extract_grouped_links(html_path, conference, year)
            links_json_path = _select_tracks(grouped_json_path)
        else:
            print("[🔎] Flat structure detected.")
            extract_flat_links_with_base(conference, html_path, year)
            links_json_path = Path(f"data/links_raw/{conference}/{year}/links.json")

    # ------------------------------------------------------------------
    # Per-paper processing — same for all conference types
    # ------------------------------------------------------------------
    links_json_path = str(links_json_path)

    if on_links_ready:
        on_links_ready(links_json_path)

    run_paper_pipeline(
        conference=conference,
        year=year,
        input_links_path=links_json_path,
        max_papers=max_papers,
        resume_from=resume_from,
        delay=delay
    )

    print(f"✅ Pipeline completed for {conference.upper()} {year}\n")

    return links_json_path
