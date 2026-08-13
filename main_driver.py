# main_driver.py
from pathlib import Path
from typing import Callable, Optional
from workflows.html_fetcher import fetch_and_save_html
from workflows.track_detector import is_track_grouped
from workflows.link_extractors.flat_link_extractor import extract_flat_links_with_base
from workflows.link_extractors.grouped_link_extractor import extract_grouped_links
from workflows.link_extractors.acm_api_fetcher import fetch_acm_links
from workflows.link_extractors.aaai_link_extractor import extract_aaai_links

from utils.track_selector_cli import select_tracks_cli
from utils.track_selector_auto import select_tracks_auto
from pipeline import run_pipeline as run_paper_pipeline

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
        selection for grouped conferences blocks on a CLI prompt. Set False to select tracks automatically via
        track_skip_keywords / track_include_keywords instead — this is what
        the agentic orchestrator uses so a run never stalls waiting on stdin
        that's never coming.

    Returns the resolved links.json path that was fed into per-paper
    processing.
    """
    print(f"\nRunning pipeline for {conference.upper()} {year}")

    def _select_tracks(grouped_path: str) -> str:
        if interactive:
            return select_tracks_cli(grouped_path)
        result = select_tracks_auto(
            grouped_path,
            skip_keywords=track_skip_keywords,
            include_keywords=track_include_keywords,
        )
        print(
            f"[INFO] Auto-selected {len(result['selected_tracks'])}/"
            f"{result['total_tracks']} tracks ({result['total_links']} links)"
            f" — skipped: {result['skipped_tracks'] or 'none'}"
        )
        return result["path"]


    if "dl.acm.org" in proceeding_url:
        print("ACM DL proceedings detected — using Playwright.")

        links_json_path = Path(f"data/links_raw/{conference}/{year}/grouped_links.json")

        if links_json_path.exists():
            print("Using cached ACM links.")
            grouped_json_path = str(links_json_path.resolve())
        else:
            grouped_json_path = fetch_acm_links(proceeding_url, conference, year)

        links_json_path = _select_tracks(grouped_json_path)

    elif "aaai" in conference.lower() or "aaai.org" in proceeding_url:
        print("AAAI proceedings detected — using two-level volume/track extractor.")

        links_json_path = Path(f"data/links_raw/{conference}/{year}/grouped_links.json")

        if links_json_path.exists():
            print("Using cached AAAI links.")
            grouped_json_path = str(links_json_path.resolve())
        else:
            grouped_json_path = extract_aaai_links(proceeding_url, conference, year)

        links_json_path = _select_tracks(grouped_json_path)


    else:
        html_path = fetch_and_save_html(proceeding_url, conference, year)

        if is_track_grouped(conference, proceeding_url):
            print("Track-based structure detected.")
            grouped_json_path = extract_grouped_links(html_path, conference, year, proceeding_url)
            links_json_path = _select_tracks(grouped_json_path)
        else:
            print("Flat structure detected.")
            extract_flat_links_with_base(conference, html_path, year)
            links_json_path = Path(f"data/links_raw/{conference}/{year}/links.json")

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

    print(f"Pipeline completed for {conference.upper()} {year}\n")

    return links_json_path
