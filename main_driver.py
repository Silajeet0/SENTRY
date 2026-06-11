# main_driver.py (updated)
from pathlib import Path
from workflows.html_fetcher import fetch_and_save_html
from workflows.track_detector import is_track_grouped
from workflows.link_extractors.flat_link_extractor import extract_flat_links_with_base
from workflows.link_extractors.grouped_link_extractor import extract_grouped_links
from utils.track_selector_cli import select_tracks_cli
from pipeline import run_pipeline as run_paper_pipeline   # ← changed

def run_pipeline(proceeding_url: str, conference: str, year: str, max_papers: int = None, resume_from: int = 0, delay: int = 10):
    print(f"\n🚀 Running pipeline for {conference.upper()} {year}")

    html_path = fetch_and_save_html(proceeding_url, conference, year)

    if is_track_grouped(conference):
        print("[🔎] Track-based structure detected.")
        grouped_json_path = extract_grouped_links(html_path, conference, year)
        links_json_path = select_tracks_cli(grouped_json_path)
    else:
        print("[🔎] Flat structure detected.")
        extract_flat_links_with_base(conference, html_path, year)
        links_json_path = Path(f"data/links_raw/{conference}/{year}/links.json")

    run_paper_pipeline(                          # ← changed
        conference=conference,
        year=year,
        input_links_path=str(links_json_path),
        max_papers=max_papers,
        resume_from=resume_from,
        delay=delay
    )

    print(f"✅ Pipeline completed for {conference.upper()} {year}\n")