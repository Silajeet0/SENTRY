"""
aaai_link_extractor.py — Extracts paper links from AAAI proceedings.

AAAI's proceedings are a TWO-LEVEL structure, unlike every other grouped
conference AEGIS already supports (ACL/ACM are a single page with track
headers directly above their paper lists):

    Level 1 — aaai.org/proceeding/aaai-<N>-<year>/  (the "landing" page)
        Lists one anchor per volume/issue, e.g.:
            "Vol 40 No. 1: AAAI-26 Technical Tracks 1"
        immediately followed by a second anchor with the track's own name,
        e.g. "AAAI Technical Track on Application Domains I" — this second
        anchor points at the EXACT SAME issue page as the first (confirmed
        by inspection: clicking either lands you on the same OJS issue
        view). Only the "Vol N No. M" anchor is followed here; the
        redundant sub-anchor is intentionally skipped so each issue page
        is only fetched once.

    Level 2 — ojs.aaai.org/index.php/AAAI/issue/view/<id>  (the OJS issue
        page linked from Level 1)
        Lists one or more technical-track sections (a heading followed by
        a list of papers). Each paper's TITLE is itself a link to:
            ojs.aaai.org/index.php/AAAI/article/view/<id>
        which is the page we want — it already contains the full author
        list with each author's institutional affiliation directly below
        their name, the DOI, and the Abstract, all as plain page text (see
        screenshot of a sample article page). This is deliberately NOT the
        "PDF" button next to each paper (that points at a different,
        galley-specific sub-path and just serves the raw PDF).

Because the article-view page already contains everything the rest of the
pipeline needs as plain HTML text, no new scraper tier or LLM-prompt work
is required: `scrapers.html_scraper.HTMLScraper` already lists "aaai.org"
in HTML_FRIENDLY_DOMAINS, and that substring check also matches the
"ojs.aaai.org" subdomain the article pages actually live on — so
pipeline.process_paper() and summarizer.abstract_fetcher's abstract
retrieval both work against these URLs completely unmodified. This module
only has to produce the same grouped_links.json shape
(`[{"track_title", "track_url", "paper_links": [...]}]`) that
grouped_link_extractor.py already produces for ACL/ACM, so every
downstream step (utils.track_selector_cli/_auto, pipeline.run_pipeline)
needs no AAAI-specific changes either.

Markup note: this is written against AAAI-26's OJS instance as observed
in screenshots, using tolerant heuristics (regex on href path, keyword-
filtered heading detection) rather than hardcoded CSS classes, since exact
PKP/OJS theme markup can vary by field/class names across deployments and
this repo has no live network access to verify it byte-for-byte. If a
future AAAI year renders differently, the two functions below
(_find_volume_links / _extract_tracks_from_volume) are the only things
that should need adjusting.
"""
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from workflows.html_fetcher import USER_AGENT
from workflows.link_extractors.skip_patterns import should_skip_title

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
}

# Matches anchor text like "Vol 40 No. 1", "Vol. 40, No. 1" etc. — the ONLY
# anchors on the landing page we follow. The sub-track anchor that follows
# each one (e.g. "AAAI Technical Track on Application Domains I") does not
# start with "Vol N No. M" and is correctly skipped by this pattern.
_VOLUME_TITLE_RE = re.compile(r"^vol\.?\s*\d+\s*,?\s*no\.?\s*\d+", re.IGNORECASE)

# Matches the article "abstract/details" page — NOT its PDF/galley link,
# which has one or more extra path segments after the numeric article id
# (e.g. /article/view/36958/34567) and is excluded by the trailing "$".
_ARTICLE_VIEW_RE = re.compile(r"/article/view/(\d+)/?$")

# Sidebar/navigation/boilerplate heading text that shows up on an OJS issue
# page but is never a real technical-track heading — must not be mistaken
# for one, or a fake "track" gets created with zero real papers under it
# (harmless on its own since empty tracks are dropped, but would also
# wrongly close out the real track's link collection early).
_OJS_BOILERPLATE_HEADING_KEYWORDS = [
    "information", "for readers", "for authors", "for librarians",
    "part of the", "published", "how to cite", "make a submission",
    "developed by", "open journal systems", "privacy statement",
    "current", "archives", "about", "search", "login", "issue",
]

# Politeness delay between successive OJS issue-page fetches — a single
# AAAI proceedings can span ~48 issues (per aaai.org's own numbering), so
# this matters more here than a one-shot single-page fetch elsewhere.
DELAY_BETWEEN_VOLUME_FETCHES = 1.5

_CHALLENGE_SIGNALS = [
    "checking your browser", "just a moment", "verify you are human",
    "ddos protection by",
]


def _fetch(url: str) -> tuple[BeautifulSoup, str]:
    """Plain requests fetch — aaai.org/ojs.aaai.org are static server-
    rendered pages (this is also why HTMLScraper's Tier 1, plain requests,
    already works against them for per-paper scraping), so no Playwright
    is needed here. Returns (soup, raw_html) — raw_html kept separately so
    the on-disk cache is the server's actual bytes, not BeautifulSoup's
    re-serialization of them."""
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()

    lowered = resp.text.lower()
    if any(signal in lowered for signal in _CHALLENGE_SIGNALS):
        raise RuntimeError(
            f"Bot challenge page detected at {url} — cannot extract AAAI "
            "links via a plain HTTP request."
        )

    return BeautifulSoup(resp.text, "html.parser"), resp.text


def _find_volume_links(soup: BeautifulSoup, base_url: str) -> list[dict]:
    """Every 'Vol N No. M' anchor on the proceedings landing page, deduped
    by resolved URL (so the same issue is only ever fetched once even if
    it's linked more than once)."""
    volumes = []
    seen_urls = set()

    for a in soup.find_all("a", href=True):
        title = a.get_text(strip=True)
        if not _VOLUME_TITLE_RE.match(title):
            continue

        url = urljoin(base_url, a["href"].strip())
        if url in seen_urls:
            continue
        seen_urls.add(url)
        volumes.append({"title": title, "url": url})

    return volumes


def _is_track_heading(text: str) -> bool:
    if not text or len(text) < 3:
        return False
    if should_skip_title(text):
        return False
    lowered = text.lower()
    return not any(kw in lowered for kw in _OJS_BOILERPLATE_HEADING_KEYWORDS)


def _extract_tracks_from_volume(soup: BeautifulSoup, volume_url: str, volume_title: str) -> list[dict]:
    """
    Walks the issue page in document order. Every genuine heading tag
    (h1/h2 only — filtered through _is_track_heading) starts a new track
    bucket; every '/article/view/{id}' anchor found afterward is a paper
    title link belonging to whichever track bucket is currently open.

    Deliberately h1/h2 ONLY, not h1-h5: standard OJS/PKP theme markup
    renders the track/section heading as an h2 (class "tocSectionTitle")
    but each individual PAPER's own title as an h3 nested inside it
    (class "title"). Treating h3 as a track boundary too would wrongly
    split a single track into one (bogus) "track" per paper — confirmed
    against a synthetic page mirroring real OJS markup in
    test_aaai_extractor_offline.py. h1/h2 also naturally excludes most
    sidebar/nav boilerplate (typically h3-h5 in OJS themes), with the
    keyword filter below as a second line of defense regardless.

    Falls back to a single bucket named after the issue itself
    (volume_title) if papers appear before any recognized heading, or if
    the page has no recognizable track headings at all — mirroring
    grouped_link_extractor.py's ACM "All papers" fallback, so a markup
    surprise degrades gracefully instead of silently dropping papers.
    """
    tracks: list[dict] = []
    current_title = volume_title
    current_links: list[str] = []
    seen_in_track: set[str] = set()

    for el in soup.find_all(["h1", "h2", "a"]):
        if el.name != "a":
            heading_text = el.get_text(strip=True)
            if not _is_track_heading(heading_text):
                continue

            if current_links:
                tracks.append({
                    "track_title": current_title,
                    "track_url": volume_url,
                    "paper_links": current_links,
                })

            current_title = heading_text
            current_links = []
            seen_in_track = set()
            continue

        href = (el.get("href") or "").strip()
        if not href:
            continue

        full_url = urljoin(volume_url, href)
        if not _ARTICLE_VIEW_RE.search(urlparse(full_url).path):
            continue

        if full_url in seen_in_track:
            continue
        seen_in_track.add(full_url)
        current_links.append(full_url)

    if current_links:
        tracks.append({
            "track_title": current_title,
            "track_url": volume_url,
            "paper_links": current_links,
        })

    return tracks


def extract_aaai_links(proceeding_url: str, conference: str, year: str) -> str:
    """
    Entry point used by main_driver.run_pipeline (mirrors
    acm_api_fetcher.fetch_acm_links's role for ACM DL).

    Fetches the AAAI proceedings landing page, follows every 'Vol N No. M'
    volume link to its OJS issue page, extracts {track_title, track_url,
    paper_links} from each, and saves the combined result as
    grouped_links.json — the exact same shape utils.track_selector_cli /
    utils.track_selector_auto already know how to flatten into a final
    links.json for pipeline.run_pipeline.

    Returns the absolute path to the saved grouped_links.json.
    """
    log.info(f"[AAAI] Fetching proceedings landing page: {proceeding_url}")
    root_soup, root_html = _fetch(proceeding_url)

    html_dir = Path(f"data/html/{conference}/{year}")
    html_dir.mkdir(parents=True, exist_ok=True)
    (html_dir / "proceedings_root.html").write_text(root_html, encoding="utf-8")

    volumes = _find_volume_links(root_soup, proceeding_url)
    if not volumes:
        raise RuntimeError(
            f"No 'Vol N No. M' volume/issue links found on {proceeding_url} "
            "— the AAAI proceedings landing page markup may have changed. "
            "Check _VOLUME_TITLE_RE / _find_volume_links in "
            "workflows/link_extractors/aaai_link_extractor.py."
        )
    log.info(f"[AAAI] Found {len(volumes)} volume/issue page(s) to fetch.")

    grouped_data: list[dict] = []
    for i, volume in enumerate(volumes, start=1):
        log.info(f"[AAAI] ({i}/{len(volumes)}) {volume['title']} -> {volume['url']}")
        try:
            volume_soup, volume_html = _fetch(volume["url"])
        except Exception as e:
            log.warning(f"[AAAI] Skipping {volume['url']} — fetch failed: {e}")
            continue

        safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", volume["title"]).strip("_").lower()
        (html_dir / f"volume_{safe_name}.html").write_text(volume_html, encoding="utf-8")

        tracks = _extract_tracks_from_volume(volume_soup, volume["url"], volume["title"])
        if not tracks:
            log.warning(f"[AAAI] No paper links found on {volume['url']}")
        grouped_data.extend(tracks)

        if i < len(volumes):
            time.sleep(DELAY_BETWEEN_VOLUME_FETCHES)

    if not grouped_data:
        raise RuntimeError(
            f"No paper links extracted from any of the {len(volumes)} AAAI "
            f"volume/issue pages under {proceeding_url}. Check "
            "_extract_tracks_from_volume in "
            "workflows/link_extractors/aaai_link_extractor.py against the "
            "current OJS markup."
        )

    save_dir = Path(f"data/links_raw/{conference}/{year}")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "grouped_links.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(grouped_data, f, indent=2, ensure_ascii=False)

    total_links = sum(len(track["paper_links"]) for track in grouped_data)
    print(
        f"[INFO] ✅ Extracted {total_links} paper links across "
        f"{len(grouped_data)} track(s) from {len(volumes)} AAAI volume(s)."
    )
    print(f"[INFO] 📁 Saved grouped links to: {save_path.resolve()}")

    return str(save_path.resolve())
