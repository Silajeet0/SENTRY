"""
acm_api_fetcher.py — Fetches paper links for ACM proceedings via Playwright.

Opens the ACM DL proceedings page in a non-headless browser (required to
pass Cloudflare Turnstile), clicks the Research Track section, extracts all
paper DOI links, and saves session cookies for reuse by browser_scraper.py.

Supported conferences (all A* under ACM):
    KDD, SIGMOD, SIGIR, SIGCOMM, STOC, FOCS, PODC, SOSP, OSDI,
    CCS, SIGGRAPH, CHI, PLDI, ASPLOS, ICSE, FSE, WWW, CSCW, UIST
"""

import re
import json
import logging
from pathlib import Path
from playwright.sync_api import sync_playwright

log = logging.getLogger(__name__)

# Where session cookies are saved for browser_scraper.py to reuse
COOKIE_PATH = Path("data/acm_session_cookies.json")


def _fetch_research_track_links(proceeding_url: str) -> tuple[list[str], list[dict]]:
    """
    Open ACM proceedings page (non-headless to pass Cloudflare),
    expand Research Track, extract DOI URLs, and return links + cookies.

    Returns:
        (paper_links, cookies) — sorted list of DOI URLs and session cookies
    """
    cookies = []

    with sync_playwright() as p:
        # Must be headless=False — Cloudflare Turnstile detects and blocks
        # headless Chromium. The proceedings page visit clears Cloudflare
        # for the entire session, so per-paper scraping inherits the clearance.
        browser = p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"]
        )

        context = browser.new_context(
            viewport={"width": 1920, "height": 1080}
        )

        page = context.new_page()

        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)

        page.goto(proceeding_url, timeout=120000)

        page.wait_for_function(
            "() => document.title.includes('Proceedings')",
            timeout=120000
        )

        log.info("Proceedings page loaded.")
        log.info(f"Title: {page.title()}")

        # Click the Research Track section to expand its paper list
        try:
            research_track = page.get_by_text("SESSION: Research Track", exact=True)
            research_track.wait_for(timeout=30000)
            research_track.click()
            log.info("Clicked SESSION: Research Track")
        except Exception as e:
            log.warning(f"Could not click Research Track: {e} — extracting all visible links")

        # Wait for links to populate — poll until stable
        prev_count = 0
        for i in range(10):
            page.wait_for_timeout(2000)
            count = page.eval_on_selector_all(
                'a[href*="/doi/10.1145/"]',
                "els => els.length"
            )
            log.info(f"t={2*(i+1)}s — {count} DOI links visible")
            if count == prev_count and i >= 3:
                # Stable for at least one cycle after minimum wait
                break
            prev_count = count

        links = page.eval_on_selector_all(
            'a[href*="/doi/10.1145/"]',
            """
            els => els.map(e => ({
                text: e.innerText.trim(),
                href: e.href
            }))
            """
        )

        # Save cookies before closing — captures Cloudflare clearance tokens
        cookies = context.cookies()
        browser.close()

    # Filter to valid paper DOI links only
    seen = set()
    paper_links = []

    for item in links:
        href = item["href"].strip()
        title = item["text"].strip()

        if not title:
            continue
        if href in seen:
            continue
        if "/doi/proceedings/" in href:
            continue
        if not re.search(r"/doi/10\.1145/\d+\.\d+$", href):
            continue

        seen.add(href)
        paper_links.append(href)

    log.info(f"Extracted {len(paper_links)} Research Track paper links")
    return sorted(paper_links), cookies


def fetch_acm_links(
    proceeding_url: str,
    conference: str,
    year: str,
) -> str:
    """
    Fetch Research Track paper links for an ACM proceedings and save as
    grouped_links.json. Also saves session cookies for browser_scraper.py.

    Args:
        proceeding_url: Full ACM DL proceedings URL
        conference:     Conference ID for output path, e.g. "ACM_KDD"
        year:           Conference year string, e.g. "2026"

    Returns:
        Absolute path to the saved grouped_links.json file.
    """
    log.info("Fetching ACM proceedings via Playwright (Research Track only)")

    paper_links, cookies = _fetch_research_track_links(proceeding_url)

    # Persist session cookies — browser_scraper.py loads these to avoid
    # re-challenging Cloudflare on every per-paper scrape
    COOKIE_PATH.parent.mkdir(parents=True, exist_ok=True)
    COOKIE_PATH.write_text(json.dumps(cookies), encoding="utf-8")
    log.info(f"Saved {len(cookies)} ACM session cookies → {COOKIE_PATH}")

    if not paper_links:
        raise RuntimeError(
            f"No paper links extracted from {proceeding_url}. "
            "Check that SESSION: Research Track is present on the page."
        )

    grouped_data = [
        {
            "track_title": "Research Track",
            "track_url": proceeding_url,
            "paper_links": paper_links
        }
    ]

    save_dir = Path(f"data/links_raw/{conference}/{year}")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "grouped_links.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(grouped_data, f, indent=2, ensure_ascii=False)

    print(f"[INFO] ✅ Extracted {len(paper_links)} ACM Research Track papers.")
    print(f"[INFO] 📁 Saved grouped links to: {save_path.resolve()}")

    return str(save_path.resolve())

def warmup_acm_cookies(proceeding_url: str) -> None:
    """
    Visit the ACM proceedings page to clear Cloudflare and save session
    cookies. Does not extract links — used by run_single_paper.py.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"]
        )
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        """)

        page.goto(proceeding_url, timeout=60000)
        page.wait_for_function(
            "() => document.title.includes('Proceedings')",
            timeout=60000
        )
        log.info(f"ACM warmup complete: {page.title()}")

        cookies = context.cookies()
        browser.close()

    COOKIE_PATH.parent.mkdir(parents=True, exist_ok=True)
    COOKIE_PATH.write_text(json.dumps(cookies), encoding="utf-8")
    log.info(f"Saved {len(cookies)} ACM session cookies → {COOKIE_PATH}")