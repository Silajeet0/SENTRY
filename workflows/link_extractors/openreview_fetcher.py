"""
openreview_fetcher.py — Fetches accepted paper links for OpenReview-hosted
conferences via Playwright browser scraping.

Why Playwright instead of the OpenReview API
──────────────────────────────────────────────
api2.openreview.net returns a ChallengeRequiredError (a server-side bot
challenge, similar in spirit to Cloudflare) for unauthenticated programmatic
access — this affects both raw requests calls and the official openreview-py
client identically, since both hit the same API endpoint. There is no way to
solve this without a real browser.

The group page (openreview.net/group?id=<venue_id>) renders normally in a
browser and lists accepted papers under tabs like "Accept (oral)",
"Accept (spotlight)", "Accept (poster)" etc. We load this page with
Playwright, click each accept tab to reveal its papers, and collect the
forum links — the same general strategy used for ACM DL's session tabs.

Supported conferences:
    ICLR, ICML (any year hosted on OpenReview, using this group-page pattern)

─────────────────────────────────────────────────────────────────────────────
HOW TO ADD A NEW CONFERENCE / YEAR
─────────────────────────────────────────────────────────────────────────────
Add an entry to OPENREVIEW_CONFERENCE_CONFIG:

    "ICLR_2025": {
        "venue_id": "ICLR.cc/2025/Conference",
        # Substrings (case-insensitive) of tab labels to click and harvest.
        # Check the actual tab names by opening:
        #   https://openreview.net/group?id=<venue_id>
        # Tabs seen so far: "Accept (oral)", "Accept (spotlight)",
        # "Accept (poster)", "Accept (regular)", "Reject" (skip this one).
        "accept_tab_keywords": ["accept"],
    },
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import re
from pathlib import Path
from playwright.sync_api import sync_playwright

log = logging.getLogger(__name__)

# Where session cookies are saved for browser_scraper.py to reuse.
# Mirrors ACM_COOKIE_PATH's role in acm_api_fetcher.py exactly.
OPENREVIEW_COOKIE_PATH = Path("data/openreview_session_cookies.json")


OPENREVIEW_CONFERENCE_CONFIG: dict[str, dict] = {
    "ICLR_2025": {
        "venue_id": "ICLR.cc/2025/Conference",
        "accept_tab_keywords": ["accept"],
    },
    "ICLR_2026": {
        "venue_id": "ICLR.cc/2026/Conference",
        "accept_tab_keywords": ["accept"],
    },
    "ICML_2024": {
        "venue_id": "ICML.cc/2024/Conference",
        "accept_tab_keywords": ["accept"],
    },
    "ICML_2025": {
        "venue_id": "ICML.cc/2025/Conference",
        "accept_tab_keywords": ["accept"],
    },
    "ICML_2026": {
        "venue_id": "ICML.cc/2026/Conference",
        "accept_tab_keywords": ["accept"],
    }
}


def _get_config(conference: str, year: str) -> dict:
    key = f"{conference.upper()}_{year}"
    if key in OPENREVIEW_CONFERENCE_CONFIG:
        return OPENREVIEW_CONFERENCE_CONFIG[key]
    for k in OPENREVIEW_CONFERENCE_CONFIG:
        if key.startswith(k):
            return OPENREVIEW_CONFERENCE_CONFIG[k]
    raise ValueError(
        f"No OpenReview config found for '{conference}' {year}.\n"
        f"Add an entry to OPENREVIEW_CONFERENCE_CONFIG in openreview_fetcher.py.\n"
        f"Available: {list(OPENREVIEW_CONFERENCE_CONFIG.keys())}"
    )


def _fetch_accepted_paper_ids(venue_id: str, accept_tab_keywords: list[str]) -> list[str]:
    """
    Open the OpenReview group page in Playwright, click each "Accept ..." tab,
    and collect all forum note IDs from the rendered paper list links.

    Returns a sorted list of unique forum IDs (the ?id=XXX part of forum URLs).
    """
    from urllib.parse import quote
    group_url = f"https://openreview.net/group?id={quote(venue_id, safe='')}"

    note_ids: set[str] = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        """)

        log.info(f"Loading OpenReview group page: {group_url}")
        page.goto(group_url, wait_until="domcontentloaded", timeout=60000)

        # Wait for tabs to render — they appear as clickable elements with
        # role="tab" or similar, containing text like "Accept (oral)".
        try:
            page.wait_for_selector("text=Accept", timeout=30000)
        except Exception:
            log.warning(
                "No 'Accept' tab found within 30s — page may require a "
                "challenge to be solved manually, or tab labels differ."
            )

        # Discover all tabs on the page
        tab_texts = page.evaluate("""
            () => [...document.querySelectorAll('a, button, [role="tab"], li')]
                .map(el => el.innerText.trim())
                .filter(t => t.length > 0 && t.length < 60)
        """)
        accept_tabs = [
            t for t in set(tab_texts)
            if any(kw.lower() in t.lower() for kw in accept_tab_keywords)
            and "reject" not in t.lower()
        ]
        log.info(f"Found {len(accept_tabs)} accept-related tabs: {accept_tabs}")

        for tab_text in accept_tabs:
            try:
                # Click the tab by its visible text
                page.get_by_text(tab_text, exact=True).first.click(timeout=10000)
                log.info(f"  Clicked tab: {tab_text}")
            except Exception as e:
                log.debug(f"  Could not click tab '{tab_text}': {e}")
                continue

            # Wait for the paper list to render after the tab click —
            # poll until forum links stop increasing in count.
            prev_count = -1
            stable_rounds = 0
            for _ in range(15):
                page.wait_for_timeout(1000)
                links = page.evaluate("""
                    () => [...document.querySelectorAll('a[href*="forum?id="]')]
                        .map(a => a.href)
                """)
                count = len(set(links))
                if count == prev_count:
                    stable_rounds += 1
                    if stable_rounds >= 2:
                        break
                else:
                    stable_rounds = 0
                prev_count = count

            # Collect forum IDs found after this tab's content loaded
            links = page.evaluate("""
                () => [...document.querySelectorAll('a[href*="forum?id="]')]
                    .map(a => a.href)
            """)
            for href in links:
                match = re.search(r"forum\?id=([\w-]+)", href)
                if match:
                    note_ids.add(match.group(1))

            log.info(f"  Total unique forum IDs so far: {len(note_ids)}")

        # Save session cookies before closing — captures the same challenge
        # clearance browser_scraper.py needs for per-paper PDF downloads.
        # This is free: the browser session that just fetched links already
        # holds valid clearance cookies, so no extra warmup visit is needed.
        cookies = context.cookies()
        OPENREVIEW_COOKIE_PATH.parent.mkdir(parents=True, exist_ok=True)
        OPENREVIEW_COOKIE_PATH.write_text(json.dumps(cookies), encoding="utf-8")
        log.info(f"Saved {len(cookies)} OpenReview session cookies → {OPENREVIEW_COOKIE_PATH}")

        browser.close()

    return sorted(note_ids)


def fetch_openreview_links(
    conference: str,
    year: str,
) -> str:
    """
    Fetch accepted paper links for an OpenReview conference and save as
    grouped_links.json.

    Args:
        conference: Conference short name, e.g. "ICLR", "ICML", "NEURIPS"
        year:       Conference year string, e.g. "2025"

    Returns:
        Absolute path to the saved grouped_links.json file.
    """
    cfg = _get_config(conference, year)
    venue_id = cfg["venue_id"]
    accept_tab_keywords = cfg["accept_tab_keywords"]

    log.info(f"Fetching OpenReview links for {conference} {year} ({venue_id})")

    note_ids = _fetch_accepted_paper_ids(venue_id, accept_tab_keywords)

    if not note_ids:
        raise RuntimeError(
            f"No accepted papers found for {venue_id}.\n"
            "Check that the group page renders tabs correctly at:\n"
            f"https://openreview.net/group?id={venue_id}\n"
            "You may need to run with headless=False (default) and manually "
            "solve any challenge that appears, then re-run."
        )

    paper_links = [
        f"https://openreview.net/forum?id={note_id}"
        for note_id in note_ids
    ]

    grouped_data = [
        {
            "track_title": "Accepted Papers",
            "track_url": f"https://openreview.net/group?id={venue_id}",
            "paper_links": paper_links,
        }
    ]

    save_dir = Path(f"data/links_raw/{conference}/{year}")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "grouped_links.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(grouped_data, f, indent=2, ensure_ascii=False)

    print(f"[INFO] ✅ Extracted {len(paper_links)} accepted papers for {conference} {year}.")
    print(f"[INFO] 📁 Saved grouped links to: {save_path.resolve()}")

    return str(save_path.resolve())


def warmup_openreview_cookies(venue_id: str) -> None:
    """
    Visit an OpenReview group page to clear the challenge wall and save
    session cookies, without extracting links. Useful for refreshing stale
    cookies mid-run without re-fetching the full paper list — mirrors
    warmup_acm_cookies() in acm_api_fetcher.py.

    Args:
        venue_id: OpenReview venue ID, e.g. "ICML.cc/2025/Conference"
    """
    from urllib.parse import quote
    group_url = f"https://openreview.net/group?id={quote(venue_id, safe='')}"

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        """)

        page.goto(group_url, wait_until="domcontentloaded", timeout=60000)

        try:
            page.wait_for_selector("text=Accept", timeout=30000)
            log.info(f"OpenReview warmup complete: {page.title()}")
        except Exception:
            log.warning(
                "OpenReview warmup: 'Accept' tab not found within 30s — "
                "challenge page may still be showing. Solve it manually in "
                "the browser window if visible."
            )

        cookies = context.cookies()
        browser.close()

    OPENREVIEW_COOKIE_PATH.parent.mkdir(parents=True, exist_ok=True)
    OPENREVIEW_COOKIE_PATH.write_text(json.dumps(cookies), encoding="utf-8")
    log.info(f"Saved {len(cookies)} OpenReview session cookies → {OPENREVIEW_COOKIE_PATH}")
