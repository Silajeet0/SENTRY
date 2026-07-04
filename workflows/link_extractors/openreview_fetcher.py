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
        # Always process the spotlight tab first. This isn't required for
        # correctness anymore (see the recovery logic below, which now
        # verifies tab identity deterministically rather than guessing), but
        # keeps behavior predictable and matches how Cloudflare's post-
        # challenge reload tends to default back to this tab anyway.
        accept_tabs.sort(key=lambda t: (0 if "spotlight" in t.lower() else 1, t))
        log.info(f"Found {len(accept_tabs)} accept-related tabs: {accept_tabs}")

        def _find_tab_locator(tab_text: str):
            """Locator for the tab labeled tab_text, exact match."""
            return page.get_by_text(tab_text, exact=True).first

        def _is_tab_active(tab_text: str) -> bool:
            return page.evaluate("""
                (tabText) => {
                    const candidates = [...document.querySelectorAll('a, button, [role="tab"], li')];
                    return candidates.some(el => {
                        if (el.innerText.trim() !== tabText) return false;
                        const li = el.closest('li') || el;
                        return el.classList.contains('active')
                            || li.classList.contains('active')
                            || el.getAttribute('aria-selected') === 'true';
                    });
                }
            """, tab_text)

        def _click_next_arrow() -> str:
            """Clicks the › arrow in the active panel. Returns a status string."""
            return page.evaluate("""
                () => {
                    const panel = document.querySelector(
                        '.tab-pane.active.in, .tab-pane.active'
                    );
                    if (!panel) return 'no-panel';

                    const lis = [...panel.querySelectorAll('ul.pagination li')];
                    if (lis.length === 0) return 'no-arrow';

                    const rightArrows = lis.filter(li =>
                        (li.className || '').includes('right-arrow')
                    );
                    if (rightArrows.length === 0) return 'no-arrow';

                    const nextLi = rightArrows[0];
                    const cls = nextLi.className || '';
                    if (cls.includes('disabled')) return 'last-page';

                    const a = nextLi.querySelector('a[role="button"], a, button');
                    if (!a) return 'no-link';
                    a.click();
                    return 'clicked';
                }
            """)

        for tab_text in accept_tabs:
            # current_tab is the single source of truth for "which tab am I
            # supposed to be scraping right now" — used to re-find and
            # re-verify the correct tab after any Cloudflare recovery.
            current_tab = tab_text

            try:
                _find_tab_locator(current_tab).click(timeout=10000)
                log.info(f"  Clicked tab: {current_tab}")
            except Exception as e:
                log.debug(f"  Could not click tab '{current_tab}': {e}")
                continue

            # Wait for this tab's content to load — poll until the active
            # panel has at least one forum link or pagination appears.
            for _ in range(10):
                page.wait_for_timeout(500)
                ready = page.evaluate("""
                    () => {
                        const panel = document.querySelector('.tab-pane.active.in, .tab-pane.active');
                        if (!panel) return false;
                        return panel.querySelectorAll('a[href*="forum?id="]').length > 0
                            || panel.querySelectorAll('ul.pagination').length > 0;
                    }
                """)
                if ready:
                    break

            tab_ids_before = len(note_ids)
            # page_num tracks the page currently on screen that has NOT yet
            # been extracted this iteration. It only increments after a
            # successful extraction + successful next-click.
            page_num = 1

            while True:
                # All page.evaluate() calls are wrapped in a single try/except.
                # Cloudflare occasionally re-challenges mid-session (~every 6min
                # of continuous scraping), destroying the execution context.
                # On detection: wait for auto-resolution, re-find current_tab
                # from a freshly re-discovered tab list (the DOM is completely
                # new after the reload — old locators/attributes are gone),
                # confirm it's genuinely active, then replay (page_num - 1)
                # next-clicks to get back to the exact page we were on before
                # falling through to the top of this loop to re-extract it.
                try:
                    # Wait for page content to stabilise
                    prev_count = -1
                    for _ in range(10):
                        page.wait_for_timeout(700)
                        count = page.evaluate("""
                            () => {
                                const panel = document.querySelector('.tab-pane.active.in, .tab-pane.active');
                                if (!panel) return -1;
                                return panel.querySelectorAll('a[href*="forum?id="]').length;
                            }
                        """)
                        if count == prev_count and count >= 0:
                            break
                        prev_count = count

                    # OpenReview's own client-side rate limiter ("Too many
                    # requests...") kicks in after enough rapid pagination
                    # clicks and replaces the paper list AND the pagination
                    # widget with an error banner. It does not reliably clear
                    # on its own — waiting longer or polling for it to vanish
                    # can hang indefinitely. So: wait a flat 10s once, then
                    # just attempt the extraction/next-click below regardless
                    # of whether the banner text is still present.
                    rate_limited = page.evaluate("""
                        () => document.body.innerText.includes('Too many requests')
                    """)
                    if rate_limited:
                        log.warning(
                            f"  Rate limited by OpenReview on tab '{current_tab}' "
                            f"page {page_num} — waiting 10s, then re-scraping "
                            "this page and moving on"
                        )
                        page.wait_for_timeout(10000)

                    # Collect forum IDs visible in the active panel
                    ids_on_page = set(page.evaluate(r"""
                        () => {
                            const panel = document.querySelector('.tab-pane.active.in, .tab-pane.active');
                            if (!panel) return [];
                            return [...new Set(
                                [...panel.querySelectorAll('a[href*="forum?id="]')]
                                .map(a => {
                                    const m = a.href.match(/forum[?]id=([\w-]+)/);
                                    return m ? m[1] : null;
                                })
                                .filter(Boolean)
                            )];
                        }
                    """))

                    new_ids = [i for i in ids_on_page if i not in note_ids]
                    note_ids.update(new_ids)

                    log.info(
                        f"  Tab '{current_tab}' page {page_num}: "
                        f"+{len(new_ids)} new links | running total: {len(note_ids)}"
                    )

                    # Click › (first right-arrow) in the active panel pagination
                    next_result = 'retry'
                    for attempt in range(3):
                        next_result = _click_next_arrow()
                        if next_result != 'no-panel':
                            break
                        log.debug(f"  Panel gone during re-render, retrying ({attempt+1}/3)...")
                        page.wait_for_timeout(1000)

                    if next_result == 'clicked':
                        page.wait_for_timeout(1500)
                        page_num += 1
                    else:
                        log.info(
                            f"  Tab '{current_tab}' done after {page_num} page(s) "
                            f"(reason: {next_result}). "
                            f"New IDs this tab: {len(note_ids) - tab_ids_before}"
                        )
                        break

                except Exception as e:
                    err = str(e)
                    if ("Execution context was destroyed" in err
                            or "most likely because of a navigation" in err
                            or "navigation" in err.lower()):
                        log.warning(
                            f"  Cloudflare re-challenge on tab '{current_tab}' "
                            f"page {page_num} — waiting up to 30s for "
                            "auto-resolution..."
                        )
                        try:
                            # Wait for Cloudflare to resolve and the group page to reload
                            page.wait_for_selector("text=Accept", timeout=30000)
                            log.info(
                                "  Cloudflare cleared — re-discovering tabs and "
                                f"re-finding '{current_tab}'"
                            )

                            # The DOM is entirely new post-reload. Re-discover
                            # tab labels from scratch rather than trusting the
                            # old tab_text string still resolves to anything
                            # meaningful, then click the one matching current_tab.
                            fresh_tab_texts = page.evaluate("""
                                () => [...document.querySelectorAll('a, button, [role="tab"], li')]
                                    .map(el => el.innerText.trim())
                                    .filter(t => t.length > 0 && t.length < 60)
                            """)
                            if current_tab not in fresh_tab_texts:
                                log.error(
                                    f"  Tab '{current_tab}' no longer found on the "
                                    "reloaded page — stopping this tab early"
                                )
                                break

                            # Click current_tab and VERIFY — by the tab's own
                            # active state, not by any indirect signal like
                            # "does some panel have links" — that it's really
                            # the one now showing, retrying if a mis-click
                            # lands on whatever tab Cloudflare defaulted to.
                            reactivated = False
                            for attempt in range(5):
                                _find_tab_locator(current_tab).click(timeout=10000)
                                page.wait_for_timeout(2000)
                                if _is_tab_active(current_tab):
                                    reactivated = True
                                    log.info(f"  Tab '{current_tab}' re-activated successfully")
                                    break
                                log.warning(
                                    f"  Re-click attempt {attempt+1}/5 did not "
                                    f"activate '{current_tab}' — retrying"
                                )
                                page.wait_for_timeout(1500)

                            if not reactivated:
                                log.error(
                                    f"  Could not re-activate tab '{current_tab}' "
                                    "after Cloudflare recovery — stopping this "
                                    "tab early"
                                )
                                break

                            # Cloudflare's reload always redisplays page 1 of
                            # whichever tab is active. Replay (page_num - 1)
                            # next-clicks to get back to the exact page we were
                            # on, then fall through to the top of the while
                            # loop to re-extract it — no reliance on dedup
                            # skipping pages, this deterministically restores
                            # position via the counter itself.
                            pages_to_replay = page_num - 1
                            if pages_to_replay > 0:
                                log.info(
                                    f"  Replaying {pages_to_replay} next-click(s) "
                                    f"to return to page {page_num}"
                                )
                            for i in range(pages_to_replay):
                                page.wait_for_timeout(500)
                                replay_result = _click_next_arrow()
                                if replay_result != 'clicked':
                                    log.error(
                                        f"  Could not replay to page {page_num} "
                                        f"(stopped at replay step {i+1}/"
                                        f"{pages_to_replay}, reason: "
                                        f"{replay_result}) — resuming from "
                                        "whatever page is now displayed"
                                    )
                                    break
                                page.wait_for_timeout(1000)

                            # Fall through to the top of the while loop, which
                            # will extract whatever page is now on screen.
                            continue

                        except Exception as wait_err:
                            log.error(
                                f"  Cloudflare did not clear within 30s: {wait_err} "
                                "— stopping this tab early"
                            )
                            break
                    else:
                        raise

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