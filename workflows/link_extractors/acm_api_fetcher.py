"""
acm_api_fetcher.py — Fetches paper links for ACM proceedings via Playwright.

Opens the ACM DL proceedings page in a non-headless browser (required to
pass Cloudflare Turnstile), navigates to each SESSION section via its
?tocHeading=headingN URL, extracts all paper DOI links, and saves session
cookies for reuse by browser_scraper.py.

Supported conferences (all A* under ACM):
    KDD, SIGMOD, SIGIR, SIGCOMM, STOC, FOCS, PODC, SOSP, OSDI,
    CCS, SIGGRAPH, CHI, PLDI, ASPLOS, ICSE, FSE, WWW, CSCW, UIST

─────────────────────────────────────────────────────────────────────────────
HOW ACM DL SESSION EXPANSION ACTUALLY WORKS
─────────────────────────────────────────────────────────────────────────────
Each SESSION heading in the main content area is an <a> element with:
    href="/doi/proceedings/10.1145/XXXXXX?tocHeading=headingN"
    class="section__title accordion-tabbed__control ..."
    aria-expanded="false"

Navigating to the ?tocHeading=headingN URL causes the server to return the
page with that section pre-expanded (papers visible in DOM). This is a full
page load per section — there is no in-page XHR.

The sidebar TOC contains visually identical <a> elements (same text, same
class prefix) but with bare href="#headingN" — those only scroll the page.
We read the full href of main-content anchors (those with ?tocHeading= in
their href) to get the correct navigation URLs.

─────────────────────────────────────────────────────────────────────────────
HOW TO ADD SUPPORT FOR A NEW CONFERENCE
─────────────────────────────────────────────────────────────────────────────
Every ACM DL proceedings page uses the same "SESSION: <name>" heading format.
To add a conference you only need to touch ONE place: CONFERENCE_SESSION_CONFIG
below. Add an entry with these fields:

    "ACM_YOURCONF": {
        # Sessions whose headings contain ANY of these substrings will be
        # SKIPPED. Case-insensitive. Leave as [] to harvest every session.
        "skip_sessions": ["Keynote", "Tutorial", "Workshop", "Demo", "Panel"],

        # If non-empty, ONLY sessions containing one of these substrings are
        # harvested — all others are skipped. Set to [] to include everything
        # not in skip_sessions. Extra keywords that match nothing are ignored.
        "include_only": [],
    },

After adding the entry, pass conference="ACM_YOURCONF" to fetch_acm_links()
and the rest of the pipeline adapts automatically.

To find the exact SESSION heading texts for a new conference:
  1. Open its dl.acm.org/doi/proceedings/... page in your browser.
  2. Open DevTools Console and run:
       [...document.querySelectorAll('a.accordion-tabbed__control[href*="tocHeading"]')]
         .map(el => el.innerText.trim())
  3. Add substrings of any you want to skip to "skip_sessions".
─────────────────────────────────────────────────────────────────────────────
"""

import re
import json
import logging
from pathlib import Path
from playwright.sync_api import sync_playwright

log = logging.getLogger(__name__)

# Where session cookies are saved for browser_scraper.py to reuse
COOKIE_PATH = Path("data/acm_session_cookies.json")


# ─────────────────────────────────────────────────────────────────────────────
# Per-conference configuration
# ─────────────────────────────────────────────────────────────────────────────
CONFERENCE_SESSION_CONFIG: dict[str, dict] = {
    # ------------------------------------------------------------------
    # KDD — single "Research Track" session; everything else is skipped.
    # ------------------------------------------------------------------
    "ACM_KDD": {
        "skip_sessions": [],
        "include_only": ["Research Track"],
    },

    # ------------------------------------------------------------------
    # CCS — many "Session A1/B2/..." slots; skip only keynote.
    # ------------------------------------------------------------------
    "ACM_CCS": {
        "skip_sessions": ["Keynote"],
        "include_only": [],
    },

    # ------------------------------------------------------------------
    # SIGMOD — only Industry Papers session wanted.
    # ------------------------------------------------------------------
    "ACM_SIGMOD": {
        "skip_sessions": ["Keynote", "Demo", "Panel", "Tutorial", "Workshop"],
        "include_only": ["Industry Papers"],
    },

    # ------------------------------------------------------------------
    # SIGGRAPH — all technical paper sessions.
    # ------------------------------------------------------------------
    "ACM_SIGGRAPH": {
        "skip_sessions": ["Keynote", "Course", "Talk", "Panel", "Poster"],
        "include_only": [],
    },

    # ------------------------------------------------------------------
    # SIGCOMM — all technical sessions.
    # ------------------------------------------------------------------
    "ACM_SIGCOMM": {
        "skip_sessions": ["Keynote", "Workshop", "Tutorial", "Panel", "Poster"],
        "include_only": [],
    },
}

# Fallback when conference is not in CONFERENCE_SESSION_CONFIG.
_DEFAULT_CONFIG: dict = {
    "skip_sessions": ["Keynote", "Tutorial", "Workshop", "Demo", "Panel", "Poster"],
    "include_only": [],
}


def _get_conference_config(conference: str) -> dict:
    """Return the session config for *conference*, falling back to defaults."""
    key = conference.upper()
    if key in CONFERENCE_SESSION_CONFIG:
        return CONFERENCE_SESSION_CONFIG[key]
    for k in CONFERENCE_SESSION_CONFIG:
        if key.startswith(k):
            return CONFERENCE_SESSION_CONFIG[k]
    log.warning(
        f"No session config found for '{conference}'. "
        "Using default (skipping Keynote/Tutorial/Workshop/Demo/Panel/Poster). "
        "Add an entry to CONFERENCE_SESSION_CONFIG to customise behaviour."
    )
    return _DEFAULT_CONFIG


def _session_is_wanted(heading_text: str, cfg: dict) -> bool:
    """
    Return True if a SESSION heading should be harvested.

    Applies include_only first (whitelist), then skip_sessions (blacklist).
    Both are case-insensitive substring matches. Extra keywords that match
    nothing are silently ignored.
    """
    text_lower = heading_text.lower()
    if cfg["include_only"]:
        return any(s.lower() in text_lower for s in cfg["include_only"])
    return not any(s.lower() in text_lower for s in cfg["skip_sessions"])


def _wait_for_page_ready(page) -> None:
    """
    Block until Cloudflare challenge is cleared and the real ACM DL
    proceedings page is loaded.

    Stage 1 — wait until title is no longer "Just a moment..." (up to 120s).
    Stage 2 — wait for main-content accordion anchors to be in the DOM.
    """
    log.info("Waiting for Cloudflare challenge to clear (solve in browser if prompted)...")

    page.wait_for_function(
        """() => {
            const t = document.title.toLowerCase();
            return t.length > 5 && !t.includes('just a moment');
        }""",
        timeout=120000,
    )

    try:
        page.wait_for_selector(
            "a.accordion-tabbed__control[href*='tocHeading']",
            state="attached",
            timeout=30000,
        )
    except Exception:
        log.warning(
            "Main-content accordion anchors not found after page load — "
            f"title: '{page.title()}'. "
            "Page may be flat (no collapsible sections) or still loading."
        )

    log.info(f"Proceedings page ready — title: {page.title()}")


def _extract_paper_links(page) -> list[dict]:
    """Return all DOI anchor elements on the current page as {text, href}."""
    return page.eval_on_selector_all(
        'a[href*="/doi/10.1145/"]',
        "els => els.map(e => ({text: e.innerText.trim(), href: e.href}))",
    )


def _filter_paper_links(raw: list[dict], seen: set) -> list[str]:
    """
    Filter raw anchor dicts to genuine paper DOI URLs, deduplicating
    against *seen* (updated in place).
    """
    result = []
    for item in raw:
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
        result.append(href)
    return result


def _fetch_conference_links(
    proceeding_url: str,
    conference: str,
) -> tuple[list[str], list[dict]]:
    """
    Open the ACM DL proceedings page (non-headless to pass Cloudflare),
    navigate to each wanted SESSION's ?tocHeading=headingN URL to load its
    papers, collect all DOI links, and save cookies.

    Returns:
        (paper_links, cookies) — sorted list of DOI URLs and session cookies.
    """
    cfg = _get_conference_config(conference)
    cookies: list[dict] = []
    seen: set[str] = set()
    all_paper_links: list[str] = []

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

        page.goto(proceeding_url, timeout=120000)
        _wait_for_page_ready(page)

        # ── Step 1: collect papers already visible on the landing page ───────
        # The first session is pre-expanded on load — capture it now.
        landing_links = _extract_paper_links(page)
        new_links = _filter_paper_links(landing_links, seen)
        if new_links:
            log.info(f"Landing page: {len(new_links)} paper links")
            all_paper_links.extend(new_links)

        # ── Step 2: discover all SESSION headings with tocHeading URLs ────────
        # Query only main-content accordion anchors — href contains "tocHeading".
        # Sidebar anchors have bare "#headingN" hrefs and are excluded.
        session_entries: list[dict] = page.evaluate("""
            () => [
                ...document.querySelectorAll(
                    'a.accordion-tabbed__control[href*="tocHeading"]'
                )
            ].map(el => ({
                text: el.innerText.trim(),
                href: el.href,
                expanded: el.getAttribute('aria-expanded')
            }))
        """)

        log.info(f"Found {len(session_entries)} SESSION headings in main content")

        wanted = [s for s in session_entries if _session_is_wanted(s["text"], cfg)]
        skipped = [s for s in session_entries if not _session_is_wanted(s["text"], cfg)]

        log.info(f"Wanting {len(wanted)} sessions, skipping {len(skipped)}")
        for s in skipped:
            log.debug(f"  SKIP: {s['text']}")

        # ── Step 3: navigate to each wanted session's tocHeading URL ─────────
        # Each navigation is a full page load with that section pre-expanded.
        # We skip sections already expanded on the landing page (aria-expanded
        # = "true") since their links were already captured in Step 1.
        for entry in wanted:
            if entry["expanded"] == "true":
                log.debug(f"  already expanded on landing: {entry['text']}")
                continue

            log.info(f"  Loading: {entry['text']}")
            page.goto(entry["href"], wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(2000)

            raw = _extract_paper_links(page)
            new_links = _filter_paper_links(raw, seen)
            log.info(f"    → {len(new_links)} new paper links")
            all_paper_links.extend(new_links)

        # ── Step 4: save cookies after all navigation ─────────────────────────
        # Cloudflare clearance cookies persist across same-domain navigations.
        cookies = context.cookies()
        browser.close()

    log.info(f"Extracted {len(all_paper_links)} paper links total")
    return sorted(all_paper_links), cookies


def fetch_acm_links(
    proceeding_url: str,
    conference: str,
    year: str,
) -> str:
    """
    Fetch paper links for an ACM proceedings and save as grouped_links.json.
    Also saves session cookies for browser_scraper.py.

    Args:
        proceeding_url: Full ACM DL proceedings URL.
        conference:     Conference ID, e.g. "ACM_KDD", "ACM_CCS", "ACM_SIGCOMM".
        year:           Conference year string, e.g. "2025".

    Returns:
        Absolute path to the saved grouped_links.json file.
    """
    log.info(f"Fetching ACM proceedings for {conference} {year} via Playwright")

    paper_links, cookies = _fetch_conference_links(proceeding_url, conference)

    COOKIE_PATH.parent.mkdir(parents=True, exist_ok=True)
    COOKIE_PATH.write_text(json.dumps(cookies), encoding="utf-8")
    log.info(f"Saved {len(cookies)} ACM session cookies → {COOKIE_PATH}")

    if not paper_links:
        cfg = _get_conference_config(conference)
        raise RuntimeError(
            f"No paper links extracted from {proceeding_url}.\n"
            f"Conference config used: include_only={cfg['include_only']}, "
            f"skip_sessions={cfg['skip_sessions']}.\n"
            "Check that the expected SESSION headings exist on the page, "
            "or update CONFERENCE_SESSION_CONFIG for this conference."
        )

    cfg = _get_conference_config(conference)
    track_label = (
        ", ".join(cfg["include_only"]) if cfg["include_only"] else "All Sessions"
    )
    grouped_data = [
        {
            "track_title": track_label,
            "track_url": proceeding_url,
            "paper_links": paper_links,
        }
    ]

    save_dir = Path(f"data/links_raw/{conference}/{year}")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "grouped_links.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(grouped_data, f, indent=2, ensure_ascii=False)

    print(f"[INFO] ✅ Extracted {len(paper_links)} papers for {conference} {year}.")
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
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        """)

        page.goto(proceeding_url, timeout=60000)

        log.info("Warmup: waiting for Cloudflare challenge to clear...")
        page.wait_for_function(
            """() => {
                const t = document.title.toLowerCase();
                return t.length > 5 && !t.includes('just a moment');
            }""",
            timeout=120000,
        )
        log.info(f"ACM warmup complete: {page.title()}")

        cookies = context.cookies()
        browser.close()

    COOKIE_PATH.parent.mkdir(parents=True, exist_ok=True)
    COOKIE_PATH.write_text(json.dumps(cookies), encoding="utf-8")
    log.info(f"Saved {len(cookies)} ACM session cookies → {COOKIE_PATH}")